"""
Flask Application - DeepFake Detection System
==============================================
Main web application with REST API endpoints for:
- Image upload and analysis
- Video upload and analysis
- Batch processing
- PDF report generation
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
import traceback

from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify, 
    send_file,
    url_for,
    session
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config, get_config
from inference import get_inference_engine, DetectionResult, VideoAnalysisResult
from utils.pdf_generator import generate_report_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(get_config())
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize folders
Config.init_folders()

# Global inference engine (lazy loaded)
_engine = None


def get_engine():
    """Get or create inference engine."""
    global _engine
    if _engine is None:
        logger.info("Initializing inference engine...")
        _engine = get_inference_engine()
        logger.info("Inference engine ready")
    return _engine


def handle_errors(f):
    """Decorator for error handling."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'message': 'An error occurred during processing'
            }), 500
    return wrapper


# ============================================
# Web Routes
# ============================================

@app.route('/')
def index():
    """Home page with upload interface."""
    return render_template('index.html')


@app.route('/results')
def results():
    """Results page."""
    return render_template('results.html')


@app.route('/about')
def about():
    """About page with information and disclaimer."""
    return render_template('about.html')


@app.route('/api-docs')
def api_docs():
    """API documentation page."""
    return render_template('api_docs.html')


# ============================================
# API Routes - Image Analysis
# ============================================

@app.route('/api/analyze/image', methods=['POST'])
@handle_errors
def analyze_image():
    """
    Analyze uploaded image for deepfake/AI-generated content.
    
    Request:
        - file: Image file (JPG, PNG, WEBP)
        - generate_heatmap: bool (optional, default: true)
    
    Response:
        - success: bool
        - result: Detection result object
    """
    # Validate file
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if not Config.is_allowed_file(file.filename, 'image'):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_IMAGE_EXTENSIONS)}'
        }), 400
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = Config.UPLOAD_FOLDER / unique_filename
    file.save(str(filepath))
    
    logger.info(f"Processing image: {filename}")
    
    # Get options
    generate_heatmap = request.form.get('generate_heatmap', 'true').lower() == 'true'
    
    # Analyze image
    engine = get_engine()
    result = engine.analyze_image(str(filepath), generate_heatmap=generate_heatmap)
    
    # Store result in session for report generation
    session['last_result'] = {
        'type': 'image',
        'filename': filename,
        'result': result.to_dict(),
        'filepath': str(filepath)
    }
    
    logger.info(f"Analysis complete: {result.prediction} ({result.confidence*100:.1f}%)")
    
    return jsonify({
        'success': True,
        'result': result.to_dict(),
        'filename': filename
    })


# ============================================
# API Routes - Video Analysis
# ============================================

@app.route('/api/analyze/video', methods=['POST'])
@handle_errors
def analyze_video():
    """
    Analyze uploaded video for deepfake content.
    
    Request:
        - file: Video file (MP4, AVI, MOV, WEBM)
        - sample_rate: int (optional)
        - max_frames: int (optional)
    
    Response:
        - success: bool
        - result: Video analysis result object
    """
    # Validate file
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if not Config.is_allowed_file(file.filename, 'video'):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_VIDEO_EXTENSIONS)}'
        }), 400
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = Config.UPLOAD_FOLDER / unique_filename
    file.save(str(filepath))
    
    logger.info(f"Processing video: {filename}")
    
    # Get options
    sample_rate = int(request.form.get('sample_rate', Config.VIDEO_SAMPLE_RATE))
    max_frames = int(request.form.get('max_frames', Config.MAX_FRAMES_PER_VIDEO))
    
    # Analyze video
    engine = get_engine()
    result = engine.analyze_video(
        str(filepath), 
        sample_rate=sample_rate,
        max_frames=max_frames
    )
    
    # Store result in session
    session['last_result'] = {
        'type': 'video',
        'filename': filename,
        'result': result.to_dict(),
        'filepath': str(filepath)
    }
    
    logger.info(f"Video analysis complete: {result.overall_prediction}")
    
    return jsonify({
        'success': True,
        'result': result.to_dict(),
        'filename': filename
    })


# ============================================
# API Routes - Batch Processing
# ============================================

@app.route('/api/analyze/batch', methods=['POST'])
@handle_errors
def analyze_batch():
    """
    Analyze multiple files in batch.
    
    Request:
        - files[]: Multiple image/video files
    
    Response:
        - success: bool
        - results: List of analysis results
    """
    if 'files[]' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        }), 400
    
    files = request.files.getlist('files[]')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({
            'success': False,
            'error': 'No files selected'
        }), 400
    
    engine = get_engine()
    results = []
    
    for file in files:
        if file.filename == '':
            continue
        
        filename = secure_filename(file.filename)
        file_type = Config.get_file_type(filename)
        
        if file_type == 'unknown':
            results.append({
                'filename': filename,
                'success': False,
                'error': 'Unsupported file type'
            })
            continue
        
        # Save file
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = Config.UPLOAD_FOLDER / unique_filename
        file.save(str(filepath))
        
        try:
            if file_type == 'image':
                result = engine.analyze_image(str(filepath))
                results.append({
                    'filename': filename,
                    'success': True,
                    'type': 'image',
                    'result': result.to_dict()
                })
            else:
                result = engine.analyze_video(str(filepath))
                results.append({
                    'filename': filename,
                    'success': True,
                    'type': 'video',
                    'result': result.to_dict()
                })
        except Exception as e:
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
    
    # Store batch results
    session['batch_results'] = results
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results),
        'successful': sum(1 for r in results if r.get('success', False))
    })


# ============================================
# API Routes - Report Generation
# ============================================

@app.route('/api/report/generate', methods=['POST'])
@handle_errors
def generate_report():
    """
    Generate PDF report for the last analysis.
    
    Response:
        - PDF file download
    """
    last_result = session.get('last_result')
    
    if not last_result:
        return jsonify({
            'success': False,
            'error': 'No analysis result available'
        }), 400
    
    # Generate report
    report_filename = f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = Config.RESULTS_FOLDER / report_filename
    
    generate_report_pdf(
        result_data=last_result,
        output_path=str(report_path)
    )
    
    return send_file(
        str(report_path),
        as_attachment=True,
        download_name=report_filename,
        mimetype='application/pdf'
    )


@app.route('/api/report/batch', methods=['POST'])
@handle_errors
def generate_batch_report():
    """Generate PDF report for batch analysis."""
    batch_results = session.get('batch_results')
    
    if not batch_results:
        return jsonify({
            'success': False,
            'error': 'No batch results available'
        }), 400
    
    report_filename = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = Config.RESULTS_FOLDER / report_filename
    
    generate_report_pdf(
        result_data={'type': 'batch', 'results': batch_results},
        output_path=str(report_path)
    )
    
    return send_file(
        str(report_path),
        as_attachment=True,
        download_name=report_filename,
        mimetype='application/pdf'
    )


# ============================================
# API Routes - Status & Info
# ============================================

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get API status and system info."""
    import torch
    
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'models_loaded': _engine is not None,
        'supported_formats': {
            'images': list(Config.ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(Config.ALLOWED_VIDEO_EXTENSIONS)
        },
        'max_file_size_mb': Config.MAX_CONTENT_LENGTH / (1024 * 1024)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': 'Bad request'}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404


@app.errorhandler(413)
def file_too_large(e):
    max_size = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify({
        'success': False, 
        'error': f'File too large. Maximum size: {max_size}MB'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepFake Detection Web App')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--preload', action='store_true', help='Preload models on startup')
    
    args = parser.parse_args()
    
    # Preload models if requested
    if args.preload:
        logger.info("Preloading models...")
        get_engine()
    
    # Run app
    logger.info(f"Starting DeepFake Detection Server on {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
