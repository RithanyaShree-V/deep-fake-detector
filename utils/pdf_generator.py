"""
PDF Report Generator for DeepFake Detection System
===================================================
Generates professional PDF reports for analysis results.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle,
    Image,
    PageBreak,
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from config import Config


def generate_report_pdf(result_data: Dict, output_path: str) -> str:
    """
    Generate a PDF report for analysis results.
    
    Args:
        result_data: Dictionary containing analysis results
        output_path: Path to save the PDF file
        
    Returns:
        Path to generated PDF file
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#6366f1'),
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#1f2937')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#4b5563')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("DeepFake Detection Report", title_style))
    content.append(Spacer(1, 12))
    
    # Horizontal line
    content.append(HRFlowable(
        width="100%",
        thickness=2,
        color=colors.HexColor('#6366f1'),
        spaceBefore=10,
        spaceAfter=20
    ))
    
    # Report metadata
    content.append(Paragraph("Report Information", heading_style))
    
    metadata_data = [
        ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Report Type:", result_data.get('type', 'Unknown').capitalize()],
        ["Filename:", result_data.get('filename', 'N/A')]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(metadata_table)
    content.append(Spacer(1, 20))
    
    # Handle different result types
    if result_data.get('type') == 'batch':
        content.extend(_generate_batch_report(result_data, styles, heading_style, body_style))
    elif result_data.get('type') == 'video':
        content.extend(_generate_video_report(result_data, styles, heading_style, subheading_style, body_style))
    else:
        content.extend(_generate_image_report(result_data, styles, heading_style, subheading_style, body_style))
    
    # Disclaimer
    content.append(Spacer(1, 30))
    content.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor('#e5e7eb'),
        spaceBefore=10,
        spaceAfter=10
    ))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_JUSTIFY,
        spaceBefore=10
    )
    
    disclaimer_text = """
    <b>DISCLAIMER:</b> This report provides probabilistic assessments based on AI analysis and should not be 
    considered legally definitive. Results are for informational purposes only. The detection system uses 
    machine learning models that may produce false positives or false negatives. Always verify findings 
    through multiple sources and professional analysis when making important decisions. The developers 
    are not responsible for any misuse or misinterpretation of results.
    """
    content.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Footer
    footer_text = "© 2026 DeepFake Detector - Built for responsible AI use"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#9ca3af'),
        alignment=TA_CENTER,
        spaceBefore=20
    )
    content.append(Paragraph(footer_text, footer_style))
    
    # Build PDF
    doc.build(content)
    
    return output_path


def _generate_image_report(
    result_data: Dict, 
    styles, 
    heading_style, 
    subheading_style, 
    body_style
) -> List:
    """Generate content for image analysis report."""
    content = []
    result = result_data.get('result', {})
    
    # Analysis Results
    content.append(Paragraph("Analysis Results", heading_style))
    
    # Prediction box
    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0)
    
    # Color based on prediction
    pred_color = _get_prediction_color(prediction)
    
    prediction_data = [
        [Paragraph(f"<b>Prediction:</b> {prediction}", body_style)],
        [Paragraph(f"<b>Confidence:</b> {confidence}%", body_style)]
    ]
    
    pred_table = Table(prediction_data, colWidths=[5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
        ('BOX', (0, 0), (-1, -1), 2, pred_color),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    content.append(pred_table)
    content.append(Spacer(1, 15))
    
    # Probability Scores
    content.append(Paragraph("Probability Scores", subheading_style))
    
    probabilities = result.get('probabilities', {})
    if probabilities:
        prob_data = [["Model", "Score"]]
        for label, value in probabilities.items():
            prob_data.append([label, f"{value}%"])
        
        prob_table = Table(prob_data, colWidths=[3.5*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(prob_table)
    content.append(Spacer(1, 15))
    
    # Face Detection
    content.append(Paragraph("Face Detection", subheading_style))
    face_detected = result.get('face_detected', False)
    face_regions = result.get('face_regions', [])
    
    if face_detected:
        content.append(Paragraph(f"✓ {len(face_regions)} face(s) detected and analyzed", body_style))
    else:
        content.append(Paragraph("✗ No faces detected in the image", body_style))
    content.append(Spacer(1, 15))
    
    # Explanation
    content.append(Paragraph("Analysis Explanation", subheading_style))
    explanation = result.get('explanation', 'No explanation available.')
    content.append(Paragraph(explanation, body_style))
    content.append(Spacer(1, 15))
    
    # Detected Artifacts
    artifacts = result.get('artifacts', [])
    if artifacts:
        content.append(Paragraph("Detected Artifacts", subheading_style))
        for artifact in artifacts:
            content.append(Paragraph(f"• {artifact}", body_style))
    
    return content


def _generate_video_report(
    result_data: Dict, 
    styles, 
    heading_style, 
    subheading_style, 
    body_style
) -> List:
    """Generate content for video analysis report."""
    content = []
    result = result_data.get('result', {})
    
    # Analysis Results
    content.append(Paragraph("Video Analysis Results", heading_style))
    
    # Overall prediction
    prediction = result.get('overall_prediction', 'Unknown')
    confidence = result.get('overall_confidence', 0)
    pred_color = _get_prediction_color(prediction)
    
    prediction_data = [
        [Paragraph(f"<b>Overall Prediction:</b> {prediction}", body_style)],
        [Paragraph(f"<b>Confidence:</b> {confidence}%", body_style)]
    ]
    
    pred_table = Table(prediction_data, colWidths=[5*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
        ('BOX', (0, 0), (-1, -1), 2, pred_color),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    content.append(pred_table)
    content.append(Spacer(1, 15))
    
    # Video Statistics
    content.append(Paragraph("Video Statistics", subheading_style))
    
    video_info = result.get('video_info', {})
    stats_data = [
        ["Metric", "Value"],
        ["Total Frames", str(result.get('total_frames', 'N/A'))],
        ["Frames Analyzed", str(result.get('frames_analyzed', 'N/A'))],
        ["Suspicious Frames", str(result.get('suspicious_frames', 'N/A'))],
        ["Suspicious Rate", f"{result.get('suspicious_percentage', 0)}%"],
        ["Duration", f"{video_info.get('duration_seconds', 'N/A')} seconds"],
        ["FPS", str(video_info.get('fps', 'N/A'))],
        ["Resolution", video_info.get('resolution', 'N/A')]
    ]
    
    stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(stats_table)
    content.append(Spacer(1, 15))
    
    # Summary
    content.append(Paragraph("Analysis Summary", subheading_style))
    summary = result.get('summary', 'No summary available.')
    content.append(Paragraph(summary, body_style))
    content.append(Spacer(1, 15))
    
    # Frame-by-frame results (limited)
    frame_results = result.get('frame_results', [])
    if frame_results:
        content.append(Paragraph("Frame Analysis (Sample)", subheading_style))
        
        frame_data = [["Frame", "Prediction", "Confidence", "Deepfake Score", "AI Score"]]
        
        # Show first 20 frames
        for frame in frame_results[:20]:
            frame_data.append([
                str(frame.get('frame_index', '')),
                frame.get('prediction', ''),
                f"{frame.get('confidence', 0)}%",
                f"{frame.get('deepfake_score', 0)}%",
                f"{frame.get('ai_generated_score', 0)}%"
            ])
        
        if len(frame_results) > 20:
            frame_data.append(['...', f'({len(frame_results) - 20} more frames)', '', '', ''])
        
        frame_table = Table(frame_data, colWidths=[0.8*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
        frame_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(frame_table)
    
    return content


def _generate_batch_report(
    result_data: Dict, 
    styles, 
    heading_style, 
    body_style
) -> List:
    """Generate content for batch analysis report."""
    content = []
    results = result_data.get('results', [])
    
    # Summary
    content.append(Paragraph("Batch Analysis Summary", heading_style))
    
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    
    summary_data = [
        ["Metric", "Count"],
        ["Total Files", str(total)],
        ["Successful", str(successful)],
        ["Failed", str(failed)]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(summary_table)
    content.append(Spacer(1, 20))
    
    # Individual Results
    content.append(Paragraph("Individual Results", heading_style))
    
    results_data = [["Filename", "Type", "Prediction", "Confidence"]]
    
    for item in results:
        if item.get('success'):
            result = item.get('result', {})
            prediction = result.get('prediction') or result.get('overall_prediction', 'N/A')
            confidence = result.get('confidence') or result.get('overall_confidence', 0)
            results_data.append([
                item.get('filename', 'Unknown')[:30],
                item.get('type', 'N/A').capitalize(),
                prediction,
                f"{confidence}%"
            ])
        else:
            results_data.append([
                item.get('filename', 'Unknown')[:30],
                'N/A',
                'Error',
                item.get('error', 'Unknown error')[:20]
            ])
    
    results_table = Table(results_data, colWidths=[2*inch, 1*inch, 1.2*inch, 1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    content.append(results_table)
    
    return content


def _get_prediction_color(prediction: str) -> colors.Color:
    """Get color based on prediction."""
    prediction_lower = prediction.lower()
    if prediction_lower == 'real':
        return colors.HexColor('#10b981')  # Green
    elif prediction_lower == 'deepfake':
        return colors.HexColor('#ef4444')  # Red
    elif 'ai' in prediction_lower or 'generated' in prediction_lower:
        return colors.HexColor('#f59e0b')  # Orange
    else:
        return colors.HexColor('#6b7280')  # Gray


if __name__ == "__main__":
    # Test PDF generation
    test_result = {
        'type': 'image',
        'filename': 'test_image.jpg',
        'result': {
            'prediction': 'Deepfake',
            'confidence': 87.5,
            'probabilities': {
                'Real (Deepfake Model)': 12.5,
                'Deepfake': 87.5,
                'Real (AI Model)': 45.2,
                'AI-Generated': 54.8
            },
            'face_detected': True,
            'face_regions': [{'box': [100, 50, 300, 280], 'confidence': 0.98}],
            'explanation': 'The model detected potential face manipulation with 87.5% confidence.',
            'artifacts': [
                'High probability of facial manipulation',
                'Check for unnatural blending around face edges',
                'Look for inconsistent lighting on face'
            ]
        }
    }
    
    output_path = 'test_report.pdf'
    generate_report_pdf(test_result, output_path)
    print(f"Test report generated: {output_path}")
