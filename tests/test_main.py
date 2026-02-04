"""
Test Suite for DeepFake Detection System
=========================================
Unit tests for all core components.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    """Tests for configuration module."""
    
    def test_config_exists(self):
        """Test that config can be imported."""
        from config import Config
        self.assertIsNotNone(Config)
    
    def test_allowed_extensions(self):
        """Test file extension validation."""
        from config import Config
        
        # Images
        self.assertTrue(Config.is_allowed_file('test.jpg', 'image'))
        self.assertTrue(Config.is_allowed_file('test.PNG', 'image'))
        self.assertFalse(Config.is_allowed_file('test.gif', 'image'))
        
        # Videos
        self.assertTrue(Config.is_allowed_file('test.mp4', 'video'))
        self.assertTrue(Config.is_allowed_file('test.AVI', 'video'))
        self.assertFalse(Config.is_allowed_file('test.mkv', 'video'))
    
    def test_get_file_type(self):
        """Test file type detection."""
        from config import Config
        
        self.assertEqual(Config.get_file_type('image.jpg'), 'image')
        self.assertEqual(Config.get_file_type('video.mp4'), 'video')
        self.assertEqual(Config.get_file_type('file.txt'), 'unknown')


class TestModelLoader(unittest.TestCase):
    """Tests for model loading module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        import torch
        cls.torch = torch
        cls.device = torch.device('cpu')
    
    def test_deepfake_detector_architecture(self):
        """Test DeepFake detector model architecture."""
        from model_loader import DeepFakeDetector
        
        model = DeepFakeDetector(num_classes=2, pretrained=False)
        
        # Test forward pass
        x = self.torch.randn(1, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 2))
    
    def test_ai_generated_detector_architecture(self):
        """Test AI-Generated detector model architecture."""
        from model_loader import AIGeneratedDetector
        
        model = AIGeneratedDetector(num_classes=2, pretrained=False)
        
        # Test forward pass
        x = self.torch.randn(1, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 2))
    
    def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        from model_loader import ModelLoader
        
        loader = ModelLoader(device='cpu')
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.device.type, 'cpu')
        self.assertEqual(len(loader.available_models), 2)


class TestInference(unittest.TestCase):
    """Tests for inference module."""
    
    def test_detection_result_dataclass(self):
        """Test DetectionResult dataclass."""
        from inference import DetectionResult
        
        result = DetectionResult(
            prediction='Deepfake',
            confidence=0.85
        )
        
        self.assertEqual(result.prediction, 'Deepfake')
        self.assertEqual(result.confidence, 0.85)
        
        # Test to_dict
        result_dict = result.to_dict()
        self.assertEqual(result_dict['prediction'], 'Deepfake')
        self.assertEqual(result_dict['confidence'], 85.0)  # Converted to percentage
    
    def test_video_analysis_result_dataclass(self):
        """Test VideoAnalysisResult dataclass."""
        from inference import VideoAnalysisResult
        
        result = VideoAnalysisResult(
            overall_prediction='Real',
            overall_confidence=0.92,
            frames_analyzed=50,
            total_frames=500
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict['frames_analyzed'], 50)
        self.assertEqual(result_dict['overall_confidence'], 92.0)
    
    def test_image_preprocessor(self):
        """Test image preprocessing."""
        from inference import ImagePreprocessor
        import numpy as np
        
        preprocessor = ImagePreprocessor(image_size=224)
        
        # Test with numpy array
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocessor.preprocess(image)
        
        self.assertEqual(tensor.shape, (1, 3, 224, 224))


class TestPDFGenerator(unittest.TestCase):
    """Tests for PDF report generation."""
    
    def test_pdf_generation(self):
        """Test PDF report generation."""
        from utils.pdf_generator import generate_report_pdf
        
        test_result = {
            'type': 'image',
            'filename': 'test.jpg',
            'result': {
                'prediction': 'Real',
                'confidence': 95.0,
                'probabilities': {'Real': 95.0, 'Fake': 5.0},
                'face_detected': True,
                'face_regions': [],
                'explanation': 'Test explanation',
                'artifacts': ['Test artifact']
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = f.name
        
        try:
            result_path = generate_report_pdf(test_result, output_path)
            
            self.assertTrue(os.path.exists(result_path))
            self.assertGreater(os.path.getsize(result_path), 0)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestFlaskApp(unittest.TestCase):
    """Tests for Flask application."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from app import app
        app.config['TESTING'] = True
        cls.client = app.test_client()
    
    def test_index_page(self):
        """Test index page loads."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_about_page(self):
        """Test about page loads."""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
    
    def test_api_docs_page(self):
        """Test API docs page loads."""
        response = self.client.get('/api-docs')
        self.assertEqual(response.status_code, 200)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_status_endpoint(self):
        """Test status endpoint."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertEqual(data['status'], 'online')
        self.assertIn('supported_formats', data)
    
    def test_analyze_no_file(self):
        """Test analyze endpoint without file."""
        response = self.client.post('/api/analyze/image')
        self.assertEqual(response.status_code, 400)
        
        data = response.get_json()
        self.assertFalse(data['success'])


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestInference))
    suite.addTests(loader.loadTestsFromTestCase(TestPDFGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestFlaskApp))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
