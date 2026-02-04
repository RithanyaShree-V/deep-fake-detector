"""
Configuration Module for DeepFake Detection System
===================================================
Centralized configuration management for the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class."""
    
    # Base Paths
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Upload Settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 104857600))  # 100MB
    
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}
    
    # Model Settings
    MODEL_PATH = BASE_DIR / 'models'
    USE_GPU = os.getenv('USE_GPU', 'auto')
    
    # Processing Settings
    VIDEO_SAMPLE_RATE = int(os.getenv('VIDEO_SAMPLE_RATE', 10))
    MAX_FRAMES_PER_VIDEO = int(os.getenv('MAX_FRAMES_PER_VIDEO', 100))
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
    
    # Detection Thresholds
    DEEPFAKE_THRESHOLD = float(os.getenv('DEEPFAKE_THRESHOLD', 0.5))
    AI_GENERATED_THRESHOLD = float(os.getenv('AI_GENERATED_THRESHOLD', 0.5))
    
    # Results Storage
    RESULTS_FOLDER = BASE_DIR / 'results'
    HEATMAPS_FOLDER = BASE_DIR / 'static' / 'heatmaps'
    
    @classmethod
    def init_folders(cls):
        """Create necessary folders if they don't exist."""
        folders = [
            cls.UPLOAD_FOLDER,
            cls.MODEL_PATH,
            cls.RESULTS_FOLDER,
            cls.HEATMAPS_FOLDER,
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_allowed_file(cls, filename: str, file_type: str = 'image') -> bool:
        """Check if file extension is allowed."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        if file_type == 'image':
            return ext in cls.ALLOWED_IMAGE_EXTENSIONS
        elif file_type == 'video':
            return ext in cls.ALLOWED_VIDEO_EXTENSIONS
        return ext in cls.ALLOWED_IMAGE_EXTENSIONS | cls.ALLOWED_VIDEO_EXTENSIONS
    
    @classmethod
    def get_file_type(cls, filename: str) -> str:
        """Determine file type from extension."""
        if '.' not in filename:
            return 'unknown'
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in cls.ALLOWED_IMAGE_EXTENSIONS:
            return 'image'
        elif ext in cls.ALLOWED_VIDEO_EXTENSIONS:
            return 'video'
        return 'unknown'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env: str = None) -> Config:
    """Get configuration based on environment."""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config_map.get(env, DevelopmentConfig)
