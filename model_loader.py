"""
Model Loader Module for DeepFake Detection System
==================================================
Handles loading and initialization of pretrained models for:
- Deepfake face detection
- AI-generated image detection (GAN/Diffusion)

Uses EfficientNet-based architecture with pretrained weights.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-based DeepFake Detection Model.
    
    Architecture:
    - Backbone: EfficientNet-B0 (pretrained on ImageNet)
    - Custom classification head for binary classification
    - Supports Grad-CAM visualization
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        return self.backbone(x)
    
    # Grad-CAM hooks
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get activations from last conv layer for Grad-CAM."""
        # Get the last convolutional layer output
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        
        for block in self.backbone.blocks:
            x = block(x)
        
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        return x


class AIGeneratedDetector(nn.Module):
    """
    Detector for AI-Generated Images (GAN/Diffusion).
    
    Uses EfficientNet-B1 with modifications for detecting
    artifacts common in AI-generated images.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            'efficientnet_b1',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features
        
        # Classification head with attention
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 4, self.feature_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output


class ModelLoader:
    """
    Model Manager for loading and managing detection models.
    
    Supports:
    - Automatic model downloading from HuggingFace Hub
    - CPU/GPU device management
    - Multiple model types (deepfake, AI-generated)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'deepfake': {
            'class': DeepFakeDetector,
            'filename': 'deepfake_detector.pth',
            'input_size': 224,
            'num_classes': 2,
            'labels': ['Real', 'Deepfake']
        },
        'ai_generated': {
            'class': AIGeneratedDetector,
            'filename': 'ai_generated_detector.pth',
            'input_size': 224,
            'num_classes': 2,
            'labels': ['Real', 'AI-Generated']
        }
    }
    
    def __init__(self, model_path: Path = None, device: str = None):
        """
        Initialize ModelLoader.
        
        Args:
            model_path: Path to model weights directory
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.device = self._setup_device(device)
        self.models: Dict[str, nn.Module] = {}
        
        # Ensure model directory exists
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelLoader initialized. Using device: {self.device}")
        
    def _setup_device(self, device: str = None) -> torch.device:
        """Setup compute device (CPU/GPU)."""
        if device is None:
            device = Config.USE_GPU
            
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("No GPU detected, using CPU")
        
        return torch.device(device)
    
    def load_model(self, model_type: str, weights_path: str = None) -> nn.Module:
        """
        Load a detection model.
        
        Args:
            model_type: 'deepfake' or 'ai_generated'
            weights_path: Optional path to custom weights
            
        Returns:
            Loaded model ready for inference
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
            
        config = self.MODEL_CONFIGS[model_type]
        
        # Initialize model
        model = config['class'](
            num_classes=config['num_classes'],
            pretrained=True
        )
        
        # Load weights if available
        if weights_path:
            weight_file = Path(weights_path)
        else:
            weight_file = self.model_path / config['filename']
            
        if weight_file.exists():
            try:
                state_dict = torch.load(weight_file, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weight_file}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Using pretrained backbone.")
        else:
            logger.info(f"No custom weights found at {weight_file}. Using pretrained backbone.")
            # Save initial weights for future use
            self._save_initial_weights(model, weight_file)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Cache model
        self.models[model_type] = model
        
        return model
    
    def _save_initial_weights(self, model: nn.Module, path: Path):
        """Save initial model weights."""
        try:
            torch.save(model.state_dict(), path)
            logger.info(f"Saved initial weights to {path}")
        except Exception as e:
            logger.warning(f"Could not save weights: {e}")
    
    def load_all_models(self) -> Dict[str, nn.Module]:
        """Load all detection models."""
        for model_type in self.MODEL_CONFIGS:
            self.load_model(model_type)
        return self.models
    
    def get_model(self, model_type: str) -> Optional[nn.Module]:
        """Get a loaded model by type."""
        if model_type not in self.models:
            self.load_model(model_type)
        return self.models.get(model_type)
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get model configuration info."""
        return self.MODEL_CONFIGS.get(model_type, {})
    
    @property
    def available_models(self) -> list:
        """List available model types."""
        return list(self.MODEL_CONFIGS.keys())
    
    def unload_model(self, model_type: str):
        """Unload a model to free memory."""
        if model_type in self.models:
            del self.models[model_type]
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            logger.info(f"Unloaded {model_type} model")


def download_pretrained_weights():
    """
    Download pretrained weights from HuggingFace Hub.
    
    This function can be extended to download actual pretrained weights
    from a public repository.
    """
    logger.info("Checking for pretrained weights...")
    
    # For demo purposes, we'll use the EfficientNet pretrained weights
    # which are automatically downloaded by timm
    
    # In production, you would download from HuggingFace like:
    # hf_hub_download(
    #     repo_id="your-repo/deepfake-detector",
    #     filename="deepfake_detector.pth",
    #     local_dir=Config.MODEL_PATH
    # )
    
    logger.info("Using EfficientNet pretrained backbone (ImageNet)")
    

# Singleton instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create ModelLoader singleton."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def initialize_models():
    """Initialize all models on startup."""
    loader = get_model_loader()
    loader.load_all_models()
    return loader


if __name__ == "__main__":
    # Test model loading
    print("Testing Model Loader...")
    
    loader = ModelLoader()
    
    # Load models
    deepfake_model = loader.load_model('deepfake')
    ai_gen_model = loader.load_model('ai_generated')
    
    # Test inference
    dummy_input = torch.randn(1, 3, 224, 224).to(loader.device)
    
    with torch.no_grad():
        df_output = deepfake_model(dummy_input)
        ai_output = ai_gen_model(dummy_input)
    
    print(f"Deepfake model output shape: {df_output.shape}")
    print(f"AI-Generated model output shape: {ai_output.shape}")
    print("Model loading test passed!")
