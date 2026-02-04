"""
Model Download Script for DeepFake Detection System
====================================================
Downloads and prepares pretrained model weights.

Usage:
    python download_models.py

This script initializes the models with pretrained EfficientNet weights
from ImageNet. For production use, you would replace this with actual
deepfake detection weights trained on specialized datasets.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from config import Config
from model_loader import ModelLoader, DeepFakeDetector, AIGeneratedDetector


def download_and_save_models():
    """
    Initialize and save model weights.
    
    This function creates the model architectures with pretrained
    EfficientNet backbones and saves the initial weights.
    """
    print("=" * 60)
    print("DeepFake Detection Model Initialization")
    print("=" * 60)
    
    # Create models directory
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "-" * 60)
    
    # Initialize DeepFake Detector
    print("\n[1/2] Initializing DeepFake Detector...")
    print("      - Architecture: EfficientNet-B0 + Custom Head")
    print("      - Pretrained: ImageNet weights")
    
    deepfake_model = DeepFakeDetector(num_classes=2, pretrained=True)
    deepfake_path = Config.MODEL_PATH / 'deepfake_detector.pth'
    
    torch.save(deepfake_model.state_dict(), deepfake_path)
    print(f"      ✓ Saved to: {deepfake_path}")
    
    # Count parameters
    num_params = sum(p.numel() for p in deepfake_model.parameters())
    print(f"      - Parameters: {num_params:,}")
    
    # Initialize AI-Generated Detector
    print("\n[2/2] Initializing AI-Generated Content Detector...")
    print("      - Architecture: EfficientNet-B1 + Attention + Custom Head")
    print("      - Pretrained: ImageNet weights")
    
    ai_gen_model = AIGeneratedDetector(num_classes=2, pretrained=True)
    ai_gen_path = Config.MODEL_PATH / 'ai_generated_detector.pth'
    
    torch.save(ai_gen_model.state_dict(), ai_gen_path)
    print(f"      ✓ Saved to: {ai_gen_path}")
    
    num_params = sum(p.numel() for p in ai_gen_model.parameters())
    print(f"      - Parameters: {num_params:,}")
    
    # Verify models
    print("\n" + "-" * 60)
    print("\nVerifying models...")
    
    # Test inference
    test_input = torch.randn(1, 3, 224, 224)
    
    deepfake_model.eval()
    ai_gen_model.eval()
    
    with torch.no_grad():
        df_output = deepfake_model(test_input)
        ai_output = ai_gen_model(test_input)
    
    print(f"  DeepFake Detector output shape: {df_output.shape}")
    print(f"  AI-Generated Detector output shape: {ai_output.shape}")
    
    print("\n" + "=" * 60)
    print("Model initialization complete!")
    print("=" * 60)
    
    print("\n📝 Note:")
    print("   These models use ImageNet-pretrained EfficientNet backbones.")
    print("   For optimal deepfake detection performance, consider:")
    print("   1. Fine-tuning on a deepfake dataset (e.g., FaceForensics++)")
    print("   2. Using specialized pretrained weights from research papers")
    print("   3. Training with your own labeled dataset")
    
    print("\n🚀 Ready to run:")
    print("   python app.py")
    
    return True


def main():
    """Main entry point."""
    try:
        success = download_and_save_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
