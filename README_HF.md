---
title: DeepFake Detector
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app_gradio.py
pinned: true
license: mit
hardware: gpu-basic
---

# 🔍 DeepFake & AI-Generated Image Detector

An AI-powered tool to detect deepfakes and AI-generated images using deep learning and image analysis.

## Features

- 🧠 **EfficientNet Models** - State-of-the-art neural networks for detection
- 📊 **Multi-Modal Analysis** - Combines deep learning with frequency/noise analysis
- ⚡ **GPU Accelerated** - Fast inference with CUDA support
- 🎯 **High Accuracy** - Trained with anti-overfitting techniques

## How It Works

1. **Upload** an image (JPG, PNG, WEBP)
2. **Analyze** using our ensemble detection system
3. **Review** detailed confidence scores and verdict

## Models Used

| Model | Architecture | Purpose |
|-------|--------------|---------|
| Deepfake Detector | EfficientNet-B0 | Face manipulation detection |
| AI-Generated Detector | EfficientNet-B1 | Synthetic image detection |

## Analysis Methods

- **Deep Learning**: Neural network pattern recognition
- **Frequency Analysis**: FFT-based artifact detection
- **Noise Analysis**: Compression and editing detection

## Usage

```python
# The app provides a simple web interface
# Just upload an image and click "Analyze"
```

## Training

Models were trained with:
- Early stopping to prevent overfitting
- Label smoothing (0.1)
- Weight decay regularization
- Data augmentation
- Gradual backbone unfreezing

## Disclaimer

This tool provides probabilistic assessments. Results should not be used as the sole basis for determining image authenticity in critical applications.

## License

MIT License
