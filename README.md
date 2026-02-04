# 🛡️ DeepFake Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI-Powered DeepFake & AI-Generated Content Detection System**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Architecture](#-architecture)

</div>

---

## 🎯 Overview

DeepFake Detector is a comprehensive web-based AI system designed to identify:
- **Deepfake Face Manipulation** - Face swaps, facial reenactment, lip sync manipulation
- **AI-Generated Images** - Content created by GANs, Diffusion models (DALL-E, Midjourney, Stable Diffusion)
- **Synthetic Media** - Any artificially generated or manipulated visual content

The system provides confidence scores, visual explanations (Grad-CAM heatmaps), and detailed analysis reports.

## ✨ Features

### Core Detection
- 🔍 **Image Analysis** - Analyze JPG, PNG, WEBP images
- 🎬 **Video Analysis** - Process MP4, AVI, MOV videos with frame sampling
- 👤 **Face Detection** - Automatic face detection and extraction using MTCNN
- 📊 **Confidence Scores** - Probability-based predictions with detailed breakdowns

### Explainability
- 🔥 **Grad-CAM Heatmaps** - Visual explanation of AI decision-making
- 📝 **Textual Analysis** - Detailed descriptions of detected artifacts
- 📈 **Frame-by-Frame Results** - Comprehensive video analysis breakdown

### User Experience
- 🎨 **Modern UI** - Clean, responsive design with dark mode
- 📤 **Drag & Drop** - Easy file upload interface
- 📦 **Batch Processing** - Analyze multiple files at once
- 📄 **PDF Reports** - Download professional analysis reports

### Developer Features
- 🔌 **REST API** - Full API for integration
- 🐳 **Production Ready** - Modular, well-documented codebase
- ⚡ **Optimized** - CPU-optimized with optional GPU support

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **VS Code** (recommended IDE)
- **Git** (optional, for cloning)

### Step-by-Step Setup

#### 1. Clone or Download the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

# Or download and extract the ZIP file
```

#### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

> **Note:** This may take several minutes as it downloads PyTorch and other ML libraries.

#### 4. Configure Environment (Optional)

```bash
# Copy example environment file
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux

# Edit .env file to customize settings
```

#### 5. Initialize Directories

The application will automatically create necessary directories on first run, or you can create them manually:

```bash
mkdir uploads models results static\heatmaps
```

#### 6. Run the Application

```bash
# Development mode
python app.py

# With model preloading (faster first request)
python app.py --preload

# Custom host/port
python app.py --host 0.0.0.0 --port 8080
```

#### 7. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📖 Usage

### Web Interface

1. **Upload Media**
   - Drag and drop an image or video onto the upload zone
   - Or click to browse and select a file

2. **Configure Options**
   - Toggle heatmap generation
   - Adjust frame sampling rate for videos

3. **Analyze**
   - Click "Analyze Media" to start detection
   - Wait for processing (progress bar shown)

4. **View Results**
   - See prediction (Real/Deepfake/AI-Generated)
   - View confidence scores and probability breakdown
   - Examine Grad-CAM heatmap visualization
   - Read detailed explanation and artifact list

5. **Download Report**
   - Click "Download Report" for a PDF summary

### Batch Processing

1. Switch to "Batch Upload" tab
2. Select multiple files (up to 10)
3. Click "Analyze All Files"
4. Download batch report

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Analyze Image
```http
POST /api/analyze/image
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- generate_heatmap: boolean (optional, default: true)

Response:
{
  "success": true,
  "filename": "image.jpg",
  "result": {
    "prediction": "Deepfake",
    "confidence": 87.5,
    "probabilities": {...},
    "face_detected": true,
    "face_regions": [...],
    "heatmap_path": "/static/heatmaps/xxx.jpg",
    "explanation": "...",
    "artifacts": [...]
  }
}
```

#### Analyze Video
```http
POST /api/analyze/video
Content-Type: multipart/form-data

Parameters:
- file: Video file (required)
- sample_rate: integer (optional, default: 10)
- max_frames: integer (optional, default: 100)

Response:
{
  "success": true,
  "result": {
    "overall_prediction": "Deepfake",
    "overall_confidence": 75.3,
    "frames_analyzed": 50,
    "suspicious_frames": 38,
    "frame_results": [...],
    "summary": "..."
  }
}
```

#### Batch Analysis
```http
POST /api/analyze/batch
Content-Type: multipart/form-data

Parameters:
- files[]: Multiple files (required)

Response:
{
  "success": true,
  "total": 5,
  "successful": 5,
  "results": [...]
}
```

#### Generate Report
```http
POST /api/report/generate

Response: PDF file download
```

#### API Status
```http
GET /api/status

Response:
{
  "status": "online",
  "version": "1.0.0",
  "gpu_available": true,
  "models_loaded": true
}
```

### Python Example

```python
import requests

# Analyze image
url = "http://localhost:5000/api/analyze/image"
files = {"file": open("suspicious_image.jpg", "rb")}
data = {"generate_heatmap": "true"}

response = requests.post(url, files=files, data=data)
result = response.json()

if result["success"]:
    print(f"Prediction: {result['result']['prediction']}")
    print(f"Confidence: {result['result']['confidence']}%")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/api/analyze/image', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => {
    console.log('Prediction:', data.result.prediction);
    console.log('Confidence:', data.result.confidence);
});
```

---

## 🏗️ Architecture

### Project Structure

```
deepfake-detection/
├── app.py                 # Flask application & API routes
├── config.py              # Configuration management
├── model_loader.py        # ML model loading & management
├── inference.py           # Detection inference pipeline
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
│
├── templates/             # HTML templates
│   ├── index.html         # Home page
│   ├── results.html       # Results page
│   ├── about.html         # About page
│   └── api_docs.html      # API documentation
│
├── static/
│   ├── css/
│   │   └── style.css      # Stylesheet
│   ├── js/
│   │   └── main.js        # Frontend JavaScript
│   └── heatmaps/          # Generated heatmap images
│
├── utils/
│   ├── __init__.py
│   └── pdf_generator.py   # PDF report generation
│
├── models/                # Model weights (auto-created)
├── uploads/               # Uploaded files (auto-created)
└── results/               # Generated reports (auto-created)
```

### ML Architecture

#### DeepFake Detector
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Head:** Custom classification layers with dropout
- **Output:** Binary classification (Real/Deepfake)

#### AI-Generated Detector
- **Backbone:** EfficientNet-B1 (ImageNet pretrained)
- **Head:** Attention mechanism + classification layers
- **Output:** Binary classification (Real/AI-Generated)

### Processing Pipeline

```
Input Image/Video
       │
       ▼
┌──────────────┐
│ Preprocessing │  ← Resize, normalize, format conversion
└──────────────┘
       │
       ▼
┌──────────────┐
│Face Detection │  ← MTCNN face detection & extraction
└──────────────┘
       │
       ▼
┌──────────────┐
│ Model Inference │  ← Dual model: Deepfake + AI-Generated
└──────────────┘
       │
       ▼
┌──────────────┐
│  Grad-CAM    │  ← Generate attention heatmap
└──────────────┘
       │
       ▼
┌──────────────┐
│ Post-Process │  ← Aggregate scores, generate explanation
└──────────────┘
       │
       ▼
    Results
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Flask environment |
| `SECRET_KEY` | dev-key | Flask secret key |
| `USE_GPU` | auto | GPU usage: auto/true/false |
| `VIDEO_SAMPLE_RATE` | 10 | Analyze every Nth frame |
| `MAX_FRAMES_PER_VIDEO` | 100 | Maximum frames to analyze |
| `IMAGE_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 8 | Inference batch size |
| `DEEPFAKE_THRESHOLD` | 0.5 | Deepfake detection threshold |
| `AI_GENERATED_THRESHOLD` | 0.5 | AI detection threshold |

---

## 🔧 Development

### Running in Debug Mode

```bash
python app.py --debug
```

### Testing Models

```bash
# Test model loading
python model_loader.py

# Test inference pipeline
python inference.py
```

### Adding Custom Models

1. Create model class in `model_loader.py`
2. Add configuration to `MODEL_CONFIGS`
3. Update inference logic in `inference.py`

---

## 📊 Performance

### Optimization Tips

1. **GPU Acceleration**
   - Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Set `USE_GPU=true` in `.env`

2. **Video Processing**
   - Increase `VIDEO_SAMPLE_RATE` for faster processing
   - Reduce `MAX_FRAMES_PER_VIDEO` for quick scans

3. **Batch Processing**
   - Increase `BATCH_SIZE` if GPU memory allows

### Benchmarks (Approximate)

| Content | CPU Time | GPU Time |
|---------|----------|----------|
| Image (1080p) | ~2s | ~0.5s |
| Video (30s, 10fps) | ~45s | ~15s |

---

## ⚠️ Disclaimer

**IMPORTANT:** This tool provides **probabilistic assessments** based on AI analysis and should **NOT** be considered legally definitive.

### Responsible Use Guidelines

✅ **DO:**
- Use for detecting potential misinformation
- Verify media authenticity before sharing
- Use as one tool among multiple verification methods
- Report findings responsibly

❌ **DON'T:**
- Use results as legal evidence without expert verification
- Make accusations based solely on AI analysis
- Harass or defame individuals based on results
- Violate others' privacy rights

### Known Limitations

- Heavily compressed images may produce less accurate results
- Very low resolution content may lack sufficient detail
- Novel deepfake techniques may not be detected
- Artistic styles or unusual lighting may affect accuracy

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

<div align="center">

**Built for responsible AI use** 🛡️

Made with ❤️ for fighting misinformation

</div>
