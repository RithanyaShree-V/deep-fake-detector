"""
DeepFake Detection - Hugging Face Spaces Gradio App
GPU-accelerated deepfake and AI-generated image detection
"""

import gradio as gr
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import cv2
import os
from typing import Tuple, Dict, Any

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Model classes
class DeepFakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class AIGeneratedDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b1', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Initialize models
print("Loading models...")
deepfake_model = DeepFakeDetector().to(device)
ai_gen_model = AIGeneratedDetector().to(device)

# Load weights if available
model_dir = "models"
if os.path.exists(f"{model_dir}/deepfake_detector.pth"):
    deepfake_model.load_state_dict(torch.load(f"{model_dir}/deepfake_detector.pth", map_location=device))
    print("✅ Loaded deepfake detector weights")
    
if os.path.exists(f"{model_dir}/ai_generated_detector.pth"):
    ai_gen_model.load_state_dict(torch.load(f"{model_dir}/ai_generated_detector.pth", map_location=device))
    print("✅ Loaded AI-generated detector weights")

deepfake_model.eval()
ai_gen_model.eval()

def preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Preprocess image for model input"""
    image = image.convert('RGB')
    image = image.resize((size, size), Image.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor.to(device)

def analyze_frequency(image: np.ndarray) -> float:
    """Analyze frequency domain for manipulation artifacts"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2
    
    low_freq = np.sum(magnitude[mask])
    high_freq = np.sum(magnitude[~mask])
    
    ratio = high_freq / (low_freq + 1e-10)
    score = min(ratio / 10.0, 1.0)
    return score

def analyze_noise(image: np.ndarray) -> float:
    """Analyze noise patterns"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize
    score = min(variance / 1000.0, 1.0)
    return score

def detect_deepfake(image: Image.Image) -> Tuple[Dict[str, float], str, str]:
    """
    Main detection function
    Returns: (confidence_dict, verdict, analysis_details)
    """
    if image is None:
        return {"Error": 1.0}, "❌ No image provided", "Please upload an image to analyze."
    
    try:
        # Preprocess
        img_tensor = preprocess_image(image)
        img_array = np.array(image.convert('RGB'))
        
        # Model predictions
        with torch.no_grad():
            # Deepfake detection
            df_output = deepfake_model(img_tensor)
            df_probs = torch.softmax(df_output, dim=1)
            df_fake_prob = df_probs[0, 1].item()
            
            # AI-generated detection
            ai_output = ai_gen_model(img_tensor)
            ai_probs = torch.softmax(ai_output, dim=1)
            ai_gen_prob = ai_probs[0, 1].item()
        
        # Enhanced analysis
        freq_score = analyze_frequency(img_array)
        noise_score = analyze_noise(img_array)
        enhanced_score = (freq_score + noise_score) / 2
        
        # Ensemble score
        final_score = 0.4 * df_fake_prob + 0.3 * ai_gen_prob + 0.3 * enhanced_score
        
        # Determine verdict
        if final_score > 0.7:
            verdict = "🚨 FAKE - High probability of manipulation"
            verdict_color = "red"
        elif final_score > 0.5:
            verdict = "⚠️ SUSPICIOUS - Possible manipulation detected"
            verdict_color = "orange"
        elif final_score > 0.3:
            verdict = "🔍 UNCERTAIN - Inconclusive results"
            verdict_color = "yellow"
        else:
            verdict = "✅ LIKELY AUTHENTIC - No significant manipulation detected"
            verdict_color = "green"
        
        # Confidence breakdown
        confidences = {
            "Deepfake Score": round(df_fake_prob * 100, 1),
            "AI-Generated Score": round(ai_gen_prob * 100, 1),
            "Frequency Anomaly": round(freq_score * 100, 1),
            "Noise Pattern": round(noise_score * 100, 1),
            "Overall Fake Probability": round(final_score * 100, 1)
        }
        
        # Analysis details
        details = f"""
## 📊 Analysis Results

### Model Predictions
- **Deepfake Detection Model**: {df_fake_prob*100:.1f}% fake probability
- **AI-Generated Detection Model**: {ai_gen_prob*100:.1f}% AI-generated probability

### Enhanced Analysis
- **Frequency Domain Analysis**: {freq_score*100:.1f}% anomaly score
- **Noise Pattern Analysis**: {noise_score*100:.1f}% manipulation indicator

### Final Assessment
- **Combined Score**: {final_score*100:.1f}%
- **Device**: {device} {'(GPU Accelerated)' if torch.cuda.is_available() else '(CPU)'}

---
*Higher scores indicate higher likelihood of manipulation*
"""
        
        return confidences, verdict, details
        
    except Exception as e:
        return {"Error": 1.0}, f"❌ Error: {str(e)}", f"An error occurred during analysis: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="DeepFake Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 DeepFake & AI-Generated Image Detector
    
    Upload an image to detect if it's a **deepfake**, **AI-generated**, or **authentic**.
    
    This tool uses:
    - 🧠 **EfficientNet** deep learning models
    - 📊 **Frequency domain analysis** 
    - 🔬 **Noise pattern detection**
    - ⚡ **GPU acceleration** (when available)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Tips
            - Upload clear, high-resolution images for best results
            - Works with faces, objects, and scenes
            - Supports JPG, PNG, WEBP formats
            """)
        
        with gr.Column(scale=1):
            verdict_output = gr.Textbox(label="Verdict", lines=2)
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=5)
            details_output = gr.Markdown(label="Detailed Analysis")
    
    # Examples
    gr.Markdown("### 📸 Example Images")
    gr.Examples(
        examples=[
            ["test_data/real/real_sample_1.jpg"],
            ["test_data/fake/fake_sample_1.jpg"],
        ],
        inputs=input_image,
        outputs=[confidence_output, verdict_output, details_output],
        fn=detect_deepfake,
        cache_examples=False
    )
    
    analyze_btn.click(
        fn=detect_deepfake,
        inputs=input_image,
        outputs=[confidence_output, verdict_output, details_output]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ About
    
    This DeepFake Detector uses an ensemble of deep learning models and traditional image analysis techniques:
    
    | Component | Purpose |
    |-----------|---------|
    | EfficientNet-B0 | Deepfake face manipulation detection |
    | EfficientNet-B1 | AI-generated content detection |
    | FFT Analysis | Frequency domain anomaly detection |
    | Noise Analysis | Compression and editing artifact detection |
    
    **Disclaimer**: This tool provides probabilistic assessments and should not be used as the sole basis for determining image authenticity in critical applications.
    
    ---
    Made with ❤️ using PyTorch, Gradio, and Hugging Face Spaces
    """)

if __name__ == "__main__":
    demo.launch()
