"""
Inference Module for DeepFake Detection System
===============================================
Handles all inference operations including:
- Image preprocessing
- Face detection
- Model inference
- Video frame extraction and batch processing
- Grad-CAM visualization for explainability
- Enhanced feature analysis (frequency, texture, noise)
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from config import Config
from model_loader import get_model_loader, ModelLoader

# Try to import enhanced detection (optional)
try:
    from enhanced_detection import get_enhanced_detector, EnhancedFeatures
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    logger.warning("Enhanced detection not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for detection results."""
    
    prediction: str  # 'Real', 'Deepfake', 'AI-Generated'
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    face_detected: bool = False
    face_regions: List[Dict] = field(default_factory=list)
    heatmap_path: Optional[str] = None
    explanation: str = ""
    artifacts: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_fake(self) -> bool:
        """Returns True if the content is detected as fake (Deepfake or AI-Generated)."""
        return self.prediction in ('Deepfake', 'AI-Generated')
    
    @property
    def deepfake_score(self) -> float:
        """Returns the deepfake probability score."""
        return self.probabilities.get('Deepfake', 0.0)
    
    @property
    def ai_generated_score(self) -> float:
        """Returns the AI-generated probability score."""
        return self.probabilities.get('AI-Generated', 0.0)
    
    @property
    def faces_detected(self) -> int:
        """Returns the number of faces detected."""
        return len(self.face_regions)
    
    @property
    def processing_time(self) -> float:
        """Returns the processing time (placeholder, set externally)."""
        return getattr(self, '_processing_time', 0.0)
    
    @processing_time.setter
    def processing_time(self, value: float):
        self._processing_time = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'prediction': self.prediction,
            'confidence': round(self.confidence * 100, 2),
            'probabilities': {k: round(v * 100, 2) for k, v in self.probabilities.items()},
            'face_detected': self.face_detected,
            'face_regions': self.face_regions,
            'heatmap_path': self.heatmap_path,
            'explanation': self.explanation,
            'artifacts': self.artifacts,
            'timestamp': self.timestamp
        }


@dataclass
class VideoAnalysisResult:
    """Data class for video analysis results."""
    
    overall_prediction: str
    overall_confidence: float
    frame_results: List[Dict] = field(default_factory=list)
    frames_analyzed: int = 0
    total_frames: int = 0
    suspicious_frames: int = 0
    summary: str = ""
    video_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'overall_prediction': self.overall_prediction,
            'overall_confidence': round(self.overall_confidence * 100, 2),
            'frame_results': self.frame_results,
            'frames_analyzed': self.frames_analyzed,
            'total_frames': self.total_frames,
            'suspicious_frames': self.suspicious_frames,
            'suspicious_percentage': round(self.suspicious_frames / max(self.frames_analyzed, 1) * 100, 2),
            'summary': self.summary,
            'video_info': self.video_info
        }


class FaceDetector:
    """
    Face Detection using MTCNN.
    
    Detects and extracts face regions from images for focused analysis.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=224,
            margin=40,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True
        )
        
    def detect_faces(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect and extract faces from image.
        
        Args:
            image: BGR image array (OpenCV format)
            
        Returns:
            Tuple of (face_images, face_regions)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        face_images = []
        face_regions = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > 0.9:  # High confidence threshold
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)
                    
                    # Extract face region
                    face = image[y1:y2, x1:x2]
                    if face.size > 0:
                        face_images.append(face)
                        face_regions.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(prob),
                            'landmarks': landmarks[i].tolist() if landmarks is not None else None
                        })
        
        return face_images, face_regions


class ImagePreprocessor:
    """
    Image preprocessing pipeline for model inference.
    """
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transform for visualization (no normalization)
        self.vis_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: NumPy array, PIL Image, or file path
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
    
    def preprocess_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """Preprocess batch of images."""
        tensors = [self.preprocess(img).squeeze(0) for img in images]
        return torch.stack(tensors)


class GradCAMVisualizer:
    """
    Grad-CAM visualization for model explainability.
    
    Generates heatmaps showing which regions influenced the model's decision.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
    def _hook_gradients(self, grad):
        """Hook to capture gradients."""
        self.gradients = grad
        
    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input tensor
            target_class: Target class for visualization (default: predicted class)
            
        Returns:
            Heatmap as numpy array
        """
        try:
            self.model.eval()
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad = True
            
            # Forward pass - get features from backbone before pooling
            x = input_tensor
            
            # Process through backbone layers to get pre-pooling features
            x = self.model.backbone.conv_stem(x)
            x = self.model.backbone.bn1(x)
            
            for block in self.model.backbone.blocks:
                x = block(x)
            
            x = self.model.backbone.conv_head(x)
            x = self.model.backbone.bn2(x)
            
            # Save features for heatmap (before global pooling)
            features = x.clone()
            if x.requires_grad:
                x.register_hook(self._hook_gradients)
            
            # Global average pooling (same as backbone)
            pooled = self.model.backbone.global_pool(x)
            
            # Get prediction through classifier
            output = self.model.classifier(pooled)
        
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Generate heatmap
            if self.gradients is not None and features is not None:
                weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * features, dim=1).squeeze()
                cam = F.relu(cam)
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                cam = cam.detach().cpu().numpy()
                
                # Resize to input size
                cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
                return cam
            
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        except Exception as e:
            # Return blank heatmap on error
            logger.warning(f"GradCAM generation failed: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (BGR)
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor
            
        Returns:
            Image with overlaid heatmap
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), 
            cv2.COLORMAP_JET
        )
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def save_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        filename: str
    ) -> str:
        """Save heatmap overlay to file."""
        overlay = self.overlay_heatmap(image, heatmap)
        
        save_path = Config.HEATMAPS_FOLDER / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), overlay)
        
        return f'/static/heatmaps/{filename}'


class DeepFakeInference:
    """
    Main inference engine for deepfake detection.
    
    Combines face detection, model inference, and explainability.
    """
    
    def __init__(self):
        self.model_loader = get_model_loader()
        self.device = self.model_loader.device
        
        # Initialize components
        self.face_detector = FaceDetector(self.device)
        self.preprocessor = ImagePreprocessor(Config.IMAGE_SIZE)
        
        # Load models
        self.deepfake_model = self.model_loader.load_model('deepfake')
        self.ai_gen_model = self.model_loader.load_model('ai_generated')
        
        # Initialize Grad-CAM visualizers
        self.gradcam_deepfake = GradCAMVisualizer(self.deepfake_model, self.device)
        self.gradcam_ai_gen = GradCAMVisualizer(self.ai_gen_model, self.device)
        
        logger.info("DeepFakeInference initialized successfully")
    
    def analyze_image(
        self, 
        image_path: str, 
        generate_heatmap: bool = True
    ) -> DetectionResult:
        """
        Analyze a single image for deepfake/AI-generated content.
        
        Args:
            image_path: Path to image file
            generate_heatmap: Whether to generate Grad-CAM heatmap
            
        Returns:
            DetectionResult with predictions and explanations
        """
        import time
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect faces
        face_images, face_regions = self.face_detector.detect_faces(image)
        
        # Preprocess image
        input_tensor = self.preprocessor.preprocess(image).to(self.device)
        
        # Run inference on both models
        with torch.no_grad():
            # Deepfake detection
            df_logits = self.deepfake_model(input_tensor)
            df_probs = F.softmax(df_logits, dim=1)[0]
            
            # AI-generated detection
            ai_logits = self.ai_gen_model(input_tensor)
            ai_probs = F.softmax(ai_logits, dim=1)[0]
        
        # Determine prediction
        df_fake_prob = df_probs[1].item()
        ai_gen_prob = ai_probs[1].item()
        
        # Enhanced detection (frequency, texture, noise analysis)
        enhanced_score = 0.5
        enhanced_features = None
        if ENHANCED_DETECTION_AVAILABLE:
            try:
                enhanced_detector = get_enhanced_detector()
                enhanced_result = enhanced_detector.analyze(image)
                enhanced_score = enhanced_result.overall_suspicion
                enhanced_features = enhanced_result.to_dict()
            except Exception as e:
                logger.warning(f"Enhanced detection failed: {e}")
        
        # Ensemble decision logic with enhanced features
        # Combine neural network predictions with traditional analysis
        ensemble_fake_score = (
            0.4 * df_fake_prob +
            0.3 * ai_gen_prob +
            0.3 * enhanced_score
        )
        
        # Decision thresholds
        if ensemble_fake_score > 0.6:
            if df_fake_prob > ai_gen_prob:
                prediction = 'Deepfake'
                confidence = (df_fake_prob + ensemble_fake_score) / 2
            else:
                prediction = 'AI-Generated'
                confidence = (ai_gen_prob + ensemble_fake_score) / 2
            target_model = 'deepfake' if df_fake_prob > ai_gen_prob else 'ai_generated'
        elif ensemble_fake_score > 0.45:
            # Suspicious but not conclusive
            prediction = 'Suspicious'
            confidence = ensemble_fake_score
            target_model = 'deepfake'
        else:
            prediction = 'Real'
            confidence = 1 - ensemble_fake_score
            target_model = 'deepfake'
        
        # Generate heatmap if requested
        heatmap_path = None
        if generate_heatmap:
            heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
            
            if target_model == 'deepfake':
                heatmap = self.gradcam_deepfake.generate_heatmap(input_tensor)
            else:
                heatmap = self.gradcam_ai_gen.generate_heatmap(input_tensor)
            
            heatmap_path = self.gradcam_deepfake.save_heatmap(
                image, heatmap, heatmap_filename
            )
        
        # Generate explanation
        explanation, artifacts = self._generate_explanation(
            prediction, 
            confidence, 
            df_fake_prob, 
            ai_gen_prob,
            len(face_regions),
            enhanced_features
        )
        
        # Build probabilities dict with enhanced features
        probabilities = {
            'Real (Deepfake Model)': df_probs[0].item(),
            'Deepfake': df_fake_prob,
            'Real (AI Model)': ai_probs[0].item(),
            'AI-Generated': ai_gen_prob,
            'Ensemble Score': ensemble_fake_score
        }
        
        if enhanced_features:
            probabilities['Enhanced Analysis'] = enhanced_score
        
        result = DetectionResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            face_detected=len(face_regions) > 0,
            face_regions=face_regions,
            heatmap_path=heatmap_path,
            explanation=explanation,
            artifacts=artifacts
        )
        
        # Set processing time
        result.processing_time = time.time() - start_time
        
        return result
    
    def analyze_video(
        self, 
        video_path: str, 
        sample_rate: int = None,
        max_frames: int = None,
        progress_callback=None
    ) -> VideoAnalysisResult:
        """
        Analyze video for deepfake content.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame
            max_frames: Maximum frames to analyze
            progress_callback: Optional callback for progress updates
            
        Returns:
            VideoAnalysisResult with frame-wise and overall predictions
        """
        sample_rate = sample_rate or Config.VIDEO_SAMPLE_RATE
        max_frames = max_frames or Config.MAX_FRAMES_PER_VIDEO
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': round(duration, 2),
            'resolution': f"{width}x{height}"
        }
        
        frame_results = []
        frame_idx = 0
        analyzed_count = 0
        suspicious_count = 0
        
        # Batch processing
        batch_frames = []
        batch_indices = []
        
        logger.info(f"Analyzing video: {total_frames} frames, sampling every {sample_rate} frames")
        
        while cap.isOpened() and analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                
                # Process batch
                if len(batch_frames) >= Config.BATCH_SIZE:
                    results = self._process_frame_batch(batch_frames, batch_indices)
                    frame_results.extend(results)
                    suspicious_count += sum(1 for r in results if r['prediction'] != 'Real')
                    analyzed_count += len(batch_frames)
                    
                    if progress_callback:
                        progress_callback(analyzed_count, min(total_frames // sample_rate, max_frames))
                    
                    batch_frames = []
                    batch_indices = []
            
            frame_idx += 1
        
        # Process remaining frames
        if batch_frames:
            results = self._process_frame_batch(batch_frames, batch_indices)
            frame_results.extend(results)
            suspicious_count += sum(1 for r in results if r['prediction'] != 'Real')
            analyzed_count += len(batch_frames)
        
        cap.release()
        
        # Calculate overall prediction
        if analyzed_count > 0:
            fake_ratio = suspicious_count / analyzed_count
            
            # Aggregate confidences
            avg_confidence = np.mean([r['confidence'] for r in frame_results])
            
            if fake_ratio > 0.5:
                # Check if more deepfake or AI-generated
                deepfake_frames = sum(1 for r in frame_results if r['prediction'] == 'Deepfake')
                ai_gen_frames = sum(1 for r in frame_results if r['prediction'] == 'AI-Generated')
                
                if deepfake_frames > ai_gen_frames:
                    overall_prediction = 'Deepfake'
                else:
                    overall_prediction = 'AI-Generated'
                overall_confidence = avg_confidence
            else:
                overall_prediction = 'Real'
                overall_confidence = 1 - fake_ratio
        else:
            overall_prediction = 'Unknown'
            overall_confidence = 0.0
        
        # Generate summary
        summary = self._generate_video_summary(
            overall_prediction, 
            analyzed_count, 
            suspicious_count,
            duration
        )
        
        return VideoAnalysisResult(
            overall_prediction=overall_prediction,
            overall_confidence=overall_confidence,
            frame_results=frame_results,
            frames_analyzed=analyzed_count,
            total_frames=total_frames,
            suspicious_frames=suspicious_count,
            summary=summary,
            video_info=video_info
        )
    
    def _process_frame_batch(
        self, 
        frames: List[np.ndarray], 
        indices: List[int]
    ) -> List[Dict]:
        """Process a batch of video frames."""
        results = []
        
        # Preprocess batch
        tensors = []
        for frame in frames:
            tensor = self.preprocessor.preprocess(frame)
            tensors.append(tensor.squeeze(0))
        
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            df_logits = self.deepfake_model(batch_tensor)
            df_probs = F.softmax(df_logits, dim=1)
            
            ai_logits = self.ai_gen_model(batch_tensor)
            ai_probs = F.softmax(ai_logits, dim=1)
        
        # Process results
        for i, frame_idx in enumerate(indices):
            df_fake = df_probs[i, 1].item()
            ai_gen = ai_probs[i, 1].item()
            
            if df_fake > Config.DEEPFAKE_THRESHOLD and df_fake > ai_gen:
                prediction = 'Deepfake'
                confidence = df_fake
            elif ai_gen > Config.AI_GENERATED_THRESHOLD:
                prediction = 'AI-Generated'
                confidence = ai_gen
            else:
                prediction = 'Real'
                confidence = max(df_probs[i, 0].item(), ai_probs[i, 0].item())
            
            results.append({
                'frame_index': frame_idx,
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'deepfake_score': round(df_fake * 100, 2),
                'ai_generated_score': round(ai_gen * 100, 2)
            })
        
        return results
    
    def _generate_explanation(
        self, 
        prediction: str, 
        confidence: float,
        df_prob: float,
        ai_prob: float,
        face_count: int,
        enhanced_features: Dict = None
    ) -> Tuple[str, List[str]]:
        """Generate textual explanation for the prediction."""
        artifacts = []
        
        if prediction == 'Deepfake':
            explanation = f"The model detected potential face manipulation with {confidence*100:.1f}% confidence. "
            
            if confidence > 0.8:
                explanation += "Strong indicators of face swapping or facial reenactment were found. "
                artifacts.append("High probability of facial manipulation")
            elif confidence > 0.6:
                explanation += "Moderate signs of manipulation detected in facial regions. "
                artifacts.append("Possible facial inconsistencies")
            
            if face_count > 0:
                artifacts.append(f"{face_count} face(s) detected and analyzed")
                explanation += f"Analysis focused on {face_count} detected face region(s). "
            
            artifacts.extend([
                "Check for unnatural blending around face edges",
                "Look for inconsistent lighting on face",
                "Observe eye blinking patterns in video"
            ])
            
        elif prediction == 'AI-Generated':
            explanation = f"The image shows characteristics of AI-generated content with {confidence*100:.1f}% confidence. "
            
            artifacts.extend([
                "Potential GAN/Diffusion model artifacts detected",
                "Check for repetitive patterns or textures",
                "Look for anatomical inconsistencies",
                "Examine background for unnatural elements"
            ])
            
            if confidence > 0.8:
                explanation += "Strong signatures of generative AI models were detected. "
                
        elif prediction == 'Suspicious':
            explanation = f"The content shows some suspicious characteristics ({confidence*100:.1f}% suspicion score). "
            explanation += "While not definitively fake, further investigation is recommended. "
            
            artifacts.extend([
                "Some anomalies detected but inconclusive",
                "Recommend manual verification",
                "Consider source credibility"
            ])
            
        else:
            explanation = f"The content appears to be authentic with {confidence*100:.1f}% confidence. "
            explanation += "No significant manipulation indicators were detected. "
            
            artifacts.append("No major artifacts detected")
            artifacts.append("Natural image characteristics preserved")
        
        # Add enhanced analysis details if available
        if enhanced_features:
            explanation += "\n\nEnhanced Analysis: "
            if enhanced_features.get('frequency_score', 0.5) > 0.6:
                artifacts.append("Unusual frequency domain patterns")
            if enhanced_features.get('texture_score', 0.5) > 0.6:
                artifacts.append("Suspicious texture characteristics")
            if enhanced_features.get('noise_analysis_score', 0.5) > 0.6:
                artifacts.append("Abnormal noise patterns detected")
            if enhanced_features.get('color_consistency_score', 0.5) > 0.6:
                artifacts.append("Color distribution anomalies")
        
        # Add general note
        explanation += "\n\nNote: This analysis is probabilistic and should not be considered legally definitive."
        
        return explanation, artifacts
    
    def _generate_video_summary(
        self, 
        prediction: str, 
        analyzed: int, 
        suspicious: int,
        duration: float
    ) -> str:
        """Generate summary for video analysis."""
        ratio = suspicious / max(analyzed, 1) * 100
        
        summary = f"Analyzed {analyzed} frames from a {duration:.1f}s video. "
        
        if prediction == 'Deepfake':
            summary += f"{suspicious} frames ({ratio:.1f}%) showed signs of deepfake manipulation. "
            summary += "The video likely contains face-swapped or manipulated facial content."
        elif prediction == 'AI-Generated':
            summary += f"{suspicious} frames ({ratio:.1f}%) appeared to be AI-generated. "
            summary += "The video may contain synthetic content created by generative AI."
        else:
            summary += f"Only {suspicious} frames ({ratio:.1f}%) showed potential issues. "
            summary += "The video appears to be authentic."
        
        return summary


# Singleton instance
_inference_engine: Optional[DeepFakeInference] = None


def get_inference_engine() -> DeepFakeInference:
    """Get or create inference engine singleton."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DeepFakeInference()
    return _inference_engine


if __name__ == "__main__":
    # Test inference
    print("Testing Inference Engine...")
    
    engine = DeepFakeInference()
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_path = Config.UPLOAD_FOLDER / "test_image.jpg"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_image)
    
    # Test analysis
    result = engine.analyze_image(str(test_path))
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print(f"Explanation: {result.explanation}")
    
    # Cleanup
    test_path.unlink()
    print("Inference test passed!")
