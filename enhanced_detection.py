"""
Enhanced Detection Features for DeepFake Detection
===================================================
Additional analysis methods to improve detection accuracy:
- Frequency domain analysis (FFT)
- Texture analysis
- Color consistency checks
- Face landmark analysis
- Ensemble voting
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeatures:
    """Enhanced feature analysis results."""
    frequency_score: float = 0.5
    texture_score: float = 0.5
    color_consistency_score: float = 0.5
    noise_analysis_score: float = 0.5
    compression_artifacts_score: float = 0.5
    overall_suspicion: float = 0.5
    features: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'frequency_score': round(self.frequency_score, 3),
            'texture_score': round(self.texture_score, 3),
            'color_consistency_score': round(self.color_consistency_score, 3),
            'noise_analysis_score': round(self.noise_analysis_score, 3),
            'compression_artifacts_score': round(self.compression_artifacts_score, 3),
            'overall_suspicion': round(self.overall_suspicion, 3)
        }


class FrequencyAnalyzer:
    """
    Analyze frequency domain characteristics.
    
    AI-generated images often have different frequency distributions
    than natural images, especially in high-frequency components.
    """
    
    @staticmethod
    def compute_fft_features(image: np.ndarray) -> Dict[str, float]:
        """
        Compute FFT-based features for fake detection.
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary of frequency features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log transform for better visualization
        magnitude_spectrum = np.log1p(magnitude_spectrum)
        
        # Divide into frequency bands
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create radial masks for different frequency bands
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Low frequency (center)
        low_mask = r < h * 0.1
        # Mid frequency
        mid_mask = (r >= h * 0.1) & (r < h * 0.3)
        # High frequency (edges)
        high_mask = r >= h * 0.3
        
        # Compute energy in each band
        total_energy = np.sum(magnitude_spectrum)
        low_energy = np.sum(magnitude_spectrum[low_mask]) / (total_energy + 1e-8)
        mid_energy = np.sum(magnitude_spectrum[mid_mask]) / (total_energy + 1e-8)
        high_energy = np.sum(magnitude_spectrum[high_mask]) / (total_energy + 1e-8)
        
        # AI-generated images often have less high-frequency content
        # or unusual periodic patterns
        high_freq_ratio = high_energy / (low_energy + 1e-8)
        
        # Check for periodic artifacts (common in GANs)
        # Look for spikes in the frequency domain
        mean_magnitude = np.mean(magnitude_spectrum)
        std_magnitude = np.std(magnitude_spectrum)
        spike_threshold = mean_magnitude + 3 * std_magnitude
        spike_count = np.sum(magnitude_spectrum > spike_threshold)
        spike_ratio = spike_count / magnitude_spectrum.size
        
        # Compute azimuthal variance (real images have more uniform distribution)
        angles = np.arctan2(y - center_y, x - center_x)
        angle_bins = np.linspace(-np.pi, np.pi, 16)
        azimuthal_energies = []
        for i in range(len(angle_bins) - 1):
            angle_mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
            azimuthal_energies.append(np.mean(magnitude_spectrum[angle_mask]))
        azimuthal_variance = np.var(azimuthal_energies) / (np.mean(azimuthal_energies) + 1e-8)
        
        return {
            'low_freq_energy': low_energy,
            'mid_freq_energy': mid_energy,
            'high_freq_energy': high_energy,
            'high_freq_ratio': high_freq_ratio,
            'spike_ratio': spike_ratio,
            'azimuthal_variance': azimuthal_variance
        }
    
    @staticmethod
    def get_frequency_suspicion_score(features: Dict[str, float]) -> float:
        """
        Compute suspicion score based on frequency features.
        
        Returns:
            Score from 0 (likely real) to 1 (likely fake)
        """
        score = 0.5
        
        # Low high-frequency content is suspicious
        if features['high_freq_energy'] < 0.1:
            score += 0.1
        elif features['high_freq_energy'] > 0.3:
            score -= 0.1
        
        # High spike ratio indicates artificial patterns
        if features['spike_ratio'] > 0.01:
            score += 0.15
        
        # Unusual azimuthal variance
        if features['azimuthal_variance'] > 0.5:
            score += 0.1
        
        return np.clip(score, 0, 1)


class TextureAnalyzer:
    """
    Analyze texture patterns for detecting AI-generated content.
    
    Uses Local Binary Patterns (LBP) and other texture descriptors.
    """
    
    @staticmethod
    def compute_lbp_histogram(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Compute Local Binary Pattern histogram.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                binary_code = 0
                for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                               (1, 1), (1, 0), (1, -1), (0, -1)]):
                    if gray[i + dy, j + dx] >= center:
                        binary_code |= (1 << k)
                lbp[i, j] = binary_code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        
        return hist
    
    @staticmethod
    def compute_texture_features(image: np.ndarray) -> Dict[str, float]:
        """
        Compute various texture features.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        
        # Compute statistics
        features = {
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'gradient_max': np.max(gradient_magnitude),
            'laplacian_var': np.var(laplacian),
            'texture_entropy': TextureAnalyzer._compute_entropy(gray),
            'smoothness': 1 - 1 / (1 + np.var(gray / 255.0))
        }
        
        return features
    
    @staticmethod
    def _compute_entropy(image: np.ndarray) -> float:
        """Compute image entropy."""
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy
    
    @staticmethod
    def get_texture_suspicion_score(features: Dict[str, float]) -> float:
        """
        Compute suspicion score based on texture features.
        """
        score = 0.5
        
        # Very smooth images are suspicious (AI tends to over-smooth)
        if features['smoothness'] > 0.95:
            score += 0.15
        
        # Low texture entropy is suspicious
        if features['texture_entropy'] < 5.0:
            score += 0.1
        
        # Very uniform gradients are suspicious
        if features['gradient_std'] < 10:
            score += 0.1
        
        # High laplacian variance indicates natural edges
        if features['laplacian_var'] > 1000:
            score -= 0.1
        
        return np.clip(score, 0, 1)


class ColorAnalyzer:
    """
    Analyze color consistency and distribution.
    
    AI-generated images may have unnatural color distributions.
    """
    
    @staticmethod
    def compute_color_features(image: np.ndarray) -> Dict[str, float]:
        """
        Compute color-based features.
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract channels
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)
        
        features = {
            # Saturation statistics
            'saturation_mean': np.mean(s),
            'saturation_std': np.std(s),
            'saturation_skew': ColorAnalyzer._compute_skewness(s),
            
            # Color channel correlations
            'channel_correlation_ab': np.corrcoef(a.ravel(), b.ravel())[0, 1],
            
            # Hue distribution
            'hue_entropy': TextureAnalyzer._compute_entropy(h),
            
            # Brightness uniformity
            'brightness_std': np.std(v),
            
            # Color histogram uniformity
            'color_uniformity': ColorAnalyzer._compute_color_uniformity(image)
        }
        
        return features
    
    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _compute_color_uniformity(image: np.ndarray) -> float:
        """Compute color distribution uniformity."""
        # Compute 3D color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)
        
        # Compute entropy (higher = more uniform)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        max_entropy = np.log2(512)  # 8*8*8 bins
        
        return entropy / max_entropy
    
    @staticmethod
    def get_color_suspicion_score(features: Dict[str, float]) -> float:
        """
        Compute suspicion score based on color features.
        """
        score = 0.5
        
        # Very uniform colors are suspicious
        if features['color_uniformity'] < 0.3:
            score += 0.1
        
        # Unusual saturation distribution
        if features['saturation_std'] < 20:
            score += 0.1
        
        # Check for NaN in correlation
        if np.isnan(features['channel_correlation_ab']):
            score += 0.05
        
        return np.clip(score, 0, 1)


class NoiseAnalyzer:
    """
    Analyze noise patterns in images.
    
    Different cameras and AI generators produce different noise signatures.
    """
    
    @staticmethod
    def compute_noise_features(image: np.ndarray) -> Dict[str, float]:
        """
        Extract noise-related features.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32)
        
        # Estimate noise using median filter residual
        denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        noise_residual = gray - denoised
        
        # High-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
        high_freq = cv2.filter2D(gray, -1, kernel)
        
        features = {
            'noise_std': np.std(noise_residual),
            'noise_mean': np.mean(np.abs(noise_residual)),
            'high_freq_energy': np.mean(np.abs(high_freq)),
            'noise_uniformity': NoiseAnalyzer._compute_noise_uniformity(noise_residual)
        }
        
        return features
    
    @staticmethod
    def _compute_noise_uniformity(noise: np.ndarray) -> float:
        """Check if noise is uniformly distributed (natural) or patterned (artificial)."""
        # Divide image into blocks and compute noise variance in each
        h, w = noise.shape
        block_size = 32
        variances = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = noise[i:i + block_size, j:j + block_size]
                variances.append(np.var(block))
        
        if len(variances) < 2:
            return 1.0
        
        # Coefficient of variation of variances
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        cv = std_var / (mean_var + 1e-8)
        
        # Lower CV means more uniform noise (natural)
        return 1.0 / (1.0 + cv)
    
    @staticmethod
    def get_noise_suspicion_score(features: Dict[str, float]) -> float:
        """
        Compute suspicion score based on noise features.
        """
        score = 0.5
        
        # Very low noise might indicate AI generation or heavy processing
        if features['noise_std'] < 2.0:
            score += 0.15
        
        # Very uniform noise is suspicious (might be added artificially)
        if features['noise_uniformity'] > 0.9:
            score += 0.1
        
        return np.clip(score, 0, 1)


class CompressionAnalyzer:
    """
    Analyze JPEG compression artifacts.
    
    AI-generated images often have different compression characteristics.
    """
    
    @staticmethod
    def compute_compression_features(image: np.ndarray) -> Dict[str, float]:
        """
        Analyze compression artifacts.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)
        
        # Detect 8x8 block artifacts (JPEG compression)
        block_artifacts = CompressionAnalyzer._detect_block_artifacts(gray)
        
        # DCT domain analysis
        dct_features = CompressionAnalyzer._analyze_dct(gray)
        
        features = {
            'block_artifact_strength': block_artifacts,
            **dct_features
        }
        
        return features
    
    @staticmethod
    def _detect_block_artifacts(gray: np.ndarray) -> float:
        """Detect 8x8 block artifacts from JPEG compression."""
        h, w = gray.shape
        
        # Compute horizontal and vertical edge differences at 8-pixel intervals
        h_edges = []
        v_edges = []
        
        for i in range(8, h - 8, 8):
            h_diff = np.abs(gray[i, :] - gray[i - 1, :])
            h_edges.append(np.mean(h_diff))
        
        for j in range(8, w - 8, 8):
            v_diff = np.abs(gray[:, j] - gray[:, j - 1])
            v_edges.append(np.mean(v_diff))
        
        # Compare to non-boundary edges
        non_boundary_edges = []
        for i in range(1, h - 1):
            if i % 8 != 0:
                diff = np.abs(gray[i, :] - gray[i - 1, :])
                non_boundary_edges.append(np.mean(diff))
        
        if len(non_boundary_edges) == 0 or len(h_edges) == 0:
            return 0.0
        
        # Block artifact strength
        boundary_mean = (np.mean(h_edges) + np.mean(v_edges)) / 2
        non_boundary_mean = np.mean(non_boundary_edges)
        
        artifact_strength = boundary_mean / (non_boundary_mean + 1e-8)
        
        return artifact_strength
    
    @staticmethod
    def _analyze_dct(gray: np.ndarray) -> Dict[str, float]:
        """Analyze DCT coefficients."""
        from scipy.fftpack import dct
        
        # Apply DCT to 8x8 blocks
        h, w = gray.shape
        dct_coeffs = []
        
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i + 8, j:j + 8]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.append(dct_block.flatten())
        
        if len(dct_coeffs) == 0:
            return {'dct_energy': 0.0, 'dct_high_freq_ratio': 0.0}
        
        dct_coeffs = np.array(dct_coeffs)
        
        # Analyze DCT coefficient distribution
        low_freq_energy = np.mean(np.abs(dct_coeffs[:, :16]))
        high_freq_energy = np.mean(np.abs(dct_coeffs[:, 16:]))
        
        return {
            'dct_energy': np.mean(np.abs(dct_coeffs)),
            'dct_high_freq_ratio': high_freq_energy / (low_freq_energy + 1e-8)
        }
    
    @staticmethod
    def get_compression_suspicion_score(features: Dict[str, float]) -> float:
        """
        Compute suspicion score based on compression features.
        """
        score = 0.5
        
        # Very low block artifacts might indicate AI generation
        if features['block_artifact_strength'] < 0.8:
            score += 0.1
        
        # Unusual DCT distribution
        if features.get('dct_high_freq_ratio', 0) < 0.1:
            score += 0.1
        
        return np.clip(score, 0, 1)


class EnhancedDetector:
    """
    Main class for enhanced detection using multiple analysis methods.
    """
    
    def __init__(self):
        self.frequency_analyzer = FrequencyAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.compression_analyzer = CompressionAnalyzer()
    
    def analyze(self, image: np.ndarray) -> EnhancedFeatures:
        """
        Perform comprehensive analysis on an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            EnhancedFeatures with all analysis scores
        """
        try:
            # Compute all features
            freq_features = self.frequency_analyzer.compute_fft_features(image)
            texture_features = self.texture_analyzer.compute_texture_features(image)
            color_features = self.color_analyzer.compute_color_features(image)
            noise_features = self.noise_analyzer.compute_noise_features(image)
            compression_features = self.compression_analyzer.compute_compression_features(image)
            
            # Compute suspicion scores
            freq_score = self.frequency_analyzer.get_frequency_suspicion_score(freq_features)
            texture_score = self.texture_analyzer.get_texture_suspicion_score(texture_features)
            color_score = self.color_analyzer.get_color_suspicion_score(color_features)
            noise_score = self.noise_analyzer.get_noise_suspicion_score(noise_features)
            compression_score = self.compression_analyzer.get_compression_suspicion_score(compression_features)
            
            # Weighted ensemble
            weights = {
                'frequency': 0.25,
                'texture': 0.20,
                'color': 0.15,
                'noise': 0.25,
                'compression': 0.15
            }
            
            overall = (
                weights['frequency'] * freq_score +
                weights['texture'] * texture_score +
                weights['color'] * color_score +
                weights['noise'] * noise_score +
                weights['compression'] * compression_score
            )
            
            return EnhancedFeatures(
                frequency_score=freq_score,
                texture_score=texture_score,
                color_consistency_score=color_score,
                noise_analysis_score=noise_score,
                compression_artifacts_score=compression_score,
                overall_suspicion=overall,
                features={
                    'frequency': freq_features,
                    'texture': texture_features,
                    'color': color_features,
                    'noise': noise_features,
                    'compression': compression_features
                }
            )
            
        except Exception as e:
            logger.warning(f"Enhanced analysis failed: {e}")
            return EnhancedFeatures()


# Singleton instance
_enhanced_detector = None

def get_enhanced_detector() -> EnhancedDetector:
    """Get singleton enhanced detector instance."""
    global _enhanced_detector
    if _enhanced_detector is None:
        _enhanced_detector = EnhancedDetector()
    return _enhanced_detector
