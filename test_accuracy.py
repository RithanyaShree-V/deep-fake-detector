"""
DeepFake Detection - Accuracy Testing Script
=============================================
This script tests the model accuracy on a dataset of real and fake images.

Usage:
    python test_accuracy.py --real_dir path/to/real --fake_dir path/to/fake
    python test_accuracy.py --test_dir path/to/test  # with 'real' and 'fake' subfolders
    python test_accuracy.py --single path/to/image.jpg  # test single image
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_inference_engine():
    """Load the inference engine."""
    print("Loading DeepFake Detection models...")
    from inference import get_inference_engine
    engine = get_inference_engine()
    print(f"✓ Models loaded successfully (Device: {engine.device})")
    return engine


def get_image_files(directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> List[Path]:
    """Get all image files from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)


def test_single_image(engine, image_path: str) -> Dict:
    """Test a single image and return detailed results."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)
    
    result = engine.analyze_image(image_path)
    
    print(f"\n📊 Results:")
    print(f"   Verdict: {'🔴 FAKE' if result.is_fake else '🟢 REAL'}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   DeepFake Score: {result.deepfake_score:.1%}")
    print(f"   AI-Generated Score: {result.ai_generated_score:.1%}")
    print(f"   Faces Detected: {result.faces_detected}")
    print(f"   Processing Time: {result.processing_time:.3f}s")
    
    if result.face_regions:
        print(f"\n   Face Regions:")
        for i, face in enumerate(result.face_regions):
            print(f"      Face {i+1}: {face}")
    
    return {
        'path': str(image_path),
        'is_fake': result.is_fake,
        'confidence': result.confidence,
        'deepfake_score': result.deepfake_score,
        'ai_generated_score': result.ai_generated_score,
        'faces_detected': result.faces_detected,
        'processing_time': result.processing_time
    }


def calculate_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict:
    """Calculate accuracy metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Basic metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive/negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'total_samples': total,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }


def test_dataset(engine, real_dir: str, fake_dir: str, save_results: bool = True) -> Dict:
    """Test accuracy on a dataset with real and fake images."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION ACCURACY TEST")
    print("="*60)
    
    # Get image files
    real_images = get_image_files(real_dir)
    fake_images = get_image_files(fake_dir)
    
    print(f"\n📁 Dataset:")
    print(f"   Real images: {len(real_images)} files from {real_dir}")
    print(f"   Fake images: {len(fake_images)} files from {fake_dir}")
    print(f"   Total: {len(real_images) + len(fake_images)} images")
    
    if len(real_images) == 0 and len(fake_images) == 0:
        print("\n❌ No images found! Please check your directories.")
        return {}
    
    # Ground truth and predictions
    y_true = []  # 0 = real, 1 = fake
    y_pred = []
    y_scores = []
    results = []
    errors = []
    
    # Test real images (expected: is_fake = False)
    print(f"\n🔍 Testing REAL images...")
    for img_path in tqdm(real_images, desc="Real images"):
        try:
            result = engine.analyze_image(str(img_path))
            y_true.append(0)  # Ground truth: real
            y_pred.append(1 if result.is_fake else 0)
            y_scores.append(result.confidence if result.is_fake else 1 - result.confidence)
            results.append({
                'path': str(img_path),
                'ground_truth': 'real',
                'predicted': 'fake' if result.is_fake else 'real',
                'correct': not result.is_fake,
                'confidence': result.confidence,
                'deepfake_score': result.deepfake_score,
                'ai_generated_score': result.ai_generated_score
            })
        except Exception as e:
            errors.append({'path': str(img_path), 'error': str(e)})
    
    # Test fake images (expected: is_fake = True)
    print(f"\n🔍 Testing FAKE images...")
    for img_path in tqdm(fake_images, desc="Fake images"):
        try:
            result = engine.analyze_image(str(img_path))
            y_true.append(1)  # Ground truth: fake
            y_pred.append(1 if result.is_fake else 0)
            y_scores.append(result.confidence if result.is_fake else 1 - result.confidence)
            results.append({
                'path': str(img_path),
                'ground_truth': 'fake',
                'predicted': 'fake' if result.is_fake else 'real',
                'correct': result.is_fake,
                'confidence': result.confidence,
                'deepfake_score': result.deepfake_score,
                'ai_generated_score': result.ai_generated_score
            })
        except Exception as e:
            errors.append({'path': str(img_path), 'error': str(e)})
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    # Print results
    print("\n" + "="*60)
    print("📊 ACCURACY RESULTS")
    print("="*60)
    
    print(f"\n   📈 Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"\n   Confusion Matrix:")
    print(f"   ┌─────────────┬──────────────┬──────────────┐")
    print(f"   │             │ Pred: REAL   │ Pred: FAKE   │")
    print(f"   ├─────────────┼──────────────┼──────────────┤")
    print(f"   │ True: REAL  │ TN: {metrics['true_negatives']:>6}   │ FP: {metrics['false_positives']:>6}   │")
    print(f"   │ True: FAKE  │ FN: {metrics['false_negatives']:>6}   │ TP: {metrics['true_positives']:>6}   │")
    print(f"   └─────────────┴──────────────┴──────────────┘")
    
    print(f"\n   Detailed Metrics:")
    print(f"   • Precision (FAKE detection): {metrics['precision']:.1%}")
    print(f"   • Recall (FAKE detection):    {metrics['recall']:.1%}")
    print(f"   • F1 Score:                   {metrics['f1_score']:.1%}")
    print(f"   • Specificity (REAL correct): {metrics['specificity']:.1%}")
    print(f"   • False Positive Rate:        {metrics['false_positive_rate']:.1%}")
    print(f"   • False Negative Rate:        {metrics['false_negative_rate']:.1%}")
    
    if errors:
        print(f"\n   ⚠️ Errors: {len(errors)} images failed to process")
    
    # Show misclassified examples
    misclassified = [r for r in results if not r['correct']]
    if misclassified:
        print(f"\n   ❌ Misclassified samples ({len(misclassified)}):")
        for i, m in enumerate(misclassified[:10]):  # Show first 10
            print(f"      {i+1}. {Path(m['path']).name}")
            print(f"         True: {m['ground_truth']}, Pred: {m['predicted']}, Conf: {m['confidence']:.1%}")
        if len(misclassified) > 10:
            print(f"      ... and {len(misclassified) - 10} more")
    
    # Save results to JSON
    if save_results:
        output = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'real_dir': real_dir,
                'fake_dir': fake_dir,
                'real_count': len(real_images),
                'fake_count': len(fake_images)
            },
            'metrics': metrics,
            'detailed_results': results,
            'errors': errors
        }
        
        output_path = Path('accuracy_results.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n   💾 Results saved to: {output_path}")
    
    return metrics


def create_sample_test_data():
    """Create sample test directories for demonstration."""
    print("\n📁 Creating sample test directories...")
    
    test_dir = Path('test_data')
    real_dir = test_dir / 'real'
    fake_dir = test_dir / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images for testing
    from PIL import Image
    import numpy as np
    
    # Create 5 "real" sample images (natural looking)
    for i in range(5):
        np.random.seed(i)
        # Create a more natural-looking gradient image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        for c in range(3):
            base = np.random.randint(100, 200)
            img_array[:, :, c] = base + np.random.randint(-20, 20, (224, 224))
        img = Image.fromarray(img_array)
        img.save(real_dir / f'real_sample_{i+1}.jpg')
    
    # Create 5 "fake" sample images (artificial patterns)
    for i in range(5):
        np.random.seed(i + 100)
        # Create more artificial-looking images
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        for c in range(3):
            img_array[:, :, c] = np.random.randint(0, 255, (224, 224))
        img = Image.fromarray(img_array)
        img.save(fake_dir / f'fake_sample_{i+1}.jpg')
    
    print(f"   ✓ Created {5} real samples in {real_dir}")
    print(f"   ✓ Created {5} fake samples in {fake_dir}")
    print(f"\n   Note: These are synthetic test images. For accurate testing,")
    print(f"   use real deepfake datasets like FaceForensics++ or DFDC.")
    
    return str(real_dir), str(fake_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Test DeepFake Detection Accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single image
  python test_accuracy.py --single path/to/image.jpg
  
  # Test with separate real and fake directories
  python test_accuracy.py --real_dir ./real_images --fake_dir ./fake_images
  
  # Test with a directory containing 'real' and 'fake' subfolders
  python test_accuracy.py --test_dir ./my_dataset
  
  # Create sample test data and run test
  python test_accuracy.py --create_samples
  
Recommended Datasets:
  - FaceForensics++ (FF++): https://github.com/ondyari/FaceForensics
  - Deepfake Detection Challenge (DFDC): https://www.kaggle.com/c/deepfake-detection-challenge
  - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
        """
    )
    
    parser.add_argument('--single', type=str, help='Path to a single image to test')
    parser.add_argument('--real_dir', type=str, help='Directory containing real images')
    parser.add_argument('--fake_dir', type=str, help='Directory containing fake images')
    parser.add_argument('--test_dir', type=str, help='Directory with real/ and fake/ subfolders')
    parser.add_argument('--create_samples', action='store_true', help='Create sample test data')
    parser.add_argument('--no_save', action='store_true', help='Do not save results to JSON')
    
    args = parser.parse_args()
    
    # Load inference engine
    engine = load_inference_engine()
    
    # Handle different modes
    if args.single:
        # Test single image
        test_single_image(engine, args.single)
        
    elif args.create_samples:
        # Create sample data and test
        real_dir, fake_dir = create_sample_test_data()
        test_dataset(engine, real_dir, fake_dir, save_results=not args.no_save)
        
    elif args.test_dir:
        # Test directory with subfolders
        test_path = Path(args.test_dir)
        real_dir = test_path / 'real'
        fake_dir = test_path / 'fake'
        
        if not real_dir.exists() or not fake_dir.exists():
            print(f"❌ Error: {args.test_dir} must contain 'real/' and 'fake/' subfolders")
            sys.exit(1)
            
        test_dataset(engine, str(real_dir), str(fake_dir), save_results=not args.no_save)
        
    elif args.real_dir and args.fake_dir:
        # Test with explicit directories
        test_dataset(engine, args.real_dir, args.fake_dir, save_results=not args.no_save)
        
    else:
        # No arguments - show help and offer to create samples
        parser.print_help()
        print("\n" + "-"*60)
        response = input("\nWould you like to create sample test data? (y/n): ")
        if response.lower() in ('y', 'yes'):
            real_dir, fake_dir = create_sample_test_data()
            test_dataset(engine, real_dir, fake_dir, save_results=True)


if __name__ == '__main__':
    main()
