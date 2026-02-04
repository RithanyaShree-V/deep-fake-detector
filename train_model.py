"""
DeepFake Detection Model Training Script
=========================================
Fine-tune the detection models on your own dataset.

Features:
- GPU acceleration when available
- Early stopping to prevent overfitting
- Learning rate scheduling
- Data augmentation
- Dropout regularization
- Weight decay (L2 regularization)

Usage:
    python train_model.py --data_dir path/to/dataset --epochs 10
    python train_model.py --data_dir path/to/dataset --model deepfake --epochs 20
    python train_model.py --quick_train  # Quick training with synthetic data
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_loader import DeepFakeDetector, AIGeneratedDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device():
    """Setup the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        logger.info("⚠️ No GPU detected, using CPU (training will be slower)")
    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class DeepFakeDataset(Dataset):
    """Dataset for deepfake detection training."""
    
    def __init__(
        self, 
        real_dir: str, 
        fake_dir: str, 
        transform=None,
        max_samples: int = None
    ):
        self.transform = transform
        self.samples = []
        
        # Collect real images (label = 0)
        real_path = Path(real_dir)
        if real_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in real_path.glob(ext):
                    self.samples.append((str(img_path), 0))
                for img_path in real_path.glob(ext.upper()):
                    self.samples.append((str(img_path), 0))
        
        # Collect fake images (label = 1)
        fake_path = Path(fake_dir)
        if fake_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in fake_path.glob(ext):
                    self.samples.append((str(img_path), 1))
                for img_path in fake_path.glob(ext.upper()):
                    self.samples.append((str(img_path), 1))
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        # Balance dataset
        self._balance_dataset()
        
        logger.info(f"Dataset created with {len(self.samples)} samples")
    
    def _balance_dataset(self):
        """Balance the dataset to have equal real and fake samples."""
        real_samples = [s for s in self.samples if s[1] == 0]
        fake_samples = [s for s in self.samples if s[1] == 1]
        
        min_count = min(len(real_samples), len(fake_samples))
        if min_count > 0:
            np.random.shuffle(real_samples)
            np.random.shuffle(fake_samples)
            self.samples = real_samples[:min_count] + fake_samples[:min_count]
            np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a blank image on error
            logger.warning(f"Error loading {img_path}: {e}")
            blank = torch.zeros(3, 224, 224)
            return blank, label


def get_train_transforms():
    """Get training data augmentation transforms (anti-overfitting)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_synthetic_training_data(num_samples: int = 500) -> Tuple[str, str]:
    """Create synthetic training data for quick training demo."""
    import cv2
    
    train_dir = Path('training_data')
    real_dir = train_dir / 'real'
    fake_dir = train_dir / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_samples} synthetic training samples...")
    
    for i in tqdm(range(num_samples), desc="Creating samples"):
        # Create "real" images - smooth gradients, natural colors
        np.random.seed(i)
        real_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Natural color gradients
        for c in range(3):
            base_color = np.random.randint(80, 180)
            gradient = np.linspace(base_color - 30, base_color + 30, 224)
            real_img[:, :, c] = np.tile(gradient, (224, 1)).astype(np.uint8)
        
        # Add subtle noise (natural images have noise)
        noise = np.random.normal(0, 5, real_img.shape).astype(np.int16)
        real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some blur (natural images aren't perfectly sharp)
        real_img = cv2.GaussianBlur(real_img, (3, 3), 0.5)
        
        cv2.imwrite(str(real_dir / f'real_{i:04d}.jpg'), real_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        # Create "fake" images - artificial patterns, sharp edges, artifacts
        np.random.seed(i + 10000)
        fake_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Artificial patterns
        for c in range(3):
            # Grid-like artifacts (common in GAN-generated images)
            pattern = np.zeros((224, 224), dtype=np.uint8)
            base = np.random.randint(50, 200)
            pattern[:] = base
            
            # Add periodic artifacts
            for y in range(0, 224, 8):
                pattern[y:y+2, :] = np.clip(base + np.random.randint(-20, 20), 0, 255)
            for x in range(0, 224, 8):
                pattern[:, x:x+2] = np.clip(base + np.random.randint(-20, 20), 0, 255)
            
            fake_img[:, :, c] = pattern
        
        # Add sharp color transitions (unnatural)
        if np.random.random() > 0.5:
            x1, y1 = np.random.randint(0, 150, 2)
            x2, y2 = x1 + np.random.randint(30, 70), y1 + np.random.randint(30, 70)
            fake_img[y1:y2, x1:x2] = np.random.randint(100, 255, 3)
        
        # No blur (AI images often lack natural blur)
        cv2.imwrite(str(fake_dir / f'fake_{i:04d}.jpg'), fake_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    logger.info(f"Created {num_samples} real and {num_samples} fake samples")
    return str(real_dir), str(fake_dir)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def train_model(
    model_type: str,
    real_dir: str,
    fake_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    val_split: float = 0.2,
    patience: int = 5,
    weight_decay: float = 0.01
):
    """
    Main training function with anti-overfitting measures.
    
    Anti-overfitting techniques used:
    1. Early stopping with patience
    2. Weight decay (L2 regularization)
    3. Dropout in model architecture
    4. Data augmentation
    5. Learning rate scheduling
    6. Gradual unfreezing of backbone
    7. Label smoothing
    """
    # Setup device (GPU if available)
    device = setup_device()
    
    # Create model
    if model_type == 'deepfake':
        model = DeepFakeDetector(pretrained=True)
        save_path = Config.MODEL_PATH / 'deepfake_detector.pth'
    else:
        model = AIGeneratedDetector(pretrained=True)
        save_path = Config.MODEL_PATH / 'ai_generated_detector.pth'
    
    # Ensure model directory exists
    Config.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    # Freeze backbone initially (transfer learning)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Create dataset with augmentation
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    full_dataset = DeepFakeDataset(real_dir, fake_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        logger.error("No training data found!")
        return None
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create separate dataset for validation with no augmentation
    val_dataset_clean = DeepFakeDataset(real_dir, fake_dir, transform=val_transform)
    val_indices = val_dataset.indices
    val_dataset_clean.samples = [val_dataset_clean.samples[i] for i in val_indices if i < len(val_dataset_clean.samples)]
    
    # DataLoader settings
    num_workers = 0  # Windows compatibility
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Loss with label smoothing (prevents overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.5, mode='max')
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print("\n" + "="*60)
    print(f"🚀 TRAINING {model_type.upper()} DETECTOR")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Epochs: {epochs} (with early stopping, patience={patience})")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print("="*60 + "\n")
    
    for epoch in range(1, epochs + 1):
        # Gradual unfreezing: unfreeze backbone after 2 epochs
        if epoch == 3:
            logger.info("🔓 Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Lower learning rate for pretrained layers
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': learning_rate * 0.01},
                {'params': model.classifier.parameters(), 'lr': learning_rate * 0.1}
            ], weight_decay=weight_decay)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Calculate overfitting indicator
        overfit_gap = train_acc - val_acc
        overfit_status = "⚠️ OVERFITTING" if overfit_gap > 10 else "✓ OK"
        
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Gap: {overfit_gap:.2f}% {overfit_status}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  💾 Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Check early stopping
        if early_stopping(val_acc):
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    # Save training history
    history_path = Path('training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'epochs_completed': len(history['train_loss']),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'history': history,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'anti_overfitting': {
                'early_stopping_patience': patience,
                'weight_decay': weight_decay,
                'label_smoothing': 0.1,
                'data_augmentation': True
            }
        }, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train DeepFake Detection Models with Anti-Overfitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Anti-overfitting techniques used:
  - Early stopping with configurable patience
  - Weight decay (L2 regularization)
  - Label smoothing (0.1)
  - Data augmentation (rotation, flip, color jitter, etc.)
  - Learning rate scheduling (ReduceLROnPlateau)
  - Gradual backbone unfreezing

Examples:
  python train_model.py --quick_train --epochs 10
  python train_model.py --data_dir ./dataset --epochs 20 --patience 5
  python train_model.py --real_dir ./real --fake_dir ./fake --weight_decay 0.02
        """
    )
    parser.add_argument('--data_dir', type=str, help='Directory with real/ and fake/ subfolders')
    parser.add_argument('--real_dir', type=str, help='Directory containing real images')
    parser.add_argument('--fake_dir', type=str, help='Directory containing fake images')
    parser.add_argument('--model', type=str, default='both', choices=['deepfake', 'ai_generated', 'both'],
                        help='Which model to train (default: both)')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 regularization (default: 0.01)')
    parser.add_argument('--quick_train', action='store_true', help='Quick training with synthetic data')
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("\n" + "="*60)
    print("🔍 SYSTEM CHECK")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - training will use CPU (slower)")
    print("="*60)
    
    # Determine directories
    if args.quick_train:
        real_dir, fake_dir = create_synthetic_training_data(500)
    elif args.data_dir:
        data_path = Path(args.data_dir)
        real_dir = str(data_path / 'real')
        fake_dir = str(data_path / 'fake')
    elif args.real_dir and args.fake_dir:
        real_dir = args.real_dir
        fake_dir = args.fake_dir
    else:
        parser.print_help()
        print("\n" + "-"*60)
        print("Please specify training data using one of:")
        print("  --data_dir path/to/dataset  (with real/ and fake/ subfolders)")
        print("  --real_dir path/to/real --fake_dir path/to/fake")
        print("  --quick_train  (use synthetic data for demo)")
        return
    
    # Train models
    if args.model in ['deepfake', 'both']:
        train_model(
            'deepfake', real_dir, fake_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr,
            patience=args.patience,
            weight_decay=args.weight_decay
        )
    
    if args.model in ['ai_generated', 'both']:
        train_model(
            'ai_generated', real_dir, fake_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr,
            patience=args.patience,
            weight_decay=args.weight_decay
        )
    
    print("\n🎉 All training complete!")


if __name__ == '__main__':
    main()
