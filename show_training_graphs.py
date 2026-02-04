"""
Training History Visualization Script
Displays training and validation accuracy/loss graphs
"""

import json
import matplotlib.pyplot as plt
import os

def show_training_history():
    history_file = os.path.join(os.path.dirname(__file__), 'training_history.json')
    
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    model_type = data.get('model_type', 'Model')
    best_val_acc = data.get('best_val_acc', 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    ax1.axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.7, label=f'Best Val Acc: {best_val_acc:.2f}%')
    ax1.set_title(f'{model_type.replace("_", " ").title()} Detector - Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(min(history['train_acc']), min(history['val_acc'])) - 2, 101])
    
    # Loss plot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax2.set_title(f'{model_type.replace("_", " ").title()} Detector - Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'DeepFake Detection Training History\nBest Validation Accuracy: {best_val_acc:.2f}%', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Graph saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("📈 TRAINING SUMMARY")
    print("="*50)
    print(f"Model Type: {model_type.replace('_', ' ').title()}")
    print(f"Epochs Completed: {data.get('epochs_completed', len(epochs))}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Overfitting Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%")
    print("="*50)
    
    plt.show()

if __name__ == "__main__":
    show_training_history()
