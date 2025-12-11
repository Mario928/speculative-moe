"""
Generate training curves from training_history.json
Creates plots for loss, accuracy, and validation metrics
"""
import json
import matplotlib.pyplot as plt
import os

def plot_training_curves(history_file, output_dir='.'):
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_top1 = [h['val_top1'] for h in history]
    val_top2 = [h['val_top2'] for h in history]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, val_top1, 'g-', linewidth=2, label='Val Top-1')
    ax2.plot(epochs, val_top2, 'r-', linewidth=2, label='Val Top-2')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add final values annotation
    final_top1 = val_top1[-1]
    final_top2 = val_top2[-1]
    ax2.annotate(f'Final: {final_top1:.1f}%', 
                xy=(epochs[-1], final_top1), 
                xytext=(epochs[-1]-3, final_top1-5),
                fontsize=10, fontweight='bold')
    ax2.annotate(f'Final: {final_top2:.1f}%', 
                xy=(epochs[-1], final_top2), 
                xytext=(epochs[-1]-3, final_top2+2),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default='checkpoints/training_history.json')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()
    
    plot_training_curves(args.history, args.output_dir)
