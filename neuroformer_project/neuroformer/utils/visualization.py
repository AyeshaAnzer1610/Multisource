"""Visualization utilities for NeuroFormer."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(epochs, val_accs, 'g-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Healthy', 'Disorder'],
    save_path: Optional[str] = None
):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    modality_names: List[str],
    save_path: Optional[str] = None
):
    """Plot attention weights across modalities."""
    plt.figure(figsize=(10, 6))
    
    attention_np = attention_weights.cpu().numpy()
    
    plt.imshow(attention_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Token Position')
    plt.ylabel('Head')
    plt.title('Cross-Modal Attention Weights')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
