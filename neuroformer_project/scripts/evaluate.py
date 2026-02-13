"""
Evaluation script for NeuroFormer.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from neuroformer.models.neuroformer import create_neuroformer
from neuroformer.data.dataset import create_dataloaders


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            eyetracking = batch['eyetracking'].to(device)
            behavioral = batch['behavioral'].to(device)
            labels = batch['label']
            
            outputs = model(eeg, eyetracking, behavioral)
            probs = outputs['probs'].cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auroc': roc_auc_score(all_labels, all_probs)
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description='Evaluate NeuroFormer')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load config (assuming it's in same directory)
    config_path = Path(args.checkpoint).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_neuroformer(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Load data
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        modalities=config['data']['modalities'],
        max_seq_len=config['data']['max_seq_len']
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics, cm = evaluate_model(model, test_loader, args.device)
    
    # Print results
    print("\n" + "=" * 50)
    print("Test Set Results")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("=" * 50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'metrics.npy', metrics)
    np.save(output_dir / 'confusion_matrix.npy', cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
