"""
Training script for NeuroFormer.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from neuroformer.models.neuroformer import create_neuroformer
from neuroformer.data.dataset import create_dataloaders
from neuroformer.training.trainer import NeuroFormerTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train NeuroFormer')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("=" * 80)
    print(f"Training {config['model']['name']}")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Create model
    print("\nCreating model...")
    model = create_neuroformer(config['model'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        modalities=config['data']['modalities'],
        max_seq_len=config['data']['max_seq_len']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = NeuroFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=args.device
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
