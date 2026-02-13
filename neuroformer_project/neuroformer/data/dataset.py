"""
Dataset classes for NeuroFormer training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import h5py


class MultimodalMentalHealthDataset(Dataset):
    """Dataset for multimodal mental health data."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        modalities: List[str] = ['eeg', 'eyetracking', 'behavioral'],
        max_seq_len: int = 500,
        augment: bool = False
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npz files
            split: 'train', 'val', or 'test'
            modalities: List of modalities to load
            max_seq_len: Maximum sequence length (for padding)
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        # Load data
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples for the split."""
        split_file = os.path.join(self.data_dir, f'{self.split}_samples.npz')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        data = np.load(split_file, allow_pickle=True)
        
        samples = []
        n_samples = len(data['labels'])
        
        for i in range(n_samples):
            sample = {
                'participant_id': data['participant_ids'][i],
                'label': int(data['labels'][i]),
                'eeg': data['eeg_features'][i] if 'eeg' in self.modalities else None,
                'eyetracking': data['eye_features'][i] if 'eyetracking' in self.modalities else None,
                'behavioral': data['beh_features'][i] if 'behavioral' in self.modalities else None
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and pad sequences
        output = {'label': torch.tensor(sample['label'], dtype=torch.long)}
        
        for modality in self.modalities:
            if sample[modality] is not None:
                seq = torch.tensor(sample[modality], dtype=torch.float32)
                
                # Apply augmentation if enabled
                if self.augment and self.split == 'train':
                    seq = self._augment_sequence(seq)
                
                # Pad or truncate
                seq_len = seq.size(0)
                if seq_len < self.max_seq_len:
                    # Pad
                    pad_len = self.max_seq_len - seq_len
                    seq = torch.cat([seq, torch.zeros(pad_len, seq.size(1))], dim=0)
                    mask = torch.cat([torch.zeros(seq_len), torch.ones(pad_len)]).bool()
                else:
                    # Truncate
                    seq = seq[:self.max_seq_len]
                    mask = torch.zeros(self.max_seq_len).bool()
                
                output[modality] = seq
                output[f'{modality}_mask'] = mask
        
        return output
    
    def _augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to sequence."""
        # Jittering (add Gaussian noise)
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(seq) * 0.01
            seq = seq + noise
        
        # Time shifting
        if torch.rand(1) < 0.3:
            shift = torch.randint(-5, 6, (1,)).item()
            if shift > 0:
                seq = torch.cat([torch.zeros(shift, seq.size(1)), seq[:-shift]], dim=0)
            elif shift < 0:
                seq = torch.cat([seq[-shift:], torch.zeros(-shift, seq.size(1))], dim=0)
        
        # Feature dropout
        if torch.rand(1) < 0.2:
            dropout_mask = torch.rand(seq.size(1)) > 0.1
            seq = seq * dropout_mask.unsqueeze(0)
        
        return seq


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_dataset = MultimodalMentalHealthDataset(
        data_dir,
        split='train',
        augment=True,
        **dataset_kwargs
    )
    
    val_dataset = MultimodalMentalHealthDataset(
        data_dir,
        split='val',
        augment=False,
        **dataset_kwargs
    )
    
    test_dataset = MultimodalMentalHealthDataset(
        data_dir,
        split='test',
        augment=False,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
