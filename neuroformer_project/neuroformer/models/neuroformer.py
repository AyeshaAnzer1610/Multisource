"""
NeuroFormer: Transformer-Based Multimodal Integration for Mental Health Diagnosis

Main model architecture implementing:
- Modality-specific transformer encoders
- Cross-modal fusion transformer
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ModalityEncoder(nn.Module):
    """Modality-specific transformer encoder."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        max_len: int = 5000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
        
        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # Create attention mask if padding mask provided
        if mask is not None:
            # Convert padding mask to attention mask
            # mask: True for padded positions
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return x


class CrossModalFusionTransformer(nn.Module):
    """Cross-modal fusion transformer for integrating multiple modalities."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        n_modalities: int = 3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_modalities = n_modalities
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Modality type embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(n_modalities, d_model)
        )
        
        # Positional encoding for concatenated sequence
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        encoded_modalities: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded_modalities: Dictionary of encoded modality tensors
                Each tensor has shape (batch_size, seq_len, d_model)
            modality_masks: Optional dictionary of padding masks
        
        Returns:
            cls_output: CLS token representation (batch_size, d_model)
            full_output: Full sequence output (batch_size, total_seq_len, d_model)
        """
        batch_size = next(iter(encoded_modalities.values())).size(0)
        device = next(iter(encoded_modalities.values())).device
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate all modalities with modality type embeddings
        sequences = [cls_tokens]
        full_mask = []
        
        modality_names = ['eeg', 'eyetracking', 'behavioral']
        for idx, modality_name in enumerate(modality_names):
            if modality_name in encoded_modalities:
                modality_seq = encoded_modalities[modality_name]
                seq_len = modality_seq.size(1)
                
                # Add modality type embedding
                modality_emb = self.modality_embeddings[idx].view(1, 1, -1)
                modality_emb = modality_emb.expand(batch_size, seq_len, -1)
                modality_seq = modality_seq + modality_emb
                
                sequences.append(modality_seq)
                
                # Handle mask
                if modality_masks is not None and modality_name in modality_masks:
                    full_mask.append(modality_masks[modality_name])
                else:
                    full_mask.append(
                        torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                    )
        
        # Concatenate along sequence dimension
        x = torch.cat(sequences, dim=1)  # (batch_size, total_seq_len, d_model)
        
        # Concatenate masks (CLS token is never masked)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        if modality_masks is not None:
            combined_mask = torch.cat([cls_mask] + full_mask, dim=1)
        else:
            combined_mask = None
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply fusion transformer
        x = self.fusion_transformer(
            x,
            src_key_padding_mask=combined_mask
        )
        
        # Extract CLS token
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        
        return cls_output, x


class NeuroFormer(nn.Module):
    """
    NeuroFormer: Transformer-based multimodal framework for mental health diagnosis.
    
    Integrates EEG, eye-tracking, and behavioral data through:
    1. Modality-specific transformer encoders
    2. Cross-modal fusion transformer
    3. Classification head
    """
    
    def __init__(
        self,
        eeg_input_dim: int = 128,
        eyetracking_input_dim: int = 96,
        behavioral_input_dim: int = 64,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_fusion_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        n_classes: int = 2,
        use_auxiliary_loss: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_auxiliary_loss = use_auxiliary_loss
        
        # Modality-specific encoders
        self.eeg_encoder = ModalityEncoder(
            input_dim=eeg_input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.eyetracking_encoder = ModalityEncoder(
            input_dim=eyetracking_input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.behavioral_encoder = ModalityEncoder(
            input_dim=behavioral_input_dim,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Cross-modal fusion transformer
        self.fusion_transformer = CrossModalFusionTransformer(
            d_model=d_model,
            n_layers=n_fusion_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_modalities=3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        # Auxiliary reconstruction decoders (for NeuroFormer++)
        if use_auxiliary_loss:
            self.eeg_decoder = nn.Linear(d_model, eeg_input_dim)
            self.eyetracking_decoder = nn.Linear(d_model, eyetracking_input_dim)
            self.behavioral_decoder = nn.Linear(d_model, behavioral_input_dim)
    
    def forward(
        self,
        eeg: torch.Tensor,
        eyetracking: torch.Tensor,
        behavioral: torch.Tensor,
        eeg_mask: Optional[torch.Tensor] = None,
        eyetracking_mask: Optional[torch.Tensor] = None,
        behavioral_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NeuroFormer.
        
        Args:
            eeg: EEG features (batch_size, seq_len, eeg_dim)
            eyetracking: Eye-tracking features (batch_size, seq_len, eye_dim)
            behavioral: Behavioral features (batch_size, seq_len, beh_dim)
            *_mask: Optional padding masks for each modality
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - logits: Classification logits (batch_size, n_classes)
                - probs: Class probabilities (batch_size, n_classes)
                - reconstructions: (optional) Reconstructed inputs for auxiliary loss
        """
        # Encode each modality
        eeg_encoded = self.eeg_encoder(eeg, eeg_mask)
        eyetracking_encoded = self.eyetracking_encoder(eyetracking, eyetracking_mask)
        behavioral_encoded = self.behavioral_encoder(behavioral, behavioral_mask)
        
        # Prepare encoded modalities dictionary
        encoded_modalities = {
            'eeg': eeg_encoded,
            'eyetracking': eyetracking_encoded,
            'behavioral': behavioral_encoded
        }
        
        # Prepare masks dictionary
        modality_masks = {}
        if eeg_mask is not None:
            modality_masks['eeg'] = eeg_mask
        if eyetracking_mask is not None:
            modality_masks['eyetracking'] = eyetracking_mask
        if behavioral_mask is not None:
            modality_masks['behavioral'] = behavioral_mask
        
        # Cross-modal fusion
        cls_output, full_output = self.fusion_transformer(
            encoded_modalities,
            modality_masks if modality_masks else None
        )
        
        # Classification
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=-1)
        
        # Prepare output dictionary
        output = {
            'logits': logits,
            'probs': probs,
            'cls_embedding': cls_output
        }
        
        # Auxiliary reconstructions if enabled
        if self.use_auxiliary_loss:
            output['reconstructions'] = {
                'eeg': self.eeg_decoder(eeg_encoded),
                'eyetracking': self.eyetracking_decoder(eyetracking_encoded),
                'behavioral': self.behavioral_decoder(behavioral_encoded)
            }
        
        return output
    
    def get_attention_weights(
        self,
        eeg: torch.Tensor,
        eyetracking: torch.Tensor,
        behavioral: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability analysis."""
        # This requires modifying the forward pass to store attention weights
        # Implementation depends on PyTorch version and specific requirements
        raise NotImplementedError("Attention weight extraction requires custom hooks")


def create_neuroformer(config: dict) -> NeuroFormer:
    """Factory function to create NeuroFormer model from config."""
    return NeuroFormer(
        eeg_input_dim=config.get('eeg_input_dim', 128),
        eyetracking_input_dim=config.get('eyetracking_input_dim', 96),
        behavioral_input_dim=config.get('behavioral_input_dim', 64),
        d_model=config.get('d_model', 256),
        n_encoder_layers=config.get('n_encoder_layers', 4),
        n_fusion_layers=config.get('n_fusion_layers', 6),
        n_heads=config.get('n_heads', 8),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.3),
        n_classes=config.get('n_classes', 2),
        use_auxiliary_loss=config.get('use_auxiliary_loss', False)
    )


if __name__ == "__main__":
    # Test model instantiation
    model = NeuroFormer()
    
    # Create dummy inputs
    batch_size = 4
    seq_len = 300
    
    eeg = torch.randn(batch_size, seq_len, 128)
    eyetracking = torch.randn(batch_size, seq_len, 96)
    behavioral = torch.randn(batch_size, seq_len, 64)
    
    # Forward pass
    output = model(eeg, eyetracking, behavioral)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
