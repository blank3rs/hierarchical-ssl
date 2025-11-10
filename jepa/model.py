"""
Complete JEPA model - combines context encoder, target encoder, and predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from jepa.context import ContextEncoder
from jepa.target import TargetEncoder
from jepa.predictor import Predictor


class JEPA(nn.Module):
    """
    JEPA for text.
    
    Context encoder: processes visible text (trainable)
    Target encoder: processes masked text (EMA-updated, frozen)
    Predictor: predicts target embeddings from context (trainable)
    
    Training: mask spans -> encode context -> encode targets (no grad) -> 
    predict targets -> compute loss -> backprop context+predictor -> EMA update target
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        encoder_layers: int = 12,  # BERT-Base: 12 layers
        encoder_heads: int = 12,
        predictor_layers: int = 2,  # Keep predictor ~6x smaller than encoder
        predictor_heads: int = 8,
        ema_momentum: float = 0.996,
        predictor_type: str = "transformer",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.context_encoder = ContextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=dropout
        )
        
        self.target_encoder = TargetEncoder(
            context_encoder=self.context_encoder,
            ema_momentum=ema_momentum
        )
        
        self.predictor = Predictor(
            hidden_dim=hidden_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            predictor_type=predictor_type,
            dropout=dropout
        )
        
        self.hidden_dim = hidden_dim
        self.ema_momentum = ema_momentum
    
    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        num_targets: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns dict with predicted, target, context_embeddings, loss."""
        if context_ids.dim() != 2:
            raise ValueError(
                f"context_ids must be 2D [batch, seq_len], got {context_ids.shape}"
            )
        
        if num_targets is None:
            if target_ids.dim() == 3:
                num_targets = target_ids.size(1)
            else:
                num_targets = 1
        
        context_embeddings = self.context_encoder(context_ids, context_mask)
        
        predicted_embeddings = self.predictor(
            context_embeddings,
            num_targets=num_targets,
            attention_mask=context_mask
        )
        
        target_embeddings = self._encode_targets(target_ids, target_mask, num_targets)
        
        loss = self.compute_loss(predicted_embeddings, target_embeddings)
        
        return {
            'predicted': predicted_embeddings,
            'target': target_embeddings,
            'context_embeddings': context_embeddings,
            'loss': loss
        }
    
    @torch.no_grad()
    def _encode_targets(
        self,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        num_targets: int
    ) -> torch.Tensor:
        """Encode targets with target encoder (no grad). Returns [batch, num_targets, hidden_dim]."""
        if target_ids.dim() == 3:
            batch_size, num_tgt, target_len = target_ids.shape
            
            target_ids_flat = target_ids.view(batch_size * num_tgt, target_len)
            target_mask_flat = target_mask.view(batch_size * num_tgt, target_len)
            
            target_embeddings_flat = self.target_encoder.get_pooled_output(
                target_ids_flat,
                target_mask_flat,
                pooling_strategy="mean"
            )
            
            target_embeddings = target_embeddings_flat.view(
                batch_size, num_tgt, self.hidden_dim
            )
            
        else:
            target_embeddings = self.target_encoder.get_pooled_output(
                target_ids,
                target_mask,
                pooling_strategy="mean"
            )
            
            target_embeddings = target_embeddings.unsqueeze(1)
        
        return target_embeddings
    
    def compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01
    ) -> torch.Tensor:
        """
        VICReg-style loss: L2 + variance + covariance regularization.
        Prevents collapse by keeping variance high and decorrelating dimensions.
        """
        predicted_norm = F.normalize(predicted, p=2, dim=-1)
        target_norm = F.normalize(target.detach(), p=2, dim=-1)
        
        repr_loss = F.mse_loss(predicted_norm, target_norm, reduction=reduction)
        
        batch_size, num_targets, hidden_dim = predicted.shape
        pred_flat = predicted.reshape(batch_size * num_targets, hidden_dim)
        target_flat = target.reshape(batch_size * num_targets, hidden_dim)
        
        pred_var = torch.var(pred_flat, dim=0) + 1e-4
        target_var = torch.var(target_flat, dim=0) + 1e-4
        
        var_loss = torch.mean(F.relu(1.0 - pred_var)) + torch.mean(F.relu(1.0 - target_var))
        
        pred_centered = pred_flat - pred_flat.mean(dim=0)
        target_centered = target_flat - target_flat.mean(dim=0)
        
        pred_cov = (pred_centered.T @ pred_centered) / (batch_size * num_targets - 1)
        target_cov = (target_centered.T @ target_centered) / (batch_size * num_targets - 1)
        
        cov_loss = self._off_diagonal(pred_cov).pow(2).sum() / hidden_dim
        cov_loss += self._off_diagonal(target_cov).pow(2).sum() / hidden_dim
        
        total_loss = repr_loss + variance_weight * var_loss + covariance_weight * cov_loss
        
        self._last_loss_components = {
            'repr_loss': repr_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss
    
    @staticmethod
    def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        """Get off-diagonal elements of square matrix."""
        n = matrix.shape[0]
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update target encoder via EMA. Call after optimizer.step()."""
        self.target_encoder.update_from_context(self.context_encoder)
    
    def get_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """Get representations for inference. Returns [batch, hidden_dim]."""
        return self.context_encoder.get_pooled_output(
            input_ids,
            attention_mask,
            pooling_strategy
        )
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'context_encoder': self.context_encoder.state_dict(),
            'target_encoder': self.target_encoder.encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'ema_momentum': self.ema_momentum,
            }
        }, save_path)
    
    def load_pretrained(self, load_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location='cpu')
        
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.target_encoder.encoder.load_state_dict(checkpoint['target_encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        
        if 'config' in checkpoint:
            self.ema_momentum = checkpoint['config'].get('ema_momentum', 0.996)

