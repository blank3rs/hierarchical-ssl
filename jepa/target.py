"""
Target encoder - processes masked text. Not directly trained, updated via EMA from context encoder.
Provides stable targets for predictor.
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class TargetEncoder(nn.Module):
    """
    Target encoder updated via EMA.
    
    Same architecture as ContextEncoder, initialized as copy, frozen params.
    Updated via EMA after each training step.
    
    EMA provides stable targets and prevents collapse (like BYOL/MoCo).
    """
    
    def __init__(
        self,
        context_encoder: nn.Module,
        ema_momentum: float = 0.996
    ):
        """Init as copy of context encoder. EMA momentum typically 0.996 (range 0.9-0.999)."""
        super().__init__()
        
        if not 0.0 < ema_momentum < 1.0:
            raise ValueError(
                f"ema_momentum must be in (0, 1), got {ema_momentum}"
            )
        
        self.ema_momentum = ema_momentum
        
        self.encoder = copy.deepcopy(context_encoder)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.hidden_dim = self.encoder.hidden_dim
        self.vocab_size = self.encoder.vocab_size
        self.max_position_embeddings = self.encoder.max_position_embeddings
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass without gradients. Returns [batch, seq_len, hidden_dim]."""
        self.encoder.eval()
        return self.encoder(input_ids, attention_mask)
    
    @torch.no_grad()
    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """Get pooled output. Returns [batch, hidden_dim]."""
        self.encoder.eval()
        return self.encoder.get_pooled_output(
            input_ids, 
            attention_mask, 
            pooling_strategy
        )
    
    @torch.no_grad()
    def update_from_context(self, context_encoder: nn.Module) -> None:
        """
        Update target encoder via EMA: θ_target = m * θ_target + (1 - m) * θ_context
        Call after each training step.
        """
        if context_encoder.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Context encoder hidden_dim {context_encoder.hidden_dim} "
                f"doesn't match target encoder {self.hidden_dim}"
            )
        
        for param_target, param_context in zip(
            self.encoder.parameters(),
            context_encoder.parameters()
        ):
            param_target.data.mul_(self.ema_momentum).add_(
                param_context.data, 
                alpha=1.0 - self.ema_momentum
            )
    
    @torch.no_grad()
    def copy_from_context(self, context_encoder: nn.Module) -> None:
        """Direct copy (not EMA) - useful for resetting."""
        for param_target, param_context in zip(
            self.encoder.parameters(),
            context_encoder.parameters()
        ):
            param_target.data.copy_(param_context.data)
    
    def set_momentum(self, new_momentum: float) -> None:
        """Update EMA momentum (useful for scheduling)."""
        if not 0.0 < new_momentum < 1.0:
            raise ValueError(
                f"new_momentum must be in (0, 1), got {new_momentum}"
            )
        self.ema_momentum = new_momentum
    
    def get_momentum(self) -> float:
        """Get current EMA momentum."""
        return self.ema_momentum
    
    def train(self, mode: bool = True):
        """Always stays in eval mode (not trained)."""
        super().train(False)
        self.encoder.eval()
        return self
    
    def eval(self):
        """Already in eval mode."""
        super().eval()
        self.encoder.eval()
        return self

