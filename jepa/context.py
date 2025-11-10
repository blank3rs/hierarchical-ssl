"""
Context encoder - processes visible text, gets trained.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ContextEncoder(nn.Module):
    """
    Encodes visible text. Sees full sequence with [MASK] tokens.
    Creates embeddings that can predict masked content.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,  # BERT vocab size
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # MPS compatibility
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """BERT-style weight init."""
        self.token_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass. Returns [batch, seq_len, hidden_dim] embeddings."""
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch_size, seq_len], got shape {input_ids.shape}"
            )
        
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_position_embeddings}"
            )
        
        position_ids = torch.arange(
            seq_len, 
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # PyTorch transformer: True = ignore, False = attend (inverted)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
            if not padding_mask.any():
                padding_mask = None
        else:
            padding_mask = None
        
        output = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )
        
        return output
    
    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """Get single vector for sequence. Returns [batch, hidden_dim]."""
        embeddings = self.forward(input_ids, attention_mask)
        
        if pooling_strategy == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return embeddings.mean(dim=1)
        
        elif pooling_strategy == "cls":
            return embeddings[:, 0, :]
        
        elif pooling_strategy == "max":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = embeddings.clone()
                embeddings[mask_expanded == 0] = -1e9
            return torch.max(embeddings, dim=1)[0]
        
        else:
            raise ValueError(
                f"Unknown pooling_strategy: {pooling_strategy}. "
                f"Must be one of: 'mean', 'cls', 'max'"
            )
    
    def get_config(self) -> Dict[str, Any]:
        """Get config dict for saving."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'max_position_embeddings': self.max_position_embeddings,
        }

