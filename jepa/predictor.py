"""
Predictor - maps context embeddings to target predictions.

Keep it lightweight! If predictor is too powerful it'll just memorize instead of learning.
"""

import torch
import torch.nn as nn
from typing import Optional


class Predictor(nn.Module):
    """
    Lightweight predictor - maps context to target predictions.
    
    Keep it simpler than encoders (2-4 transformer layers, or 2-3 layer MLP, or cross-attention).
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 768,  # Keep FFN same as hidden_dim (much lighter)
        dropout: float = 0.1,
        predictor_type: str = "transformer"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.predictor_type = predictor_type
        
        if predictor_type == "transformer":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.predictor = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers
            )
            self.target_queries = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
        elif predictor_type == "mlp":
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.extend([
                        nn.Linear(hidden_dim, intermediate_size),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ])
                elif i == num_layers - 1:
                    layers.append(nn.Linear(intermediate_size, hidden_dim))
                else:
                    layers.extend([
                        nn.Linear(intermediate_size, intermediate_size),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ])
            self.predictor = nn.Sequential(*layers)
            
            self.target_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(8)  # Up to 8 targets
            ])
            
        elif predictor_type == "cross_attention":
            self.target_queries = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=intermediate_size,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(num_layers)
            ])
            
        else:
            raise ValueError(
                f"Unknown predictor_type: {predictor_type}. "
                f"Must be one of: 'transformer', 'mlp', 'cross_attention'"
            )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Init weights."""
        if hasattr(self, 'target_queries'):
            nn.init.normal_(self.target_queries, mean=0.0, std=0.02)
    
    def forward(
        self,
        context_embeddings: torch.Tensor,
        num_targets: int = 1,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict target embeddings. Returns [batch, num_targets, hidden_dim]."""
        if context_embeddings.dim() != 3:
            raise ValueError(
                f"context_embeddings must be 3D [batch, seq, dim], "
                f"got shape {context_embeddings.shape}"
            )
        
        batch_size = context_embeddings.size(0)
        
        if self.predictor_type == "transformer":
            queries = self.target_queries.expand(batch_size, num_targets, -1)
            
            memory_key_padding_mask = None
            if attention_mask is not None:
                memory_key_padding_mask = (attention_mask == 0)
            
            predictions = self.predictor(
                tgt=queries,
                memory=context_embeddings,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
        elif self.predictor_type == "mlp":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    context_embeddings.size()
                ).float()
                sum_embeddings = torch.sum(context_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_context = sum_embeddings / sum_mask
            else:
                pooled_context = context_embeddings.mean(dim=1)
            
            base_prediction = self.predictor(pooled_context)
            
            predictions_list = []
            for i in range(num_targets):
                proj_idx = min(i, len(self.target_projections) - 1)
                target_pred = self.target_projections[proj_idx](base_prediction)
                predictions_list.append(target_pred)
            
            predictions = torch.stack(predictions_list, dim=1)
            
        elif self.predictor_type == "cross_attention":
            queries = self.target_queries.expand(batch_size, num_targets, -1)
            
            memory_key_padding_mask = None
            if attention_mask is not None:
                memory_key_padding_mask = (attention_mask == 0)
            
            for layer in self.layers:
                queries = layer(
                    tgt=queries,
                    memory=context_embeddings,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            
            predictions = queries
        
        return predictions
    
    def predict_single(
        self,
        context_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict single target. Returns [batch, hidden_dim]."""
        predictions = self.forward(context_embeddings, num_targets=1, attention_mask=attention_mask)
        return predictions.squeeze(1)

