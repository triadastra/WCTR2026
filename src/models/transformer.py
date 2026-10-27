"""
Transformer model for time series forecasting.

Implements a standard transformer encoder with configurable encoding strategy.

Uses a pre-norm Transformer encoder and a next-step regression head.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseModel, ModelFactory
from .attention import PositionalEncoding
from .combo_attention import AxialComboCA, MatchedAxialCategoricalEncoder


def _prediction_head(hidden_size: int, dropout: float, kind: str) -> nn.Module:
    """``mlp`` is the published Transformer head (hidden -> hidden/2 -> 2);
    ``linear`` is the Linear(hidden, 2) readout the LSTM/GRU/S4/Mamba arms use."""
    if kind == "linear":
        return nn.Linear(hidden_size, 2)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )
    raise ValueError("prediction_head must be linear or mlp")


class TransformerModel(BaseModel):
    """
    Transformer model with configurable encoding strategy.

    Architecture:
    1. Encode categorical features
    2. Add positional encoding
    3. Process through transformer encoder layers
    4. Output layer for prediction
    """
    # Two-sided over the input window? unmasked nn.TransformerEncoder: every step attends to the whole window.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
        activation: str = "gelu",
        max_seq_len: int = 1000,
        positional_encoding: str = "sinusoidal",
        prediction_head: str = "mlp",
        **kwargs
    ) -> None:
        """
        Build transformer model.

        Args:
            hidden_size: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            d_ff: Feed-forward dimension
            activation: Activation function
            max_seq_len: Maximum sequence length
            positional_encoding: Type of positional encoding
            **kwargs: Additional arguments
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Project input to model dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_size)

        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
            self.use_learned_pos = False
        elif positional_encoding == "learned":
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_size)
            self.use_learned_pos = True
        else:
            raise ValueError("positional_encoding must be sinusoidal or learned")

        # Transformer encoder. norm_first=True (pre-norm) is far more
        # stable than the default post-norm without learning-rate warmup;
        # post-norm + lr=1e-4 still diverged to NaN on this dataset
        # within epoch 0. Pre-norm trains cleanly with the same lr.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layer
        self.output_layer = _prediction_head(hidden_size, dropout, prediction_head)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_numeric: (batch_size, seq_len, num_features)
            state_ids: (batch_size,)
            comm_ids: (batch_size,)
            flow_ids: (batch_size,)

        Returns:
            predictions: (batch_size, 2)
        """
        # Encode features
        x_encoded = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)

        # Project to model dimension
        x = self.input_projection(x_encoded)  # (B, seq_len, hidden_size)

        # Add positional encoding
        if self.use_learned_pos:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.pos_encoding(positions)
        else:
            x = self.pos_encoding(x)

        # Every token is an observed month before the single forecast target.
        # Bidirectional attention within this history window is intentional;
        # the target month itself is excluded by TradeDataset's windowing.
        x = self.transformer_encoder(x)  # (B, seq_len, hidden_size)

        # Use last time step for prediction
        x_last = x[:, -1, :]  # (B, hidden_size)

        # Output layer
        predictions = self.output_layer(x_last)  # (B, 2)

        return predictions


# Register model
class TransformerComboAxial(BaseModel):
    """Transformer with S4ND-style AXIAL cross-attention across the combo
    lattice, then a per-group Transformer encoder over time.

    Pipeline:
        1. AxialComboCA over the (S,C,Flow) lattice -> (B, L, G, H)
        2. Residual from in_proj(x_raw) + LayerNorm
        3. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group Transformer
        4. Prediction head on the last timestep -> (B, G, 2)
    """

    requires_combo_loader = True
    # Two-sided over the input window? unmasked self-attention over time within each group.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
        activation: str = "gelu",
        max_seq_len: int = 1000,
        attn_dropout: float = 0.0,
        dims: int = 4,
        num_combos: Optional[int] = None,
        features_per_group: Optional[int] = None,
        combo_coords=None,
        lattice_dims=None,
        matched_encoding: bool = True,
        comm_embed_dim: int = 32,
        flow_embed_dim: int = 2,
        prediction_head: str = "mlp",
        **kwargs,
    ) -> None:
        assert (
            num_combos is not None and features_per_group is not None
            and combo_coords is not None and lattice_dims is not None
        ), "TransformerComboAxial needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.hidden_size = hidden_size
        self.num_groups = num_combos
        self.group_encoder = (
            MatchedAxialCategoricalEncoder(
                features_per_group=features_per_group,
                combo_coords=combo_coords,
                dims=dims,
                num_commodities=self.num_commodities,
                num_flows=self.num_flows,
                comm_embed_dim=comm_embed_dim,
                flow_embed_dim=flow_embed_dim,
            )
            if matched_encoding else nn.Identity()
        )
        encoded_features = (
            self.group_encoder.output_dim if matched_encoding else features_per_group
        )
        self.axial = AxialComboCA(
            encoded_features, hidden_size, lattice_dims, combo_coords,
            dims=dims, attn_dropout=attn_dropout,
        )
        self.in_proj_raw = nn.Linear(encoded_features, hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, activation=activation, batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = _prediction_head(hidden_size, dropout, prediction_head)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_encoded = self.group_encoder(x_numeric)
        h = self.axial(x_encoded)                              # (B, L, G, H)
        h = self.post_norm(h + self.in_proj_raw(x_encoded))   # (B, L, G, H)
        B, L, G, H = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * G, L, H)        # per-group sequence
        h = self.pos_encoding(h)
        h = self.transformer_encoder(h)                       # (B*G, L, H)
        return self.output_layer(h[:, -1, :]).reshape(B, G, -1)


# Axial combo variants. Both name families run the S4ND matched-information
# protocol (see MatchedAxialCategoricalEncoder): identities that are NOT grid
# axes are re-injected as embeddings, exactly as s4nd does with its cfid
# encoder. matched_axial_* is kept as an alias so its config files resolve.
_AXIAL_DIMS = {
    "cross_attention_2d": 2, "cross_attention_3d": 3, "cross_attention_4d": 4,
    "matched_axial_2d": 2, "matched_axial_3d": 3, "matched_axial_4d": 4,
}


def make_transformer(variant: str, **kwargs) -> BaseModel:
    """Flat variants -> TransformerModel; axial variants -> combo model."""
    if variant in _AXIAL_DIMS:
        return TransformerComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    return TransformerModel(variant=variant, **kwargs)


ModelFactory.register("transformer", make_transformer)
# Same architecture under its own name so the linear-head reruns never share a
# run directory, results row or SHAP table with the published MLP-head arms.
# config/models/transformer_linhead.yaml sets prediction_head: linear.
ModelFactory.register("transformer_linhead", make_transformer)
