"""
Attention modules for time series forecasting.

Implements:
- CrossAttentionModule: Cross-attention mechanism
- FiLMModule: Feature-wise Linear Modulation
- MultiHeadAttention: Standard multi-head self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import logging
from typing import Optional


def safe_attention_softmax(scores, owner):
    """Return zero attention for empty rows without NaNs in forward/backward."""
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    if not getattr(owner, "_logged_empty_attention", False) and empty.any():
        owner._logged_empty_attention = True
        message = f"{type(owner).__name__} received all-masked attention rows; returning zero context"
        logger = getattr(owner, "attention_warning_logger", None)
        if logger is not None:
            logger(message)
        else:
            logging.getLogger(__name__).warning(message)
    safe = scores.masked_fill(empty, 0.0)
    return torch.softmax(safe, dim=-1).masked_fill(empty, 0.0)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.

    Implements the attention mechanism from "Attention Is All You Need".
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_k: Key/query dimension per head (default: d_model // num_heads)
            d_v: Value dimension per head (default: d_model // num_heads)
            dropout: Dropout rate
        """
        super().__init__()

        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads

        assert d_k * num_heads == d_model, "d_k * num_heads must equal d_model"
        assert d_v * num_heads == d_model, "d_v * num_heads must equal d_model"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * d_v, bias=False)

        # Output projection
        self.W_o = nn.Linear(num_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            query: Query tensor
                Shape: (batch_size, seq_len_q, d_model)
            key: Key tensor
                Shape: (batch_size, seq_len_k, d_model)
            value: Value tensor
                Shape: (batch_size, seq_len_v, d_model)
            mask: Optional attention mask
                Shape: (batch_size, seq_len_q, seq_len_k)

        Returns:
            Output tensor
                Shape: (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (B, seq_len_q, num_heads * d_k)
        K = self.W_k(key)  # (B, seq_len_k, num_heads * d_k)
        V = self.W_v(value)  # (B, seq_len_v, num_heads * d_v)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, seq_len_q, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, seq_len_k, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        # (B, num_heads, seq_len_v, d_v)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (B, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = safe_attention_softmax(scores, self)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        # (B, num_heads, seq_len_q, d_v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        # (B, seq_len_q, num_heads * d_v)

        # Output projection
        output = self.W_o(context)  # (B, seq_len_q, d_model)

        return output


class CrossAttentionModule(nn.Module):
    """
    Cross-attention module for conditioning on categorical features.

    This module allows the model to attend over categorical feature
    combinations when processing the time series.
    """

    def __init__(
        self,
        d_query: int,
        d_key: int,
        d_value: int,
        num_heads: int = 4,
        d_k: int = 32,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize cross-attention module.

        Args:
            d_query: Query dimension (typically time series features)
            d_key: Key dimension (categorical features)
            d_value: Value dimension (categorical features)
            num_heads: Number of attention heads
            d_k: Key/query dimension per head
            d_v: Value dimension per head (default: same as d_k)
            dropout: Dropout rate
        """
        super().__init__()

        if d_v is None:
            d_v = d_k

        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        # Projections
        self.W_q = nn.Linear(d_query, num_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_key, num_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_value, num_heads * d_v, bias=False)

        # Output projection
        self.W_o = nn.Linear(num_heads * d_v, d_value, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention.

        Args:
            query: Query tensor
                Shape: (batch_size, seq_len_q, d_query)
            key: Key tensor
                Shape: (batch_size, seq_len_k, d_key)
            value: Value tensor
                Shape: (batch_size, seq_len_v, d_value)

        Returns:
            Output tensor
                Shape: (batch_size, seq_len_q, d_value)
        """
        batch_size = query.size(0)

        # Degeneracy tripwire. With a SINGLE key, softmax over the key axis is
        # identically 1.0 for every query, so the attention weights are constant,
        # W_q and W_k receive exactly zero gradient at every step, and the module
        # collapses to the fixed linear map W_o(W_v(key)). Verified empirically:
        # W_q/W_k grad_absmax stays 0.0 across optimizer steps. The parameters
        # are still counted in get_num_params(), so an arm built this way reports
        # capacity it never trains. Warn once per module rather than silently
        # producing an "attention" arm that is not one.
        if key.size(1) == 1 and not getattr(self, "_warned_single_key", False):
            self._warned_single_key = True
            warnings.warn(
                f"{type(self).__name__} received a single key token: softmax over "
                "the key axis is identically 1.0, so W_q/W_k get zero gradient "
                "and this reduces to W_o(W_v(key)) -- a fixed linear map, not "
                "attention. Do not report an arm built this way as an attention "
                "mechanism. See encodings.CrossAttentionEncoding.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Linear projections
        Q = self.W_q(query)  # (B, seq_len_q, num_heads * d_k)
        K = self.W_k(key)  # (B, seq_len_k, num_heads * d_k)
        V = self.W_v(value)  # (B, seq_len_v, num_heads * d_v)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = safe_attention_softmax(scores, self)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_v
        )

        # Output projection
        output = self.W_o(context)

        return output


class FiLMModule(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) module.

    FiLM generates scale (gamma) and shift (beta) parameters conditioned
    on some input (e.g., categorical features) and applies them to modulate
    another input (e.g., time series features).

    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize FiLM module.

        Args:
            condition_dim: Dimension of conditioning input
            feature_dim: Dimension of features to modulate (can be set later)
            hidden_dim: Hidden dimension for FiMLP network
            num_layers: Number of layers in FiMLP network
            dropout: Dropout rate
        """
        super().__init__()

        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Build FiMLP network
        layers = []
        input_dim = condition_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # Final layer outputs hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))

        self.fimlp = nn.Sequential(*layers)

        # Create projection layers if feature_dim is known
        if feature_dim is not None:
            self._create_projection_layers(feature_dim)
        else:
            # Use placeholder Identity layers that will be replaced
            self.gamma_proj = nn.Identity()
            self.beta_proj = nn.Identity()
            self._projections_created = False

    def _create_projection_layers(self, feature_dim: int):
        """Create gamma and beta projection layers."""
        # Create new projection layers with zero initialization (identity FiLM)
        gamma_proj = nn.Linear(self.hidden_dim, feature_dim)
        beta_proj = nn.Linear(self.hidden_dim, feature_dim)

        # Initialize to identity: gamma≈0, beta≈0
        nn.init.zeros_(gamma_proj.weight)
        nn.init.zeros_(gamma_proj.bias)
        nn.init.zeros_(beta_proj.weight)
        nn.init.zeros_(beta_proj.bias)

        # Register as submodules properly (replaces placeholders)
        self.gamma_proj = gamma_proj
        self.beta_proj = beta_proj
        self._projections_created = True
        self.feature_dim = feature_dim

    def set_feature_dim(self, feature_dim: int):
        """
        Set feature dimension and create projection layers.

        Args:
            feature_dim: Dimension of features to modulate
        """
        if not getattr(self, '_projections_created', False) or self.feature_dim != feature_dim:
            self._create_projection_layers(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: Features to modulate
                Shape: (batch_size, seq_len, feature_dim)
            condition: Conditioning input
                Shape: (batch_size, seq_len, condition_dim)

        Returns:
            Modulated features
                Shape: (batch_size, seq_len, feature_dim)
        """
        feature_dim = features.size(-1)

        # Create projections if needed or if dimension changed
        if not getattr(self, '_projections_created', False) or self.feature_dim != feature_dim:
            self._create_projection_layers(feature_dim)

        batch_size, seq_len, _ = features.shape

        # Process condition through FiMLP
        condition_repr = self.fimlp(condition)  # (B, seq_len, hidden_dim)

        # Generate gamma and beta
        gamma = self.gamma_proj(condition_repr)  # (B, seq_len, feature_dim)
        beta = self.beta_proj(condition_repr)  # (B, seq_len, feature_dim)

        # Apply modulation: features * (1 + gamma) + beta
        # Using (1 + gamma) ensures identity initialization (gamma≈0 means minimal change)
        modulated = (1.0 + gamma) * features + beta

        return modulated


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds position information to input embeddings using sinusoidal functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability after adding positions
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor
                Shape: (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
                Shape: (batch_size, seq_len, d_model)
        """
        return self.dropout(x + self.pe[:, : x.size(1), :])
