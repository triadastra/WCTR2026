"""Mamba-2 model for per-series trade forecasting.

Runs Mamba-2 straight from the vendored source in ``external/mamba/``
(cloned state-spaces/mamba), NOT a pip package -- mirroring how the S4
family uses ``external/s4/``. The SSD scan is Triton (compiled at runtime,
no build). Two compiled CUDA kernels are deliberately bypassed so there is
zero build dependency:

  * ``selective_scan_cuda`` -- Mamba-1's kernel, never used by Mamba-2.
  * ``causal_conv1d``       -- optional; Mamba-2 falls back to a plain
                              PyTorch causal conv (``use_mem_eff_path=False``).

Replaces the LSTM-stub "mamba" formerly registered by ssm_models.py.
"""
from typing import Optional

import torch
import torch.nn as nn

from .vendor_imports import load_mamba2
Mamba2 = load_mamba2()

from .base import BaseModel, ModelFactory
from .combo_attention import AxialComboCA, MatchedAxialCategoricalEncoder


class MambaModel(BaseModel):
    """Stacked pre-norm Mamba-2 blocks over per-series windows; prediction is
    read from the last timestep, mirroring the LSTM/GRU heads."""

    requires_combo_loader = False
    # Two-sided over the input window? the Mamba-2 selective scan is causal.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = False

    def _build_model(self, d_model: int = 128, n_layers: int = 4,
                     d_state: int = 64, d_conv: int = 4, expand: int = 2,
                     headdim: int = 32, dropout: float = 0.1, **kwargs) -> None:
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim, use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids, group_mask=None):
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)
        x = self.input_proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            x = x + self.dropout(blk(norm(x)))   # pre-norm residual
        return self.output_proj(x[:, -1, :])


class MambaComboAxial(BaseModel):
    """Mamba-2 with S4ND-style AXIAL cross-attention across the combo lattice,
    then per-group Mamba-2 blocks over time.

    The *bolted-on* counterpart to ``mamba_nd``: same Mamba-2 sequence model,
    but the grid arrives through axial cross-attention rather than the ND scan.
    Because the cross-attention covers every axis inside one module and is
    direction-free, this arm has none of the scan's depth/direction budget
    problems -- it sits in the ACA family at the same n_layers=4 as the rest.

    Pipeline mirrors S4ComboAxial / GPTComboAxial; only the temporal block
    differs.
    """

    requires_combo_loader = True
    # Two-sided over the input window? the Mamba-2 scan over time within each group is causal.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = False

    def _build_model(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        dims: int = 4,
        num_combos: Optional[int] = None,
        features_per_group: Optional[int] = None,
        combo_coords=None,
        lattice_dims=None,
        matched_encoding: bool = True,
        comm_embed_dim: int = 32,
        flow_embed_dim: int = 2,
        **kwargs,
    ) -> None:
        assert (
            num_combos is not None and features_per_group is not None
            and combo_coords is not None and lattice_dims is not None
        ), "MambaComboAxial needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.d_model = d_model
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
            encoded_features, d_model, lattice_dims, combo_coords,
            dims=dims, attn_dropout=attn_dropout,
        )
        self.in_proj_raw = nn.Linear(encoded_features, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim, use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_encoded = self.group_encoder(x_numeric)
        h = self.axial(x_encoded)                            # (B, L, G, D)
        h = self.pre_norm(h + self.in_proj_raw(x_encoded))   # (B, L, G, D)
        B, L, G, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * G, L, D)       # per-group sequence
        for blk, norm in zip(self.blocks, self.norms):
            h = h + self.dropout(blk(norm(h)))               # pre-norm residual
        return self.output_proj(h[:, -1, :]).reshape(B, G, -1)


# Axial combo variants; see the note in gru.py. Both name families run the
# S4ND matched-information protocol.
_AXIAL_DIMS = {
    "cross_attention_2d": 2, "cross_attention_3d": 3, "cross_attention_4d": 4,
    "matched_axial_2d": 2, "matched_axial_3d": 3, "matched_axial_4d": 4,
}


def make_mamba(variant: str, **kwargs) -> BaseModel:
    if variant in _AXIAL_DIMS:
        return MambaComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    return MambaModel(variant=variant, **kwargs)


ModelFactory.register("mamba", make_mamba)
