"""Forecasting adapters around the authors' modular S4 SequenceModel.

Exp 1, Exp 3 and axial S4 use the same modular backbone as Exp 2's external
runner. Category/grid encoding and forecasting heads are benchmark adapters.
See external/s4/UPSTREAM.md for the audited local patches.
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel, ModelFactory
from .combo_attention import AxialComboCA, MatchedAxialCategoricalEncoder
from .s4_modular import IMPLEMENTATION, create_modular_s4


class S4Model(BaseModel):
    """Stacked pre-norm real-S4 (``S4Block``) layers over the input window; the
    prediction is read from the last timestep, mirroring the LSTM/GRU/Mamba heads."""

    requires_combo_loader = False
    # The canonical modular recipe uses a causal kernel.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = False

    def _build_model(self, d_model: int = 128, n_layers: int = 4,
                     d_state: int = 64, dropout: float = 0.1,
                     n_ssm: int = 1, dt_transform: str = "softplus",
                     tie_dropout: bool = True, implementation: str = IMPLEMENTATION,
                     bidirectional: bool = False,
                     **kwargs) -> None:
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.backbone = create_modular_s4(
            d_model=d_model, n_layers=n_layers, d_state=d_state, dropout=dropout,
            n_ssm=n_ssm, dt_transform=dt_transform, tie_dropout=tie_dropout,
            implementation=implementation, bidirectional=bidirectional)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x_numeric, state_ids, comm_ids, flow_ids, group_mask=None):
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)  # (B, L, input_dim)
        x = self.input_proj(x)                                              # (B, L, d_model)
        x, _ = self.backbone(x)
        return self.output_proj(x[:, -1, :])                                # (B, 2)


class S4ComboAxial(BaseModel):
    """Real S4 with S4ND-style AXIAL cross-attention across the combo lattice,
    then per-group S4 blocks over time.

    This is the *bolted-on* counterpart to ``s4nd``: same S4 sequence model,
    but the grid is injected by axial cross-attention instead of S4ND's native
    separable per-axis convolution. The temporal backbone matches modular 1-D
    S4; S4ND still differs in axis kernels and bidirectionality.

    Pipeline:
        1. Matched identity encoder (only non-structural ids) -> (B, L, G, F')
        2. AxialComboCA over the (S,C,Flow) lattice          -> (B, L, G, D)
        3. Residual from in_proj_raw(x) + LayerNorm
        4. Permute to (B*G, L, D), per-group S4 blocks over time
        5. Head on the last timestep                          -> (B, G, 2)
    """

    requires_combo_loader = True
    # Two-sided over the input window? causal S4 kernel over time within each group.
    # Recorded into metrics.json by BaseModel.architecture_provenance().
    bidirectional = False

    def _build_model(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
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
        n_ssm: int = 1,
        dt_transform: str = "softplus",
        tie_dropout: bool = True,
        implementation: str = IMPLEMENTATION,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        assert (
            num_combos is not None and features_per_group is not None
            and combo_coords is not None and lattice_dims is not None
        ), "S4ComboAxial needs combo metadata from the combo dataloader."
        self.encoder = nn.Identity()
        self.d_model = d_model
        self.bidirectional = bidirectional
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
        self.backbone = create_modular_s4(
            d_model=d_model, n_layers=n_layers, d_state=d_state, dropout=dropout,
            n_ssm=n_ssm, dt_transform=dt_transform, tie_dropout=tie_dropout,
            implementation=implementation, bidirectional=bidirectional)
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
        h, _ = self.backbone(h)
        return self.output_proj(h[:, -1, :]).reshape(B, G, -1)


# Axial combo variants; see the note in gru.py. Both name families run the
# S4ND matched-information protocol.
_AXIAL_DIMS = {
    "cross_attention_2d": 2, "cross_attention_3d": 3, "cross_attention_4d": 4,
    "matched_axial_2d": 2, "matched_axial_3d": 3, "matched_axial_4d": 4,
}


def make_s4(variant: str, **kwargs) -> BaseModel:
    if variant in ("onehot", "embeddings"):
        return S4Model(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return S4ComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    raise ValueError(
        f"Unsupported S4 variant: {variant!r}. S4 supports: onehot, embeddings, "
        "cross_attention_{2d,3d,4d}."
    )


ModelFactory.register("s4", make_s4)


def make_s4_bidir(variant: str, **kwargs) -> BaseModel:
    if kwargs.pop("bidirectional", True) is not True:
        raise ValueError("s4_bidir requires bidirectional=True")
    return make_s4(variant, bidirectional=True, **kwargs)


ModelFactory.register("s4_bidir", make_s4_bidir)
