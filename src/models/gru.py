"""
GRU models for time series forecasting.

Three concrete classes, all in this one file, registered as ``"gru"``:

================  ===========================  ================  ============================
Variant flag      Class                        Input shape       Paper-included file
================  ===========================  ================  ============================
onehot            GRUModel                     (B, L, F)         GRU/model.py::GRU
embeddings        GRUModel                     (B, L, F)         GRU/model.py::GRU
cross_attention   GRUComboCAOnly               (B, L, G, F)      GRU/model.py::GRUWithCrossAttentionOnly
film_attention    GRUComboCAFiLM               (B, L, G, F)      GRU/model.py::GRUWithCrossAttention
================  ===========================  ================  ============================

All architectures follow ``code/paper included/GRU/model.py`` (no imports
across the repo boundary). The two combo classes share a small backbone
helper to remove the duplication that existed in the paper code. The math
is unchanged EXCEPT for one correctness fix: the per-group flatten in
``_combo_temporal_head`` now permutes ``(B, L, G, H) -> (B, G, L, H)``
before reshaping to ``(B*G, L, H)``. The paper code reshaped directly,
which interleaved group and timestep and fed the recurrence temporally-
scrambled sequences. See the comment in ``_combo_temporal_head``.

A single registration ``ModelFactory.register("gru", make_gru)``
dispatches to the right class based on the ``variant`` argument.

Combo variants advertise ``requires_combo_loader = True`` so that
``scripts/train.py`` knows to ask ``TradeDataPipeline`` for the
``(B, L, G, F)`` combo-window dataloader.

The cross-attention / FiLM modules live in ``src/models/combo_attention.py``
and are shared between ``lstm.py`` and ``gru.py`` — paper had identical
copies in both files.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseModel, ModelFactory
from .combo_attention import (
    FiLMByGroup,
    IdentityCrossAttention,
    build_cross_attention,
    AxialComboCA,
    MatchedAxialCategoricalEncoder,
)


# ---------------------------------------------------------------------------
# Variant 1: normal 1D GRU
# Paper class: `GRU` in paper included/GRU/model.py
# Used by variants: onehot, embeddings
# ---------------------------------------------------------------------------

class GRUModel(BaseModel):
    """Per-series GRU (paper-equivalent base block).

        input_proj : Linear(input_dim -> hidden_size)
        gru        : GRU(hidden_size -> hidden_size, num_layers, dropout)
        output_proj: Linear(hidden_size -> 2)

    Prediction taken from ``gru_out[:, -1, :]`` — last timestep of the
    full output sequence. For a unidirectional single-batch GRU this
    equals ``h_n[-1]``, but staying with the paper's expression preserves
    state-dict key parity.
    """

    requires_combo_loader = False

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(self.input_dim, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.output_proj = nn.Linear(out_dim, 2)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,  # ignored; kept for uniform call signature
    ) -> torch.Tensor:
        x = self.encode_features(x_numeric, state_ids, comm_ids, flow_ids)
        x = self.input_proj(x)
        out, hidden = self.gru(x)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1) if self.bidirectional else out[:, -1, :]
        return self.output_proj(pooled)


# ---------------------------------------------------------------------------
# Shared backbone helpers (one of each kept in this file rather than a
# global module, because both helpers reference attribute name `gru` —
# keeping it local makes the file self-contained).
# ---------------------------------------------------------------------------

def _build_combo_backbone(
    module: nn.Module,
    features_per_group: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    """Attach in_proj, post_norm, per-group GRU, output_proj to ``module``."""
    module.hidden_size = hidden_size
    module.in_proj = nn.Linear(features_per_group, hidden_size)
    module.post_norm = nn.LayerNorm(hidden_size)
    module.bidirectional = False  # per-group recurrence over time is causal
    module.gru = nn.GRU(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
        batch_first=True,
    )
    module.output_proj = nn.Linear(hidden_size, 2)


def _combo_temporal_head(
    module: nn.Module,
    h_modulated: torch.Tensor,  # (B, L, G, H) — already attended (+ optionally FiLM'd)
    x_raw: torch.Tensor,        # (B, L, G, F) — for residual shortcut
) -> torch.Tensor:
    """Apply residual + per-group GRU + output projection. Returns (B, G, 2)."""
    B, L, G, _ = x_raw.shape
    shortcut = module.in_proj(x_raw)                                # (B, L, G, H)
    h = module.post_norm(h_modulated + shortcut)                    # (B, L, G, H)
    # Flatten groups into the batch so each row is ONE group's full time
    # sequence before the GRU. The group axis must be moved next to batch
    # first: a bare `reshape(B*G, L, H)` on a (B, L, G, H) tensor interleaves
    # group and timestep, feeding the GRU temporally-scrambled sequences.
    # (This corrects the paper-included code, which omitted the permute.)
    h = h.permute(0, 2, 1, 3).reshape(B * G, L, module.hidden_size)  # (B*G, L, H)
    out, _ = module.gru(h)
    # out[:, -1] is the last timestep per group; reshape back to (B, G, ·)
    # in the same b*G+g order the rows were created in.
    return module.output_proj(out[:, -1, :]).reshape(B, G, -1)


# ---------------------------------------------------------------------------
# Variant 2: cross-attention ONLY (no FiLM)
# Paper class: `GRUWithCrossAttentionOnly` in paper included/GRU/model.py
# Used by variant: cross_attention
# ---------------------------------------------------------------------------

class GRUComboCAOnly(BaseModel):
    """GRU over groups with cross-attention across groups per timestep.

    Pipeline (paper architecture; per-group flatten corrected):
        1. CrossAttention across groups, per timestep -> (B, L, G, H)
        2. Residual from in_proj(x) + LayerNorm
        3. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group GRU
        4. Linear(H -> 2) on last timestep

    Output: (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        num_latents: int = 0,
        num_combos: Optional[int] = None,
        features_per_group: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert num_combos is not None and features_per_group is not None, (
            "GRUComboCAOnly needs num_combos and features_per_group "
            "from the combo dataloader. The training script wires this "
            "automatically when model.requires_combo_loader is True."
        )

        # BaseModel created an encoder we don't use (combo input is
        # (B, L, G, F) raw; no per-series categorical encoding needed).
        # Replace with Identity so its parameters don't enter the optimizer.
        self.encoder = nn.Identity()

        self.num_groups = num_combos
        self.features_per_group = features_per_group

        self.cross_attention = build_cross_attention(
            features_per_group, hidden_size, attn_dropout, num_latents
        )
        if self.variant == "identity_cross_attention":
            self.cross_attention = IdentityCrossAttention(
                features_per_group, hidden_size, kwargs.get("combo_coords"),
                self.num_states, self.num_commodities, self.num_flows,
                attn_dropout=attn_dropout)
        _build_combo_backbone(self, features_per_group, hidden_size, num_layers, dropout)

    def forward(
        self,
        x_numeric: torch.Tensor,         # (B, L, G, F)
        state_ids: torch.Tensor = None,  # ignored for combo path
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_attn, _ = self.cross_attention(x_numeric, mask_g=group_mask)
        return _combo_temporal_head(self, h_attn, x_numeric)


# ---------------------------------------------------------------------------
# Variant 3: cross-attention + FiLM
# Paper class: `GRUWithCrossAttention` in paper included/GRU/model.py
# Used by variant: film_attention
# ---------------------------------------------------------------------------

class GRUComboCAFiLM(BaseModel):
    """GRU over groups with cross-attention + FiLM-by-group conditioning.

    Pipeline (paper architecture; per-group flatten corrected):
        1. CrossAttention across groups, per timestep -> (B, L, G, H)
        2. FiLMByGroup: per-(B,L,G) gamma/beta + sigmoid gate -> (B, L, G, H)
        3. Residual from in_proj(x) + LayerNorm
        4. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group GRU
        5. Linear(H -> 2) on last timestep

    Output: (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        num_latents: int = 0,
        film_hidden: int = 128,
        gate_temperature: float = 1.0,
        min_gate: float = 0.0,
        normalize_group_weights: bool = False,
        num_combos: Optional[int] = None,
        features_per_group: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert num_combos is not None and features_per_group is not None, (
            "GRUComboCAFiLM needs num_combos and features_per_group "
            "from the combo dataloader. The training script wires this "
            "automatically when model.requires_combo_loader is True."
        )

        self.encoder = nn.Identity()  # see GRUComboCAOnly for rationale

        self.num_groups = num_combos
        self.features_per_group = features_per_group

        self.cross_attention = build_cross_attention(
            features_per_group, hidden_size, attn_dropout, num_latents
        )
        self.film = FiLMByGroup(
            num_groups=num_combos,
            hidden_dim=hidden_size,
            film_hidden=film_hidden,
            gate_temperature=gate_temperature,
            min_gate=min_gate,
            normalize_group_weights=normalize_group_weights,
        )
        _build_combo_backbone(self, features_per_group, hidden_size, num_layers, dropout)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_attn, _ = self.cross_attention(x_numeric, mask_g=group_mask)
        h_film, _ = self.film(h_attn, mask_g=group_mask)
        return _combo_temporal_head(self, h_film, x_numeric)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

class GRUComboAxial(BaseModel):
    """GRU over groups with AXIAL cross-attention (S4ND-style 2d/3d/4d).

    Instead of one flat cross-attention over all combos, attend separately
    along each lattice axis (State / +Commodity / +Flow), mirroring S4ND's
    separable grid mixing. Then the same per-group GRU over time.

    Pipeline:
        1. AxialComboCA over the (S,C,Flow) lattice -> (B, L, G, H)
        2. Residual from in_proj(x) + LayerNorm
        3. Permute to (B,G,L,H), reshape to (B*G, L, H), per-group GRU
        4. Linear(H -> 2) on last timestep

    Output: (B, G, 2)
    """

    requires_combo_loader = True

    def _build_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
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
        ), (
            "GRUComboAxial needs num_combos, features_per_group, combo_coords "
            "and lattice_dims from the combo dataloader (wired by train.py)."
        )
        self.encoder = nn.Identity()
        self.num_groups = num_combos
        self.features_per_group = features_per_group
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
        _build_combo_backbone(self, encoded_features, hidden_size, num_layers, dropout)

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_encoded = self.group_encoder(x_numeric)
        h_attn = self.axial(x_encoded)
        return _combo_temporal_head(self, h_attn, x_encoded)


# Axial combo variants. Both name families run the S4ND matched-information
# protocol (see MatchedAxialCategoricalEncoder): identities that are NOT grid
# axes are re-injected as embeddings, exactly as s4nd does with its cfid
# encoder. matched_axial_* is kept as an alias so its config files resolve.
_AXIAL_DIMS = {
    "cross_attention_2d": 2, "cross_attention_3d": 3, "cross_attention_4d": 4,
    "matched_axial_2d": 2, "matched_axial_3d": 3, "matched_axial_4d": 4,
}


def make_gru(variant: str, **kwargs) -> BaseModel:
    """Pick the right GRU class for the requested variant."""
    if variant in ("onehot", "embeddings"):
        return GRUModel(variant=variant, **kwargs)
    if variant in ("cross_attention", "identity_cross_attention"):
        return GRUComboCAOnly(variant=variant, **kwargs)
    if variant == "film_attention":
        return GRUComboCAFiLM(variant=variant, **kwargs)
    if variant in _AXIAL_DIMS:
        return GRUComboAxial(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)
    raise ValueError(
        f"Unknown GRU variant: {variant!r}. Expected one of: onehot, embeddings, "
        "cross_attention, film_attention, cross_attention_{2d,3d,4d}, "
        "matched_axial_{2d,3d,4d}."
    )


ModelFactory.register("gru", make_gru)
