"""Mamba-ND grid model for the multidimensional trade benchmark.

Implements the Mamba-ND method (Li et al., ECCV 2024; vendored under
``external/Mamba-ND``): scan the data with a 1-D selective SSM separately
along each grid axis, alternating the axis (and direction) layer by layer.
This is the Mamba analogue of S4ND's separable per-axis convolutions.

Two faithful adaptations vs. the upstream repo (see external/Mamba-ND/
VENDORING_NOTE.md):
  * Mixer = **Mamba-2 (SSD)** from ``external/mamba`` rather than Mamba-1.
    The box has only the Triton SSD path; Mamba-1's ``selective_scan_cuda``
    kernel is unavailable. Mamba-ND's scan scheme is mixer-agnostic, so this
    is a drop-in (and uses the same block as the benchmark-winning Mamba-2).
  * The per-axis scan (rearrange axis -> sequence, optional flip, mix,
    rearrange back) follows ``external/Mamba-ND/.../Block.forward`` but drops
    its mmcv ``build_dropout``/FFN wrappers so no mmcv dependency is needed.

Each layer's scan axis and direction come from ``scan_schedule.py`` (default
``"cyclic"``). The original schedule phase-locked the two -- it advanced the
axis every layer while flipping direction every layer, where upstream advances
the axis every TWO layers (``z = i // 2``) -- pinning every axis to a single
direction for the 2d and 4d variants. Pass ``scan_schedule="legacy"`` to
reproduce runs recorded before that fix.

Grid axes over the dense (State, Commodity, Flow) lattice + time:
    cross_attention_2d -> scan Time + State            (cf. s4nd_2d)
    cross_attention_3d -> scan Time + State + Commodity (cf. s4nd_3d)
    cross_attention_4d -> scan Time + State + Commodity + Flow (cf. s4nd_4d)

Combos are scattered onto the dense lattice (padded cells masked/zeroed),
scanned, then gathered back; prediction is the last-timestep readout per
kept combo, like the other combo models. GPU/Triton only.
"""


import torch
import torch.nn as nn
from typing import Optional

from .vendor_imports import load_mamba2
Mamba2 = load_mamba2()

from .base import BaseModel, ModelFactory  # noqa: E402
from .combo_attention import MatchedAxialCategoricalEncoder  # noqa: E402
from .scan_schedule import build_scan_schedule  # noqa: E402

# variant name -> categorical grid axes scanned (Time is always scanned).
# Tensor axis positions in the dense grid (B, T, S, C, Fl, H): T=1, S=2, C=3, Fl=4.
# (variant -> categorical scan axes lives inline in MambaNDModel._build_model;
# a module-level copy here once duplicated it and risked silent divergence.)


class MambaND(nn.Module):
    """Alternating per-axis Mamba-2 scans over the (Time, State, Commodity,
    Flow) lattice. Returns (B, L, G, H) for the kept combos.

    ``scan_schedule`` picks which axis each layer sweeps and in which
    direction; see ``scan_schedule.build_scan_schedule``. The default
    ``"cyclic"`` replaces the original schedule, which phase-locked axis order
    against direction and pinned every axis to one direction for the 2d and 4d
    variants. Pass ``scan_schedule="legacy"`` to reproduce runs recorded before
    that fix.
    """

    def __init__(self, features_per_group, d_model, lattice_dims, combo_coords,
                 cat_axes, n_layers=4, d_state=64, d_conv=4, expand=2,
                 headdim=32, dropout=0.1, scan_schedule="cyclic",
                 bidirectional=True):
        super().__init__()
        self.S, self.C, self.Fl = (int(v) for v in lattice_dims)
        self.N = self.S * self.C * self.Fl
        # Max folded-batch rows per Mamba-2 call (keeps the Triton kernel within
        # grid limits when scanning a short axis folds many rows into batch).
        self._scan_chunk = 8192

        coords = torch.as_tensor(combo_coords, dtype=torch.long)            # (G,3)
        flat = coords[:, 0] * (self.C * self.Fl) + coords[:, 1] * self.Fl + coords[:, 2]
        self.register_buffer("flat_idx", flat)                              # (G,)
        cell_valid = torch.zeros(self.N)
        cell_valid[flat] = 1.0
        self.register_buffer("vmask6",
                             cell_valid.view(1, 1, self.S, self.C, self.Fl, 1))

        self.in_proj = nn.Linear(features_per_group, d_model)
        # Scan axes (tensor positions): Time (1) always + the variant's categorical axes.
        self.scan_positions = [1] + list(cat_axes)
        self.scan_schedule = scan_schedule
        self.layer_pos, self.layer_reverse = build_scan_schedule(
            self.scan_positions, n_layers, schedule=scan_schedule,
            bidirectional=bidirectional,
        )
        # What the schedule ACHIEVES at this depth, per axis -- recorded into
        # metrics.json so a results table can say whether an arm was two-sided
        # rather than trusting the `bidirectional` request in the YAML.
        self.bidirectional_requested = bool(bidirectional)
        _axis_name = {1: "Time", 2: "State", 3: "Commodity", 4: "Flow"}
        seen = {}
        for pos, rev in zip(self.layer_pos, self.layer_reverse):
            seen.setdefault(pos, set()).add("backward" if rev else "forward")
        self.scan_coverage = {
            _axis_name.get(pos, str(pos)): sorted(seen.get(pos, ()))
            for pos in self.scan_positions
        }
        self.mixers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim, use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def _scan_axis(self, x, pos, mixer, norm, reverse):
        """Pre-norm residual Mamba-2 scan along tensor axis ``pos`` of the dense
        grid x = (B, T, S, C, Fl, H). All other grid axes fold into the batch.

        Sparse lattice cells are compacted out before the mixer is called.
        Zeroing them only after a scan is insufficient for a recurrent SSM:
        each artificial step still advances/decays its hidden state and changes
        every real cell after a gap.
        """
        grid_axes = [1, 2, 3, 4]
        others = [a for a in grid_axes if a != pos]
        perm = [0] + others + [pos, 5]                # (B, o1, o2, o3, A, H)
        d = x.permute(*perm).contiguous()
        shp = d.shape
        A, H = shp[-2], shp[-1]
        seq = d.reshape(-1, A, H)                      # (M=B*o1*o2*o3, A, H)

        valid6 = self.vmask6.expand(x.shape[:-1] + (1,))
        valid = (valid6.permute(*perm).contiguous()
                 .reshape(-1, A) > 0)
        mixed = torch.zeros_like(seq)

        # Rows sharing a lattice occupancy pattern can be compacted and mixed
        # together.  The number of patterns is bounded by the small categorical
        # lattice, while batch/time repetitions collapse to the same pattern.
        patterns, pattern_ids = torch.unique(valid, dim=0, return_inverse=True)
        for pattern_id, pattern in enumerate(patterns):
            positions = pattern.nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue
            rows = (pattern_ids == pattern_id).nonzero(as_tuple=False).squeeze(-1)
            compact = seq.index_select(0, rows).index_select(1, positions)
            compact = norm(compact)
            if reverse:
                compact = compact.flip(1)
            # Scanning a short axis folds every other axis into the batch and
            # can overrun Triton's grid limits, so retain the existing chunking.
            if compact.shape[0] <= self._scan_chunk:
                compact = mixer(compact)
            else:
                compact = torch.cat([
                    mixer(compact[i:i + self._scan_chunk])
                    for i in range(0, compact.shape[0], self._scan_chunk)
                ], dim=0)
            if reverse:
                compact = compact.flip(1)
            mixed[rows[:, None], positions[None, :], :] = compact

        seq = (seq + self.dropout(mixed)) * valid.unsqueeze(-1)  # residual on real cells only
        out = seq.reshape(*shp)
        inv = [0] * 6
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        return out.permute(*inv).contiguous()

    def forward(self, x):                              # x: (B, L, G, F)
        B, L, G, _ = x.shape
        h = self.in_proj(x)                            # (B, L, G, H)
        H = h.shape[-1]
        dense = x.new_zeros(B, L, self.N, H)
        dense[:, :, self.flat_idx, :] = h
        dense = dense.view(B, L, self.S, self.C, self.Fl, H)
        for pos, mixer, norm, rev in zip(
                self.layer_pos, self.mixers, self.norms, self.layer_reverse):
            dense = self._scan_axis(dense, pos, mixer, norm, rev) * self.vmask6
        dense = dense.view(B, L, self.N, H)
        return dense[:, :, self.flat_idx, :]           # (B, L, G, H)


class MambaNDModel(BaseModel):
    """Mamba-ND over the combo lattice; last-timestep readout per combo."""

    requires_combo_loader = True

    def _build_model(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        dropout: float = 0.1,
        dims: int = 4,
        scan_schedule: str = "cyclic",
        bidirectional: bool = True,
        depth_policy: str = "fixed",
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
        ), "MambaNDModel needs combo metadata from the combo dataloader."
        cat_axes = {2: (2,), 3: (2, 3), 4: (2, 3, 4)}[dims]
        # Depth policy. One layer scans ONE axis here, unlike s4nd (whose layer
        # convolves every axis, bidirectionally, at any rank) and unlike the
        # ACA models (whose axial module covers every axis in one call). So at
        # a FIXED n_layers the per-axis budget shrinks as the rank grows -- 2
        # layers/axis at 2d, 1 at 4d -- which would make "more grid dimensions
        # perform worse" partly an artifact of the depth choice rather than a
        # finding. "coverage" removes that confound by scaling depth with the
        # axis count (2 * n_axes = 4/6/8), giving every axis exactly one
        # forward and one backward pass at EVERY rank. It costs parameter
        # matching against the 4-layer models. "fixed" is therefore the primary
        # protocol; "coverage" is available only as a separately labelled
        # ablation.
        n_axes = 1 + len(cat_axes)          # Time is always scanned
        if depth_policy == "coverage":
            if n_layers != 2 * n_axes:
                # Say so out loud: the recorded sweep trained 3d/4d at 6/8
                # layers while the YAML read "left at 4 deliberately".
                print(
                    f"[mamba_nd] depth_policy='coverage' overrides n_layers "
                    f"{n_layers} -> {2 * n_axes} (2 per scan axis, {dims}d). "
                    f"Set depth_policy: fixed for the iso-layer arm."
                )
            n_layers = 2 * n_axes
        elif depth_policy != "fixed":
            raise ValueError(
                f"unknown depth_policy {depth_policy!r}; expected 'fixed' or 'coverage'"
            )
        self.depth_policy = depth_policy
        self.n_layers = n_layers
        self.encoder = nn.Identity()
        self.num_groups = num_combos
        # Same information protocol as s4nd (and the axial cross-attention
        # models): re-inject only the identities that are NOT grid axes --
        # 2d -> Commodity+Flow, 3d -> Flow, 4d -> nothing. Without this the
        # scan sees raw features only, so mamba_nd_2d/3d were compared against
        # s4nd_2d/3d while missing categorical information those arms had.
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
        self.grid = MambaND(
            encoded_features, d_model, lattice_dims, combo_coords, cat_axes,
            n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, dropout=dropout, scan_schedule=scan_schedule,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Linear(d_model, 2)
        # Achieved, not requested: True only when every scanned axis got both
        # directions at this depth (see MambaND.scan_coverage).
        self.bidirectional = all(
            len(dirs) == 2 for dirs in self.grid.scan_coverage.values()
        )

    def architecture_provenance(self) -> dict:
        out = super().architecture_provenance()
        out.update({
            "bidirectional_requested": self.grid.bidirectional_requested,
            "scan_coverage": self.grid.scan_coverage,
            "scan_schedule": self.grid.scan_schedule,
            "depth_policy": self.depth_policy,
            "n_layers": int(self.n_layers),
        })
        return out

    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor = None,
        comm_ids: torch.Tensor = None,
        flow_ids: torch.Tensor = None,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.grid(self.group_encoder(x_numeric))     # (B, L, G, H)
        return self.output_proj(h[:, -1, :])            # (B, G, 2)


# matched_axial_* is the same rank alias every other axial family accepts
# (lstm/gru/transformer/gpt/mamba/s4); rejecting it here was an inconsistency
# trap for anyone driving mamba_nd through the shared variant vocabulary.
_AXIAL_DIMS = {
    "cross_attention_2d": 2, "cross_attention_3d": 3, "cross_attention_4d": 4,
    "matched_axial_2d": 2, "matched_axial_3d": 3, "matched_axial_4d": 4,
}


def make_mamba_nd(variant: str, **kwargs) -> BaseModel:
    if variant not in _AXIAL_DIMS:
        raise ValueError(
            f"mamba_nd supports only cross_attention_{{2d,3d,4d}} (got {variant!r})."
        )
    return MambaNDModel(variant=variant, dims=_AXIAL_DIMS[variant], **kwargs)


ModelFactory.register("mamba_nd", make_mamba_nd)
