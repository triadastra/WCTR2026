"""Cross-group attention, identity-aware attention, FiLM and axial mixing.

The legacy combo blocks originated in the paper implementation. Mask guards
and the separately named identity variant are subsequent corrections/extensions.
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention import safe_attention_softmax


# Compatibility shim copied from paper. Some PyTorch builds lack nn.init.full_.
if not hasattr(nn.init, "full_"):
    def _full_(tensor, fill_value):
        return nn.init.constant_(tensor, float(fill_value))
    nn.init.full_ = _full_


class CrossAttention(nn.Module):
    """Single-head scaled dot-product attention ACROSS GROUPS, per timestep.

    Input:
        x: (B, L, G, Fin)
        mask_g (optional): (B, G) or (B, L, G), 1=valid, 0=padded

    Output:
        h: (B, L, G, H)
        attn_weights: (B, L, G, G)
    """

    def __init__(self, feature_dim: int, hidden_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask_g: Optional[torch.Tensor] = None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.einsum("blgh,blkh->blgk", Q, K) * self.scale

        if mask_g is not None:
            if mask_g.dim() == 2:
                mask_g = mask_g[:, None, :]
            key_mask = (mask_g[:, :, None, :] > 0).to(scores.dtype)
            scores = scores.masked_fill(key_mask == 0, float("-inf"))

        attn = safe_attention_softmax(scores, self)
        attn = self.dropout(attn)
        h = torch.einsum("blgk,blkh->blgh", attn, V)
        h = self.out_norm(h)
        if mask_g is not None:
            h = h.masked_fill(~(mask_g > 0).unsqueeze(-1), 0.0)
        return h, attn


class IdentityCrossAttention(CrossAttention):
    """Experimental group attention: identity informs Q/K, values stay numeric."""
    architecture_version = "identity_cross_attention_v1"

    def __init__(self, feature_dim, hidden_dim, combo_coords, num_states,
                 num_commodities, num_flows, attn_dropout=0.0):
        super().__init__(feature_dim, hidden_dim, attn_dropout)
        if combo_coords is None:
            raise ValueError("identity_cross_attention requires combo_coords")
        coords = torch.as_tensor(combo_coords, dtype=torch.long)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("combo_coords must have shape (groups, 3)")
        self.register_buffer("identity_coords", coords)
        self.identity_embeddings = nn.ModuleList([
            nn.Embedding(n, hidden_dim) for n in
            (num_states, num_commodities, num_flows)])
        for emb in self.identity_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        self.identity_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.identity_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask_g=None):
        if x.shape[2] != len(self.identity_coords):
            raise ValueError("group count differs from identity coordinates")
        identity = sum(emb(self.identity_coords[:, i]) for i, emb in
                       enumerate(self.identity_embeddings))
        Q = self.q_proj(x) + self.identity_q(identity)[None, None]
        K = self.k_proj(x) + self.identity_k(identity)[None, None]
        V = self.v_proj(x)
        scores = torch.einsum("blgh,blkh->blgk", Q, K) * self.scale
        if mask_g is not None:
            if mask_g.dim() == 2:
                mask_g = mask_g[:, None, :]
            scores = scores.masked_fill(~(mask_g[:, :, None, :] > 0), float("-inf"))
        attn = self.dropout(safe_attention_softmax(scores, self))
        h = self.out_norm(torch.einsum("blgk,blkh->blgh", attn, V))
        if mask_g is not None:
            h = h.masked_fill(~(mask_g > 0).unsqueeze(-1), 0.0)
        return h, attn


class LatentCrossAttention(nn.Module):
    """Perceiver-style cross-attention: G groups -> M latents -> G groups.
    O(G*M) instead of O(G^2). Used when ``num_latents > 0``.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_latents: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert num_latents > 0, "num_latents must be > 0 for LatentCrossAttention"
        self.num_latents = num_latents
        self.q_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask_g: Optional[torch.Tensor] = None):
        B, L, G, _ = x.shape
        x = x.reshape(B * L, G, -1)
        Q_in = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        if mask_g is not None:
            if mask_g.dim() == 2:
                mask_g = mask_g[:, None, :].expand(B, L, G)
            mask_flat = mask_g.reshape(B * L, G).to(Q_in.dtype)
        else:
            mask_flat = None

        # Stage 1: latents <- inputs
        Q_lat = self.latents[None, :, :].expand(B * L, -1, -1)
        scores_1 = torch.einsum("bmh,bgh->bmg", Q_lat, K) * self.scale
        if mask_flat is not None:
            key_mask = mask_flat[:, None, :].to(scores_1.dtype)
            scores_1 = scores_1.masked_fill(key_mask == 0, float("-inf"))
        attn_1 = safe_attention_softmax(scores_1, self)
        attn_1 = self.dropout(attn_1)
        Z = torch.einsum("bmg,bgh->bmh", attn_1, V)

        # Stage 2: inputs <- latents
        scores_2 = torch.einsum("bgh,bmh->bgm", Q_in, Z) * self.scale
        attn_2 = torch.softmax(scores_2, dim=-1)
        attn_2 = self.dropout(attn_2)
        H_out = torch.einsum("bgm,bmh->bgh", attn_2, Z)

        if mask_flat is not None:
            H_out = H_out * mask_flat[:, :, None]

        H_out = self.out_norm(H_out)
        if mask_flat is not None:
            H_out = H_out.masked_fill(~(mask_flat > 0).unsqueeze(-1), 0.0)
        return H_out.reshape(B, L, G, -1), None


class FiLMByGroup(nn.Module):
    """FiLM conditioner + per-group gate ("yank weight").

    Produces ``(gamma, beta)`` and a scalar gate per group, conditioned on:
      - the group slot embedding (learned per group index)
      - the global context (masked mean over groups)
    """

    def __init__(
        self,
        num_groups: int,
        hidden_dim: int,
        film_hidden: int = 128,
        gate_temperature: float = 1.0,
        min_gate: float = 0.0,
        normalize_group_weights: bool = False,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.gate_temperature = gate_temperature
        self.min_gate = float(min_gate)
        self.normalize_group_weights = normalize_group_weights

        self.group_embed = nn.Embedding(num_groups, film_hidden)
        nn.init.normal_(self.group_embed.weight, std=0.02)

        # FiLM MLP: [context || group_embed] -> (gamma, beta).
        # Last layer zero-initialised => starts as identity FiLM.
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_dim + film_hidden, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, 2 * hidden_dim),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Gate MLP -> scalar per group. Bias init -2.0 => small initial gates.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim + film_hidden, film_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden // 2, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.full_(self.gate_mlp[-1].bias, -2.0)

    def forward(self, h_attn: torch.Tensor, mask_g: Optional[torch.Tensor] = None):
        B, L, G, H = h_attn.shape

        if mask_g is None:
            mask = torch.ones(B, L, G, 1, device=h_attn.device, dtype=h_attn.dtype)
        else:
            mask = (
                mask_g[:, None, :, None].to(h_attn.dtype) if mask_g.dim() == 2
                else mask_g[:, :, :, None].to(h_attn.dtype)
            )

        # Global context = masked mean over groups
        denom = mask.sum(dim=2).clamp_min(1e-6)
        ctx = (h_attn * mask).sum(dim=2) / denom

        gid = torch.arange(G, device=h_attn.device)
        gemb = self.group_embed(gid)[None, None, :, :].expand(B, L, G, -1)

        ctx_exp = ctx[:, :, None, :].expand(-1, -1, G, -1)
        film_in = torch.cat([ctx_exp, gemb], dim=-1)
        film_out = self.film_mlp(film_in)
        gamma, beta = film_out.chunk(2, dim=-1)

        y = (1.0 + gamma) * h_attn + beta

        g_logits = self.gate_mlp(film_in) / self.gate_temperature
        w = torch.sigmoid(g_logits)
        if self.min_gate > 0.0:
            w = self.min_gate + (1.0 - self.min_gate) * w
        w = w * mask
        if self.normalize_group_weights:
            w = w / w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        y = y * w

        return y, w


def build_cross_attention(
    features_per_group: int,
    hidden_size: int,
    attn_dropout: float = 0.0,
    num_latents: int = 0,
) -> nn.Module:
    """Pick CrossAttention (default) or LatentCrossAttention (if num_latents>0)."""
    if num_latents and num_latents > 0:
        return LatentCrossAttention(
            feature_dim=features_per_group,
            hidden_dim=hidden_size,
            num_latents=num_latents,
            attn_dropout=attn_dropout,
        )
    return CrossAttention(
        feature_dim=features_per_group,
        hidden_dim=hidden_size,
        attn_dropout=attn_dropout,
    )


class MatchedAxialCategoricalEncoder(nn.Module):
    """Append embeddings only for identities absent from the ND grid axes.

    Port of s4nd's ``CFIDEncoder`` (external/s4/src/tasks/encoders.py) to the
    axial cross-attention path, so ACA and s4nd see the same information at
    every rank. The axial variants were modelled on S4ND but originally fed the
    backbone raw features only, which starved cross_attention_2d/3d of the
    commodity/flow identities s4nd_2d/3d re-inject -- see the s4nd dataset
    configs (s4nd_3d carries ``encoder_extra_dim: 2``, the flow embedding).

    This mirrors the S4ND information protocol used by the benchmark:

    ``dims=2``
        State is structural; append Commodity and Flow embeddings.
    ``dims=3``
        State and Commodity are structural; append a Flow embedding.
    ``dims=4``
        State, Commodity, and Flow are all structural; append nothing.

    ``combo_coords`` stores the static ``(state, commodity, flow)`` coordinate
    for every group in a combo batch. Embeddings are therefore computed once
    per group and broadcast over batch and time, rather than redundantly being
    carried in every dataloader sample.
    """

    def __init__(
        self,
        features_per_group: int,
        combo_coords,
        dims: int,
        num_commodities: int,
        num_flows: int,
        comm_embed_dim: int = 32,
        flow_embed_dim: int = 2,
    ) -> None:
        super().__init__()
        if dims not in (2, 3, 4):
            raise ValueError(f"dims must be 2, 3 or 4 (got {dims})")

        coords = torch.as_tensor(combo_coords, dtype=torch.long)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "combo_coords must have shape (num_groups, 3) for "
                "(state, commodity, flow)"
            )
        self.register_buffer("commodity_ids", coords[:, 1])
        self.register_buffer("flow_ids", coords[:, 2])
        self.dims = dims
        self.features_per_group = int(features_per_group)

        self.commodity_embedding = None
        self.flow_embedding = None
        extra_dim = 0
        if dims == 2:
            self.commodity_embedding = nn.Embedding(
                num_commodities, comm_embed_dim
            )
            nn.init.normal_(self.commodity_embedding.weight, std=0.02)
            extra_dim += comm_embed_dim
        if dims in (2, 3):
            self.flow_embedding = nn.Embedding(num_flows, flow_embed_dim)
            nn.init.normal_(self.flow_embedding.weight, std=0.02)
            extra_dim += flow_embed_dim

        self.extra_dim = extra_dim
        self.output_dim = self.features_per_group + extra_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected combo input (B,L,G,F), got {tuple(x.shape)}")
        if x.shape[2] != self.commodity_ids.numel():
            raise ValueError(
                f"input has {x.shape[2]} groups but combo_coords has "
                f"{self.commodity_ids.numel()}"
            )
        if x.shape[-1] != self.features_per_group:
            raise ValueError(
                f"input has {x.shape[-1]} features per group; expected "
                f"{self.features_per_group}"
            )

        embeddings = []
        if self.commodity_embedding is not None:
            embeddings.append(self.commodity_embedding(self.commodity_ids))
        if self.flow_embedding is not None:
            embeddings.append(self.flow_embedding(self.flow_ids))
        if not embeddings:
            return x

        group_features = torch.cat(embeddings, dim=-1).to(dtype=x.dtype)
        group_features = group_features.view(1, 1, x.shape[2], -1)
        group_features = group_features.expand(x.shape[0], x.shape[1], -1, -1)
        return torch.cat([x, group_features], dim=-1)


class AxialComboCA(nn.Module):
    """Cross-attention applied SEPARATELY along each chosen lattice axis.

    The RNN analogue of S4ND's separable (per-axis) grid mixing. The flat
    combo set is scattered onto the dense ``(State, Commodity, Flow)``
    lattice, cross-attention is run independently along each selected axis
    (padded/non-kept cells masked out), then the kept-combo cells are
    gathered back. The number of axes mirrors the S4ND grid variants:

        dims=2 -> attend along State                  (cf. s4nd_2d)
        dims=3 -> attend along State, then Commodity  (cf. s4nd_3d)
        dims=4 -> attend along State, Commodity, Flow (cf. s4nd_4d)

    Input:  x  (B, L, G, F)
    Output: h  (B, L, G, H)
    """

    _AXES = {2: (0,), 3: (0, 1), 4: (0, 1, 2)}

    def __init__(self, features_per_group, hidden_dim, lattice_dims,
                 combo_coords, dims, attn_dropout: float = 0.0):
        super().__init__()
        assert dims in self._AXES, f"dims must be 2, 3 or 4 (got {dims})"
        self.S, self.C, self.Fl = (int(v) for v in lattice_dims)
        self.N = self.S * self.C * self.Fl
        self.dims = dims
        self.axes = self._AXES[dims]

        coords = torch.as_tensor(combo_coords, dtype=torch.long)            # (G, 3)
        flat = coords[:, 0] * (self.C * self.Fl) + coords[:, 1] * self.Fl + coords[:, 2]
        self.register_buffer("flat_idx", flat)                              # (G,)
        cell_valid = torch.zeros(self.N)
        cell_valid[flat] = 1.0
        self.register_buffer("cell_valid", cell_valid.view(self.S, self.C, self.Fl))
        self.register_buffer("vmask6",
                             cell_valid.view(1, 1, self.S, self.C, self.Fl, 1))

        self.in_proj = nn.Linear(features_per_group, hidden_dim)
        self.ca = nn.ModuleList(
            [CrossAttention(hidden_dim, hidden_dim, attn_dropout) for _ in self.axes]
        )

    def _axis_ca(self, dense, ax, ca):
        """Attend across lattice axis ``ax`` in {0:State,1:Commodity,2:Flow}.

        ``dense`` is (B, L, S, C, Fl, H). We fold the two non-attended lattice
        axes into the batch so CrossAttention attends across the chosen axis.
        """
        B, L, S, C, Fl, H = dense.shape
        sizes = (S, C, Fl)
        pos = 2 + ax
        others = [2 + i for i in range(3) if i != ax]
        perm = [0] + others + [1, pos, 5]                  # (B, o1, o2, L, A, H)
        d = dense.permute(*perm).contiguous()
        o1, o2 = (sizes[i] for i in range(3) if i != ax)
        A = sizes[ax]
        d = d.view(B * o1 * o2, L, A, H)
        # validity mask for this axis: cell_valid -> (o1, o2, A) -> (B*o1*o2, A)
        vm = self.cell_valid.permute(*[i for i in range(3) if i != ax], ax).contiguous()
        vm = vm.view(1, o1 * o2, A).expand(B, o1 * o2, A).reshape(B * o1 * o2, A)
        # Lines with NO valid cell (e.g. a (commodity,flow) column with zero kept
        # states) would mask every key -> softmax over all -inf -> NaN in both
        # forward and backward. Give such dead lines an all-valid mask (their
        # output is zeroed by vmask6 afterwards), so no query row is ever fully
        # masked and gradients stay finite.
        dead = vm.sum(dim=-1, keepdim=True) == 0
        vm = torch.where(dead, torch.ones_like(vm), vm)
        out, _ = ca(d, mask_g=vm)                          # (B*o1*o2, L, A, H)
        out = torch.nan_to_num(out)                        # belt-and-suspenders
        out = out.view(B, o1, o2, L, A, H)
        inv = [0] * 6
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        return out.permute(*inv).contiguous()

    def forward(self, x):                                  # (B, L, G, F)
        B, L, G, _ = x.shape
        h = self.in_proj(x)                                # (B, L, G, H)
        H = h.shape[-1]
        dense = x.new_zeros(B, L, self.N, H)
        dense[:, :, self.flat_idx, :] = h
        dense = dense.view(B, L, self.S, self.C, self.Fl, H)
        for ax, ca in zip(self.axes, self.ca):
            dense = (dense + self._axis_ca(dense, ax, ca)) * self.vmask6
        dense = dense.view(B, L, self.N, H)
        return dense[:, :, self.flat_idx, :]               # (B, L, G, H)
