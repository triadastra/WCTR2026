"""Occupancy masks for fixed-coordinate grids (batch, spatial axes, time)."""

import torch


def mask_grid(x, mask, *, transposed=False):
    """Zero absent tokens, including nonfinite padding, without moving cells."""
    if mask is None:
        return x
    expected = (x.shape[0], *x.shape[2:]) if transposed else x.shape[:-1]
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(expected):
        raise ValueError("grid_mask must be boolean and match every non-feature axis")
    mask = mask.to(device=x.device)
    return x.masked_fill(~mask.unsqueeze(1 if transposed else -1), 0.0)
