"""S4 target selection, importable without either runtime's src package."""
import torch


def grid_cell_mask(shape, dataset, extra, *, device):
    """Resolve occupancy from identities, never from observed target values.

    ``shape`` is (batch, categorical axes...), without time or feature axes.
    Missing cells retain their lattice coordinates; they are not compacted.
    """
    shape = tuple(shape)
    mask = None
    if "valid_mask" in extra:
        mask = torch.as_tensor(extra["valid_mask"], device=device)
    elif getattr(dataset, "kept_mask", None) is not None:
        kept = torch.as_tensor(dataset.kept_mask, device=device)
        if tuple(kept.shape) == shape[1:]:
            mask = kept.unsqueeze(0).expand(shape)
        elif len(shape) == 3 and kept.ndim == 3 and tuple(kept.shape[:2]) == shape[1:]:
            if "flow_ids" not in extra:
                raise ValueError("Per-flow grid requires flow_ids")
            flow = torch.as_tensor(extra["flow_ids"], device=device)
            if flow.ndim == 0 or flow.shape[0] != shape[0] or flow.numel() == 0:
                raise ValueError("Per-flow grid has misaligned flow_ids")
            flow = flow.reshape(shape[0], -1)
            ids = flow[:, 0].long()
            if not ((flow == ids[:, None]).all() and (ids >= 0).all()
                    and (ids < kept.shape[-1]).all()):
                raise ValueError("Per-flow grid requires one valid flow ID per sample")
            mask = kept[:, :, ids].permute(2, 0, 1)
        else:
            raise ValueError("Unsupported kept-mask/output shape")
    elif str(getattr(dataset, "_name_", "")).startswith("s4nd_"):
        raise ValueError("S4ND trade grid is missing its occupancy mask")
    if mask is not None:
        if tuple(mask.shape) != shape or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Empty or misaligned kept mask")
        mask = mask.bool()
        if not mask.any():
            raise ValueError("Empty or misaligned kept mask")
    return mask


def select_valid_targets(pred, target, dataset, extra):
    if pred.shape != target.shape:
        raise ValueError("Prediction/target shape mismatch")
    mask = grid_cell_mask(pred.shape[:-1], dataset, extra, device=pred.device)
    if mask is not None:
        return pred[mask], target[mask]
    return pred, target
