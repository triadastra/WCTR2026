"""
Loss functions for time series forecasting.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean"):
        """
        Initialize MSE loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            predictions: (batch_size, 2) - [value, weight]
            targets: (batch_size, 2) - [value, weight]

        Returns:
            Loss value
        """
        return self.mse(predictions, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error loss."""

    def __init__(self, reduction: str = "mean"):
        """
        Initialize MAE loss.

        Args:
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.mae = nn.L1Loss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE loss.

        Args:
            predictions: (batch_size, 2) - [value, weight]
            targets: (batch_size, 2) - [value, weight]

        Returns:
            Loss value
        """
        return self.mae(predictions, targets)


class WeightedLoss(nn.Module):
    """Weighted combination of value and weight losses."""

    def __init__(self, value_weight: float = 1.0, weight_weight: float = 1.0):
        """
        Initialize weighted loss.

        Args:
            value_weight: Weight for value loss
            weight_weight: Weight for weight loss
        """
        super().__init__()
        self.value_weight = value_weight
        self.weight_weight = weight_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            predictions: (batch_size, 2) - [value, weight]
            targets: (batch_size, 2) - [value, weight]

        Returns:
            Loss value
        """
        loss_value = self.mse(predictions[:, 0], targets[:, 0])
        loss_weight = self.mse(predictions[:, 1], targets[:, 1])

        return self.value_weight * loss_value + self.weight_weight * loss_weight


class MaskedMSELoss(nn.Module):
    """Group-masked MSE loss for combo-window models.

    Combo models output ``(B, G, 2)``: one prediction per (state,
    commodity, flow) combination at each batch element. Not every combo
    exists at every target time — the combo dataloader marks invalid
    slots in ``group_mask`` ``(B, G)`` (1=valid, 0=padded).

    The training objective is lane-weighted MSE: every valid series-time
    observation has equal weight, without first averaging within each state.
    Value and weight channels are weighted equally.

    This loss ignores padded slots so the optimizer doesn't chase zeros
    for missing series. Output scale matches per-series MSE: total
    squared error over valid (B*G) slots divided by the count of valid
    slots, then averaged over the 2 targets.

    Matches the masking convention in
    ``paper included/{LSTM,GRU}/inference_*.py`` walk-forward eval.
    """

    def forward(
        self,
        predictions: torch.Tensor,        # (B, G, 2)
        target_value: torch.Tensor,       # (B, G)
        target_weight: torch.Tensor,      # (B, G)
        group_mask: torch.Tensor,         # (B, G), 1=valid, 0=padded
    ) -> torch.Tensor:
        targets = torch.stack([target_value, target_weight], dim=-1)  # (B, G, 2)
        sq_err = (predictions - targets) ** 2                          # (B, G, 2)
        mask = group_mask.unsqueeze(-1).to(sq_err.dtype)               # (B, G, 1)
        masked = sq_err * mask
        denom = mask.sum().clamp_min(1.0) * sq_err.shape[-1]            # valid_slots * 2_targets
        return masked.sum() / denom


class MaskedMAELoss(nn.Module):
    """Group-masked MAE loss — same convention as MaskedMSELoss."""

    def forward(
        self,
        predictions: torch.Tensor,
        target_value: torch.Tensor,
        target_weight: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = torch.stack([target_value, target_weight], dim=-1)
        abs_err = (predictions - targets).abs()
        mask = group_mask.unsqueeze(-1).to(abs_err.dtype)
        masked = abs_err * mask
        denom = mask.sum().clamp_min(1.0) * abs_err.shape[-1]
        return masked.sum() / denom


class HuberLoss(nn.Module):
    """Huber loss (smooth L1 loss)."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold for transitioning from quadratic to linear
            reduction: Reduction method
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: (batch_size, 2) - [value, weight]
            targets: (batch_size, 2) - [value, weight]

        Returns:
            Loss value
        """
        residual = torch.abs(predictions - targets)
        # clamp, not torch.min against a fresh CPU tensor — the latter is a
        # device mismatch as soon as predictions live on GPU.
        quadratic = torch.clamp(residual, max=self.delta)
        linear = residual - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
