"""
Inference module for making predictions with trained models.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import create_model


class Predictor:
    """
    Unified predictor for all models.

    Handles loading trained models and making predictions.
    """

    def __init__(self, checkpoint_path: str, config: Dict[str, Any]):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.device = self._get_device()

        # Load model
        self.model = self._load_model()

        # Load artifacts
        self.artifacts = self._load_artifacts()

    def _get_device(self) -> torch.device:
        """Get device from config."""
        training_cfg = self.config.get("training", {})
        device_str = training_cfg.get("device", "cuda")

        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Get model info
        model_cfg = self.config.get("model", {})

        # Load artifacts to get vocab sizes
        artifacts_dir = Path(self.checkpoint_path).parent.parent / "artifacts"

        if artifacts_dir.exists():
            # Load ID maps
            with open(artifacts_dir / "id_maps.json", "r") as f:
                id_maps = json.load(f)

            num_states = id_maps["num_states"]
            num_commodities = id_maps["num_commodities"]
            num_flows = id_maps["num_flows"]

            # Load features
            with open(artifacts_dir / "features.json", "r") as f:
                features = json.load(f)
            num_numeric_features = len(features["feat_cols"])
        else:
            raise FileNotFoundError(f"Artifacts not found at {artifacts_dir}")

        # Create model
        model = create_model(
            name=model_cfg.get("name", "lstm"),
            variant=model_cfg.get("variant", "embeddings"),
            num_numeric_features=num_numeric_features,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
            **model_cfg,
        )

        # Load state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def _load_artifacts(self) -> Dict[str, Any]:
        """Load preprocessing artifacts."""
        artifacts_dir = Path(self.checkpoint_path).parent.parent / "artifacts"

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts not found at {artifacts_dir}")

        # Load stats
        with open(artifacts_dir / "series_stats.json", "r") as f:
            stats_data = json.load(f)

        stats = {
            "exact": pd.DataFrame(stats_data["exact"]),
            "cf": pd.DataFrame(stats_data["cf"]),
            "c": pd.DataFrame(stats_data["c"]),
            "global": stats_data["global"],
        }

        # Load ID maps
        with open(artifacts_dir / "id_maps.json", "r") as f:
            id_maps = json.load(f)

        # Load features
        with open(artifacts_dir / "features.json", "r") as f:
            features = json.load(f)

        return {
            "stats": stats,
            "state2id": id_maps["state2id"],
            "comm2id": id_maps["comm2id"],
            "flow2id": id_maps["flow2id"],
            "feat_cols": features["feat_cols"],
        }

    def predict(
        self,
        dataloader: DataLoader,
        denormalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on a dataset.

        Args:
            dataloader: Data loader
            denormalize: Whether to denormalize predictions

        Returns:
            Dictionary with predictions, targets, and metadata
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_state_ids = []
        all_comm_ids = []
        all_flow_ids = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                x_numeric = batch["x_numeric"].to(self.device)
                state_ids = batch["state_ids"].to(self.device)
                comm_ids = batch["comm_ids"].to(self.device)
                flow_ids = batch["flow_ids"].to(self.device)
                target_value = batch["target_value"].to(self.device)
                target_weight = batch["target_weight"].to(self.device)

                # Predict
                predictions = self.model(x_numeric, state_ids, comm_ids, flow_ids)

                # Collect
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(
                    torch.stack([target_value, target_weight], dim=-1).cpu().numpy()
                )
                all_state_ids.append(state_ids.cpu().numpy())
                all_comm_ids.append(comm_ids.cpu().numpy())
                all_flow_ids.append(flow_ids.cpu().numpy())

        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        state_ids = np.concatenate(all_state_ids, axis=0)
        comm_ids = np.concatenate(all_comm_ids, axis=0)
        flow_ids = np.concatenate(all_flow_ids, axis=0)

        # Denormalize if requested
        if denormalize:
            predictions = self._denormalize(
                predictions,
                state_ids,
                comm_ids,
                flow_ids,
            )
            targets = self._denormalize(
                targets,
                state_ids,
                comm_ids,
                flow_ids,
            )

        return {
            "predictions": predictions,
            "targets": targets,
            "state_ids": state_ids,
            "comm_ids": comm_ids,
            "flow_ids": flow_ids,
        }

    def _denormalize(
        self,
        data: np.ndarray,
        state_ids: np.ndarray,
        comm_ids: np.ndarray,
        flow_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Denormalize predictions/targets.

        Args:
            data: Normalized data (n_samples, 2)
            state_ids: State IDs
            comm_ids: Commodity IDs
            flow_ids: Flow IDs

        Returns:
            Denormalized data
        """
        # This is a simplified denormalization
        # In practice, you'd need to look up the exact stats for each series
        stats = self.artifacts["stats"]
        global_stats = stats["global"]

        denorm = data.copy()
        denorm[:, 0] = data[:, 0] * global_stats["val_std"] + global_stats["val_mean"]
        denorm[:, 1] = data[:, 1] * global_stats["wt_std"] + global_stats["wt_mean"]

        return denorm

    def predict_batch(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make predictions on a single batch.

        Args:
            x_numeric: Numeric features
            state_ids: State IDs
            comm_ids: Commodity IDs
            flow_ids: Flow IDs

        Returns:
            Predictions
        """
        self.model.eval()

        with torch.no_grad():
            x_numeric = x_numeric.to(self.device)
            state_ids = state_ids.to(self.device)
            comm_ids = comm_ids.to(self.device)
            flow_ids = flow_ids.to(self.device)

            predictions = self.model(x_numeric, state_ids, comm_ids, flow_ids)

        return predictions.cpu()
