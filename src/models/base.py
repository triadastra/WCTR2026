"""
Base model class and factory for time series forecasting models.

All models must inherit from BaseModel and implement the required methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .encodings import EncodingFactory, EncodingStrategy


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all forecasting models.

    All models must inherit from this class and implement:
    - forward(): Forward pass
    - predict(): Generate predictions
    - Optional: configure() for model-specific configuration
    """

    def __init__(
        self,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs
    ):
        """
        Initialize base model.

        Args:
            variant: Encoding variant name
                Options: "onehot", "embeddings", "cross_attention", "film_attention"
            num_numeric_features: Number of numeric input features
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions (default: 2)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        self.variant = variant
        self.num_numeric_features = num_numeric_features
        self.num_states = num_states
        self.num_commodities = num_commodities
        self.num_flows = num_flows

        # Create encoding strategy
        self.encoder = self._create_encoder(variant, **kwargs)

        # Materialize the FiLM projection layers NOW, while the feature dim is
        # known. Left to the lazy path they are created inside the first
        # forward() — AFTER the optimizer is built (so they never train: FiLM
        # stays a zero-initialized no-op), on the CPU while activations are on
        # GPU (device-mismatch crash), and absent from a freshly built model at
        # checkpoint-load time (unexpected-keys failure). FiLM modulates the
        # raw numeric channels, so feature_dim == num_numeric_features.
        film = getattr(self.encoder, "film", None)
        if film is not None and hasattr(film, "set_feature_dim"):
            film.set_feature_dim(num_numeric_features)

        # Calculate input dimension after encoding
        self.input_dim = self.encoder.output_dim(num_numeric_features)

        # Model-specific initialization
        self._build_model(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs) -> None:
        """
        Build model architecture.

        Called by __init__ to construct model-specific layers.
        Subclasses must implement this to define their architecture.

        Args:
            **kwargs: Model-specific parameters
        """
        pass

    @abstractmethod
    def forward(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Predictions
                Shape: (batch_size, output_dim)
        """
        pass

    def predict(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate predictions (alias for forward).

        This method can be overridden for more complex prediction logic
        (e.g., autoregressive decoding).

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Predictions
                Shape: (batch_size, output_dim)
        """
        # Restore the caller's train/eval mode afterwards — a bare self.eval()
        # would silently leave dropout disabled if called mid-training.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                predictions = self.forward(x_numeric, state_ids, comm_ids, flow_ids)
        finally:
            if was_training:
                self.train()
        return predictions

    def _create_encoder(self, variant: str, **kwargs) -> EncodingStrategy:
        """
        Create encoding strategy based on variant.

        Args:
            variant: Encoding variant name
            **kwargs: Additional encoder parameters

        Returns:
            EncodingStrategy instance
        """
        # Axial combo variants build a throwaway per-series encoder here;
        # _build_model replaces it with nn.Identity and handles group-aware
        # encoding itself, appending only the identities that are not grid axes
        # (MatchedAxialCategoricalEncoder). cross_attention_* and its
        # matched_axial_* alias behave identically. Reuse the cross_attention
        # spec so the factory accepts either family; self.variant keeps the
        # original name for checkpoints/results.
        if variant == "identity_cross_attention":
            if type(self).__name__ not in ("LSTMComboCAOnly", "GRUComboCAOnly"):
                raise ValueError("identity_cross_attention supports LSTM/GRU combo models only")
            variant = "cross_attention"
        if variant in (
            "cross_attention_2d", "cross_attention_3d", "cross_attention_4d",
            "matched_axial_2d", "matched_axial_3d", "matched_axial_4d",
        ):
            variant = "cross_attention"

        # Extract encoder parameters
        encoder_kwargs = {
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
        }

        # Add variant-specific parameters
        if variant == "onehot":
            pass  # No additional params needed
        elif variant == "embeddings":
            encoder_kwargs.update({
                "state_embed_dim": kwargs.get("state_embed_dim", 8),
                "comm_embed_dim": kwargs.get("comm_embed_dim", 32),
                "flow_embed_dim": kwargs.get("flow_embed_dim", 2),
            })
        elif variant in ["cross_attention", "film_attention"]:
            encoder_kwargs.update({
                "state_embed_dim": kwargs.get("state_embed_dim", 8),
                "comm_embed_dim": kwargs.get("comm_embed_dim", 32),
                "flow_embed_dim": kwargs.get("flow_embed_dim", 2),
                "num_numeric_features": self.num_numeric_features,
                "num_heads": kwargs.get("cross_attn_num_heads", 4),
                "d_k": kwargs.get("cross_attn_d_k", 32),
                "dropout": kwargs.get("dropout", 0.0),
            })
            if variant == "film_attention":
                encoder_kwargs.update({
                    "film_hidden_dim": kwargs.get("film_hidden_dim", 64),
                    "film_num_layers": kwargs.get("film_num_layers", 2),
                })

        return EncodingFactory.create(variant, **encoder_kwargs)

    def encode_features(
        self,
        x_numeric: torch.Tensor,
        state_ids: torch.Tensor,
        comm_ids: torch.Tensor,
        flow_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode features using the configured encoding strategy.

        This is a convenience method that can be used by subclasses
        to encode features before processing.

        Args:
            x_numeric: Numeric time series features
                Shape: (batch_size, seq_len, num_numeric_features)
            state_ids: State indices
                Shape: (batch_size,)
            comm_ids: Commodity indices
                Shape: (batch_size,)
            flow_ids: Flow direction indices
                Shape: (batch_size,)

        Returns:
            Encoded features
                Shape: (batch_size, seq_len, encoded_dim)
        """
        return self.encoder.encode(x_numeric, state_ids, comm_ids, flow_ids)

    def get_num_params(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_provenance(self) -> dict:
        """Facts about the built network that a results table must carry.

        Written into ``metrics.json`` by ``scripts/inference.py`` so a
        leaderboard can show, next to each arm, whether it is two-sided over
        the input window (``bidirectional``: True / False / None when the
        notion does not apply, e.g. trees) and which encoder revision it ran
        (``architecture_versions``, e.g. ``categorical_attention_v2``).
        Subclasses extend it (see ``MambaNDModel`` for achieved scan coverage).
        """
        bidirectional = getattr(self, "bidirectional", None)
        versions = {}
        for name, module in self.named_modules():
            version = getattr(module, "architecture_version", None)
            if version:
                versions[name or "model"] = str(version)
        return {
            "class": type(self).__name__,
            "variant": self.variant,
            "num_params": int(self.get_num_params()),
            "bidirectional": None if bidirectional is None else bool(bidirectional),
            "architecture_versions": versions,
        }

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "variant": self.variant,
            "num_numeric_features": self.num_numeric_features,
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])


class ModelFactory:
    """
    Factory for creating model instances.

    Provides a unified interface for creating models by name and variant.
    """

    _models = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """
        Register a model class.

        Args:
            name: Model name (e.g., "lstm", "gru", "transformer")
            model_class: Model class (must inherit from BaseModel)
        """
        cls._models[name] = model_class

    @classmethod
    def create(
        cls,
        name: str,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            name: Model name
            variant: Encoding variant
            num_numeric_features: Number of numeric input features
            num_states: Number of unique states
            num_commodities: Number of unique commodities
            num_flows: Number of flow directions
            **kwargs: Additional model-specific parameters

        Returns:
            BaseModel instance

        Raises:
            ValueError: If model name is unknown
        """
        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available models: {list(cls._models.keys())}"
            )

        model_class = cls._models[name]

        return model_class(
            variant=variant,
            num_numeric_features=num_numeric_features,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
            **kwargs
        )

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())


# Convenience function for creating models
def create_model(
    name: str,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int = 2,
    **kwargs
) -> BaseModel:
    """
    Create a model instance.

    This is a convenience wrapper around ModelFactory.create().

    Args:
        name: Model name (e.g., "lstm", "gru", "transformer")
        variant: Encoding variant
        num_numeric_features: Number of numeric input features
        num_states: Number of unique states
        num_commodities: Number of unique commodities
        num_flows: Number of flow directions
        **kwargs: Additional model-specific parameters

    Returns:
        BaseModel instance
    """
    return ModelFactory.create(
        name=name,
        variant=variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
        **kwargs
    )
