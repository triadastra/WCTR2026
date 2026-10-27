"""Random-forest tabular forecasting baseline.

WHY ADD A THIRD TREE ARM
------------------------
xgboost and lightgbm are both GRADIENT BOOSTING. Random forest is bagging, so
it separates "trees behave this way on this data" from "boosting behaves this
way" -- a distinction the benchmark could not previously make, and one that
matters for interpreting the tree family's behaviour under the 2024 structural
break.

It also fills out the Tree row of the vs-baseline structure heatmap, which
otherwise has two models against SSM's four.

CATEGORICAL HANDLING, AND WHY IT DIFFERS FROM LIGHTGBM
------------------------------------------------------
LightGBM splits natively on `category` dtype columns, which is why
lightgbm.py routes X through to_feature_frame(). scikit-learn's forests have no
categorical support -- they would treat a category dtype as an error and an
ordinal code as a number with a meaningful order. So this model consumes the
RAW numeric matrix from extract_tabular_xy(): under `onehot` the identities are
already indicator columns and nothing is lost, while under `embeddings` the
state/commodity IDs are ordinal codes and the forest can only split them by
arbitrary index ranges.

That is a real limitation, not an oversight, and it is the standard way an RF
baseline is run on this kind of panel. It also predicts the direction of the
result: RF should do relatively better under `onehot` than `embeddings`, the
opposite of the neural arms. Report the pair rather than only the better one.
"""

import numpy as np
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel, ModelFactory
from .tabular_common import (
    extract_tabular_xy,
    fallback_feature_names,
    load_tree_bundle,
    save_tree_bundle,
    validate_tabular_variant,
)


class RandomForestModel(BaseModel):
    """Random-forest peer baseline using the same rows as XGBoost/LightGBM."""

    def __init__(
        self,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs,
    ):
        validate_tabular_variant("RandomForest", variant)
        super().__init__(
            variant=variant,
            num_numeric_features=num_numeric_features,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
            **kwargs,
        )
        self.encoder = nn.Identity()
        self.requires_training = False
        self.model_value = None
        self.model_weight = None
        self.feature_names = fallback_feature_names(
            variant,
            num_numeric_features,
            num_states,
            num_commodities,
            num_flows,
        )

    def _build_model(
        self,
        n_estimators: int = 300,
        max_depth: int = None,
        min_samples_leaf: int = 2,
        # Breiman's p/3 for regression. sklearn's default of 1.0 considers every
        # feature at every split, which removes the decorrelation bagging is for
        # and is markedly slower on this feature count.
        max_features: float = 0.33,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.rf_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_jobs": n_jobs,
            "random_state": int(random_state),
        }

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "RandomForest model does not support forward(). Use fit() and predict()."
        )

    def _extract_numpy(self, loader) -> tuple[np.ndarray, np.ndarray]:
        X, y, feature_names = extract_tabular_xy(
            loader,
            variant=self.variant,
            num_numeric_features=self.num_numeric_features,
            num_states=self.num_states,
            num_commodities=self.num_commodities,
            num_flows=self.num_flows,
        )
        self.feature_names = feature_names
        return X, y

    def fit(self, train_loader, val_loader=None, verbose: bool = True) -> None:
        X_train, y_train = self._extract_numpy(train_loader)
        # val_loader is accepted for interface parity with the boosting arms but
        # unused: a forest has no early stopping to spend a validation split on.
        if verbose:
            print(
                f"RandomForest tabular features: {X_train.shape[1]} "
                f"({self.variant}); rows={X_train.shape[0]}"
            )
            print("Training value model...")
        self.model_value = RandomForestRegressor(**self.rf_params)
        self.model_value.fit(X_train, y_train[:, 0])
        if verbose:
            print("Training weight model...")
        self.model_weight = RandomForestRegressor(**self.rf_params)
        self.model_weight.fit(X_train, y_train[:, 1])

    def evaluate_loader(self, loader) -> dict[str, float]:
        preds, targets = self.predict_loader(loader)
        err = preds - targets
        return {
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
            "value_mae": float(np.mean(np.abs(err[:, 0]))),
            "weight_mae": float(np.mean(np.abs(err[:, 1]))),
        }

    def predict_loader(self, loader) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._extract_numpy(loader)
        return self.predict(X), y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_value is None or self.model_weight is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return np.column_stack(
            [self.model_value.predict(X), self.model_weight.predict(X)]
        )

    def save(self, path: str, run_contract=None) -> None:
        metadata = {
            "variant": self.variant,
            "num_numeric_features": self.num_numeric_features,
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
            "rf_params": self.rf_params,
            "feature_names": self.feature_names,
            "run_contract": run_contract,
            "model_complexity": {
                "value_trees": len(getattr(self.model_value, "estimators_", [])),
                "weight_trees": len(getattr(self.model_weight, "estimators_", [])),
            },
        }
        save_tree_bundle(path, self.model_value, self.model_weight, metadata)

    def load(self, path: str) -> None:
        model_value, model_weight, metadata = load_tree_bundle(path)
        expected = {
            "variant": self.variant,
            "num_numeric_features": self.num_numeric_features,
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(
                "RandomForest checkpoint identity mismatch "
                f"({', '.join(mismatches)}); refusing to relabel the trained model"
            )
        self.model_value = model_value
        self.model_weight = model_weight
        self.rf_params = metadata["rf_params"]
        self.run_contract = metadata.get("run_contract")
        self.feature_names = metadata.get(
            "feature_names",
            fallback_feature_names(
                self.variant,
                self.num_numeric_features,
                self.num_states,
                self.num_commodities,
                self.num_flows,
            ),
        )


ModelFactory.register("random_forest", RandomForestModel)
