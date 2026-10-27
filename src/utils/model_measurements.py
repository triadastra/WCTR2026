"""Measured model size; native trees are not zero-parameter neural networks."""
import math


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # e.g. XGBoost's missing=nan setting, not a metric
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def model_measurements(model, family):
    if family in {"xgboost", "lightgbm", "random_forest"}:
        result = {"parameter_count_kind": "not applicable: native tree estimators"}
        for channel in ("value", "weight"):
            estimator = getattr(model, f"model_{channel}", None)
            if estimator is None:
                continue
            measured = {}
            if hasattr(estimator, "get_params"):
                measured["estimator_parameters"] = estimator.get_params()
            if hasattr(estimator, "estimators_"):
                trees = list(estimator.estimators_)
                measured.update(trees=len(trees), nodes=sum(t.tree_.node_count for t in trees),
                                leaves=sum(t.tree_.n_leaves for t in trees))
            elif hasattr(estimator, "booster_"):
                booster = estimator.booster_
                measured.update(trees=booster.num_trees(), features=booster.num_feature())
            elif hasattr(estimator, "get_booster"):
                booster = estimator.get_booster()
                measured.update(boosted_rounds=booster.num_boosted_rounds(), features=booster.num_features())
            result[channel] = measured
        return _json_value(result)
    parameters = list(model.named_parameters())
    total = sum(p.numel() for _, p in parameters)
    trainable = sum(p.numel() for _, p in parameters if p.requires_grad)
    return {
        "parameter_count_kind": "torch parameter elements",
        "total_parameters": total, "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "parameter_bytes": sum(p.numel() * p.element_size() for _, p in parameters),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "tensors": [{"name": name, "shape": list(p.shape), "elements": p.numel(),
                     "dtype": str(p.dtype), "trainable": p.requires_grad}
                    for name, p in parameters],
    }
