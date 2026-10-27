from .base import BaseModel, ModelFactory, create_model

# Importing each model file registers it with ModelFactory. Order doesn't
# matter; we just need the side-effect imports so ``create_model("lstm", ...)``
# can find the class.
from . import lstm  # noqa: F401
from . import gru  # noqa: F401
from . import transformer  # noqa: F401
from . import gpt  # noqa: F401
try:
    from . import xgboost  # noqa: F401
except Exception as _xgboost_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"XGBoost unavailable ({_xgboost_err!r}); 'xgboost' model not registered.")
try:
    from . import lightgbm  # noqa: F401
except Exception as _lightgbm_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"LightGBM unavailable ({_lightgbm_err!r}); 'lightgbm' model not registered.")
# Random forest (sklearn). Bagging rather than boosting, so it separates "trees
# behave this way" from "boosting behaves this way" -- xgboost and lightgbm are
# both boosting and cannot distinguish the two on their own.
try:
    from . import random_forest  # noqa: F401
except Exception as _rf_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"sklearn unavailable ({_rf_err!r}); 'random_forest' model not registered.")
# Real Mamba-2 (vendored state-spaces/mamba under external/mamba/). It needs
# triton/GPU, so import it DEFENSIVELY: on CPU-only machines / fresh clones the
# import fails and 'mamba' is simply not registered, while `import src.models`
# still works. The old ssm_models LSTM "mamba" stub is intentionally NOT
# imported, so a missing real Mamba errors loudly rather than silently
# training an LSTM under the 'mamba' name.
try:
    from . import mamba  # noqa: F401
except Exception as _mamba_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Mamba-2 unavailable ({_mamba_err!r}); 'mamba' model not registered.")
# S4 uses the authors' modular backbone, including in aggregate and axial arms.
# Imports are isolated from our src package; there is no standalone fallback.
try:
    from . import s4  # noqa: F401
except Exception as _s4_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Real S4 unavailable ({_s4_err!r}); 's4' model not registered.")
# Mamba-ND grid model (external/Mamba-ND method over the Mamba-2 SSD mixer).
# Needs triton/GPU like 'mamba'; import defensively so CPU-only imports still work.
try:
    from . import mamba_nd  # noqa: F401
except Exception as _mnd_err:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"Mamba-ND unavailable ({_mnd_err!r}); 'mamba_nd' model not registered.")
from . import baselines  # noqa: F401  # seasonal_naive, random_walk, moving_average

__all__ = ["BaseModel", "ModelFactory", "create_model"]
