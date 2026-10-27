"""Deterministic seed control for reproducible training and inference.

Usage:
    from src.utils.seeding import set_seed, seed_worker

    g = set_seed(42)
    loader = DataLoader(ds, ..., generator=g, worker_init_fn=seed_worker)
"""

import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Seed every RNG used during training and inference.

    Sets Python's ``random``, NumPy, PyTorch (CPU + all CUDA devices), and
    ``PYTHONHASHSEED``. Enables cuDNN deterministic mode and disables the
    cuDNN autotuner, requests deterministic algorithms from PyTorch, and sets
    ``CUBLAS_WORKSPACE_CONFIG`` so deterministic cuBLAS GEMMs are available on
    CUDA. All CUDA-specific knobs are no-ops / harmless on CPU-only setups.

    Returns a ``torch.Generator`` seeded with ``seed`` that should be passed
    to every ``DataLoader`` via ``generator=g`` so dataloader shuffling is
    also reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs under
    # torch.use_deterministic_algorithms; harmless on CPU-only setups.
    #
    # cuBLAS reads this variable when its handle is created, which happens on
    # the first CUDA matmul. Setting it here works only because set_seed() is
    # called before any device work; if CUDA is already up the assignment is a
    # no-op and determinism silently degrades to whatever cuBLAS chose. Say so
    # rather than letting a run claim a reproducibility it does not have.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            warnings.warn(
                "CUDA was already initialized before set_seed(): "
                "CUBLAS_WORKSPACE_CONFIG is set too late for cuBLAS to read it, "
                "so deterministic GEMMs are not guaranteed for this run. Call "
                "set_seed() before any CUDA work, or export "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment.",
                RuntimeWarning,
                stacklevel=2,
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Request deterministic algorithms globally. warn_only=True keeps ops that
    # lack a deterministic implementation working (with a warning) instead of
    # raising, so CPU-only and unusual ops still run.
    torch.use_deterministic_algorithms(True, warn_only=True)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for per-worker determinism.

    PyTorch already seeds each worker's ``torch`` RNG deterministically from
    the base seed, but NumPy and Python ``random`` are not — they have to be
    seeded explicitly here, or augmentations / shuffles inside the worker
    process drift between runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
