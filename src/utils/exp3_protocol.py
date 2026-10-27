"""Single contract for Exp 3 launch, resume, and publication."""
from itertools import product
from pathlib import Path

from .config import load_config
from .experiment_manifest import load_manifest, exp1_source_domain
from .run_contract import build_run_contract

ROOT = Path(__file__).resolve().parents[2]


def manifest():
    return load_manifest(ROOT / 'config/experiment_manifest.env')


def coordinates():
    m = manifest()
    return tuple(product(m['PAPER_SWEEP_MODELS'].split(),
                         map(int, m['PAPER_WINDOWS'].split()),
                         map(int, m['PAPER_LAGS'].split()),
                         map(int, m['PAPER_SEEDS'].split())))


def run_name(model, window, lag, seed):
    return f'{model}_embeddings_w{window}_l{lag}_seed{seed}'


def resolved_config(model, window, lag, seed):
    if (model, window, lag, seed) not in coordinates():
        raise ValueError('run coordinates are outside the canonical Exp 3 grid')
    m = manifest()
    cfg = load_config(str(ROOT / 'config/base.yaml'))
    cfg = load_config(str(ROOT / f'config/models/{model}.yaml'), cfg)
    cfg = load_config(str(ROOT / 'config/variants/embeddings.yaml'), cfg)
    cfg = apply_exp3_config(cfg, model)
    cfg['data'].update(input_len=window, lag_count=lag,
                       train_target_start=m['PAPER_SWEEP_TRAIN_START'])
    cfg['training']['batch_size'] = int(m['PAPER_SWEEP_BATCH'])
    return cfg


def expected_contract(model, window, lag, seed):
    m = manifest()
    return build_run_contract(resolved_config(model, window, lag, seed),
                              model=model, variant='embeddings', seed=seed,
                              aggregate=False,
                              source_fingerprint=m['PAPER_SOURCE_FINGERPRINT'],
                              source_domain=exp1_source_domain(m))


def validate_contract(payload, model, window, lag, seed):
    try:
        expected = expected_contract(model, window, lag, seed)
    except (ValueError, KeyError) as exc:
        return False, str(exc)
    identity = {'model': model, 'variant': 'embeddings', 'seed': seed}
    if any(payload.get(k) != v or type(payload.get(k)) is not type(v)
           for k, v in identity.items()):
        return False, 'Exp 3 payload identity differs from directory coordinates'
    if payload.get('run_contract') != expected:
        return False, 'Exp 3 model/data/training/selection contract is not canonical'
    from .training_completion import validate_completion
    try:
        validate_completion(payload.get('training_completion'), expected)
    except (ValueError, TypeError) as exc:
        return False, str(exc)
    return True, 'ok'


def apply_exp3_config(cfg, model):
    """Explicit experiment overlay; keep other experiments' heads unchanged."""
    if model == 'transformer':
        cfg['model']['prediction_head'] = 'linear'
    return cfg
