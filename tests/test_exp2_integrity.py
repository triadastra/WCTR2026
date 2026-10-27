"""Experiment 2 leakage, optimization, provenance, and completion guards."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataloader import DataConfig, TradeDataPipeline
from src.utils.exp2_protocol import expected_runs, expected_contract, validate_contract, resolved_s4_config
from src.utils.experiment_manifest import load_manifest
from src.utils.s4_benchmark import select_valid_targets

ROOT = Path(__file__).resolve().parents[1]


def _panel():
    rows = []
    for state, scale in (("A", 1), ("B", 2)):
        for i, month in enumerate(pd.date_range("2017-01-01", "2021-05-01", freq="MS")):
            rows.append({"State": state, "Commodity": "01", "Country": "X", "Import/Export": "Import",
                         "Time": month, "Value": scale * (i + 1), "Weight": scale * (i + 3)})
    return pd.DataFrame(rows)


def _pipe(frame):
    pipe = TradeDataPipeline(DataConfig(start="2017-01-01", end="2021-05-01", input_len=3, lag_count=2,
        train_end="2019-12-01", val_start="2020-01-01", val_end="2020-12-01",
        test_start="2021-01-01", test_end="2021-05-01"))
    pipe.df_full = frame
    pipe.create_splits()
    return pipe


def test_future_values_cannot_change_training_cohort_scaling_or_inputs():
    frame = _panel()
    changed = frame.copy()
    changed.loc[changed.Time > "2019-12-01", ["Value", "Weight"]] = 1e12
    original, perturbed = _pipe(frame), _pipe(changed)
    assert original.kept_combos == perturbed.kept_combos
    pd.testing.assert_frame_equal(original.df_train, perturbed.df_train)
    for name in original.stats:
        if isinstance(original.stats[name], pd.DataFrame):
            pd.testing.assert_frame_equal(original.stats[name], perturbed.stats[name])
    a = original.get_dataloaders(batch_size=2, num_workers=0, shuffle_train=False)[0]
    b = perturbed.get_dataloaders(batch_size=2, num_workers=0, shuffle_train=False)[0]
    for x, y in zip(a, b):
        for key in x:
            if torch.is_tensor(x[key]):
                torch.testing.assert_close(x[key], y[key])


def test_target_month_and_later_values_never_enter_its_input():
    frame = _panel()
    changed = frame.copy()
    changed.loc[changed.Time >= "2021-03-01", ["Value", "Weight"]] *= 100
    pipes = [_pipe(f) for f in (frame, changed)]
    for combo in (False, True):
        loaders = [p.get_dataloaders(batch_size=1, num_workers=0, combo=combo)[2] for p in pipes]
        if not combo:
            for a, b in zip(loaders[0].dataset.samples, loaders[1].dataset.samples):
                if a["target_time"] <= pd.Timestamp("2021-03-01"):
                    torch.testing.assert_close(a["x_numeric"], b["x_numeric"])
        else:
            for i in range(3):
                torch.testing.assert_close(loaders[0].dataset[i]["x_numeric"], loaders[1].dataset[i]["x_numeric"])


@pytest.mark.parametrize("layout", ["state", "groups", "flow", "full"])
def test_s4_loss_and_gradients_ignore_padding(layout):
    if layout == "state":
        shape = (2, 3, 2)
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        ds, extra = SimpleNamespace(), {"valid_mask": mask}
    elif layout == "groups":
        shape = (2, 2, 3, 2)
        kept = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        ds, extra = SimpleNamespace(kept_mask=kept), {}
        mask = kept[None].expand(shape[:-1])
    elif layout == "flow":
        shape = (2, 2, 3, 2)
        kept = torch.tensor([[[1, 0], [0, 1], [1, 0]], [[0, 0], [1, 1], [0, 0]]], dtype=torch.bool)
        ds, extra = SimpleNamespace(kept_mask=kept), {"flow_ids": torch.tensor([[[0]], [[1]]])}
        mask = kept.permute(2, 0, 1)
    else:
        shape = (2, 2, 3, 2, 2)
        kept = torch.tensor([[[1, 0], [0, 1], [1, 0]], [[0, 0], [1, 1], [0, 0]]], dtype=torch.bool)
        ds, extra = SimpleNamespace(kept_mask=kept), {}
        mask = kept[None].expand(shape[:-1])
    pred = torch.ones(shape, requires_grad=True)
    target = torch.zeros(shape)
    target[~mask] = 1e6
    p, t = select_valid_targets(pred, target, ds, extra)
    loss = (p - t).square().mean()
    assert loss.item() == 1.0
    loss.backward()
    assert torch.count_nonzero(pred.grad[~mask]) == 0
    torch.testing.assert_close(pred.grad[mask], torch.full_like(pred.grad[mask], 2 / p.numel()))


def test_s4_empty_or_missing_masks_fail_closed():
    with pytest.raises(ValueError, match="Empty"):
        select_valid_targets(torch.ones(1, 2, 2), torch.zeros(1, 2, 2), SimpleNamespace(),
                             {"valid_mask": torch.zeros(1, 2, dtype=torch.bool)})
    with pytest.raises(ValueError, match="flow_ids"):
        select_valid_targets(torch.ones(1, 2, 3, 2), torch.zeros(1, 2, 3, 2),
                             SimpleNamespace(kept_mask=torch.ones(2, 3, 2)), {})


def test_manifest_has_exactly_57_arms_and_three_paired_seeds():
    runs = expected_runs(load_manifest(ROOT / "config/experiment_manifest.env"))
    assert len(runs) == 171
    assert len({(r.model, r.variant) for r in runs}) == 57
    assert {r.seed for r in runs} == {26, 105, 115}
    assert len({r.directory_name for r in runs}) == len(runs)


@pytest.mark.parametrize("kind", ["native", "baseline", "s4"])
def test_exp2_rejects_training_recipe_drift(kind):
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    run = next(r for r in expected_runs(manifest) if r.kind == kind)
    contract = expected_contract(ROOT, run, manifest)
    payload = {"model": run.model, "variant": run.variant, "seed": run.seed, "run_contract": contract}
    assert validate_contract(payload, ROOT)[0]
    bad = copy.deepcopy(payload)
    bad["run_contract"]["training_config"]["epochs"] = 1
    assert not validate_contract(bad, ROOT)[0]
    bad = copy.deepcopy(payload)
    bad["run_contract"]["data_config"]["cohort_from_train_only"] = False
    assert not validate_contract(bad, ROOT)[0]


def test_s4_recipe_has_no_private_lr_decay_or_last_epoch_test():
    manifest = load_manifest(ROOT / "config/experiment_manifest.env")
    for run in expected_runs(manifest, [26]):
        if run.kind != "s4":
            continue
        cfg = resolved_s4_config(ROOT, run)
        assert cfg["model"]["layer"]["lr"] is None
        assert cfg["model"]["layer"]["d_state"] == 64
        assert cfg["train"]["optimizer_param_grouping"] is None
        assert cfg["train"]["test"] is False
        assert cfg["trainer"]["val_check_interval"] == 1.0
        assert cfg["loader"]["drop_last"] is False
        assert cfg["callbacks"]["early_stopping"]["min_delta"] == 1e-4


def test_lightning_reduces_and_weights_by_real_targets():
    source = (ROOT / "external/s4/train.py").read_text()
    tree = ast.parse(source)
    step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_shared_step")
    # Run the actual method without importing the CUDA/Lightning runtime.
    namespace = {"select_valid_targets": select_valid_targets}
    exec(compile(ast.Module(body=[step], type_ignores=[]), "train.py", "exec"), namespace)
    class Fake:
        dataset = SimpleNamespace()
        task = SimpleNamespace(torchmetrics=lambda *args: None)
        loss = loss_val = staticmethod(lambda p, t, **kw: (p - t).square().mean())
        metrics = staticmethod(lambda *args, **kw: {})
        def _process_state(self, *args, **kw): pass
        def forward(self, batch): return batch[0], batch[1], {}
        def log_dict(self, metrics, **kw): self.logged = metrics, kw
    fake = Fake()
    batch = (torch.ones(2, 3, 2), torch.zeros(2, 3, 2),
             {"valid_mask": torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)})
    assert namespace["_shared_step"](fake, batch, 0, "val").item() == 1.0
    assert fake.logged[1]["batch_size"] == 3


def test_paper_collector_rejects_missing_exp2_seeds(tmp_path):
    import shutil
    from scripts.build_paper_results import _exp2_rows
    (tmp_path / 'config').mkdir()
    shutil.copy(ROOT / 'config/experiment_manifest.env', tmp_path / 'config/experiment_manifest.env')
    (tmp_path / 'outputs/full').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='missing 171'):
        _exp2_rows(tmp_path)


def test_s4_evaluation_failure_returns_nonzero(tmp_path, monkeypatch):
    source = ast.parse((ROOT / 'scripts/s4_eval.py').read_text())
    main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def fail(*args):
        raise RuntimeError('bad checkpoint')
    namespace = {'Path': Path, 'RUNS': tmp_path, 'EXPERIMENTS': {'s4_embeddings': ''}, 'evaluate': fail}
    exec(compile(ast.Module(body=[main], type_ignores=[]), 's4_eval.py', 'exec'), namespace)
    (tmp_path / 's4_s4_embeddings_seed26').mkdir()
    monkeypatch.setattr('sys.argv', ['s4_eval.py', 's4_embeddings', '--seed', '26'])
    assert namespace['main']() == 1
    monkeypatch.setattr('sys.argv', ['s4_eval.py', 's4_embeddings', '--seed', '105'])
    assert namespace['main']() == 1
    monkeypatch.setattr('sys.argv', ['s4_eval.py'])
    assert namespace['main']() == 1


def test_s4_onehot_contract_uses_public_variant_name():
    manifest = load_manifest(ROOT / 'config/experiment_manifest.env')
    run = next(r for r in expected_runs(manifest) if r.experiment == 's4_onehot')
    assert expected_contract(ROOT, run, manifest)['identity']['variant'] == 'onehot'
