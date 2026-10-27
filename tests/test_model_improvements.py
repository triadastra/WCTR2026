import copy

import pytest
import torch

from src.models import create_model
from src.models.attention import MultiHeadAttention, PositionalEncoding
from src.models.combo_attention import CrossAttention, IdentityCrossAttention, LatentCrossAttention
from src.models.encodings import CrossAttentionEncoding, FiLMAttentionEncoding


@pytest.mark.parametrize("encoder_class", [CrossAttentionEncoding, FiLMAttentionEncoding])
def test_categorical_attention_uses_numeric_queries_and_trains_qk(encoder_class):
    torch.manual_seed(12)
    enc = encoder_class(3, 4, num_numeric_features=5, dropout=0)
    x = torch.randn(2, 6, 5)
    ids = (torch.tensor([0, 1]), torch.tensor([1, 3]), torch.tensor([0, 1]))
    seen = []
    handle = enc.cross_attn.register_forward_pre_hook(lambda m, args: seen.append(args))
    out = enc.encode(x, *ids)
    handle.remove()
    assert seen[0][0] is x
    assert seen[0][1].shape == (2, 3, enc.embed_dim)
    context = out[..., 5:]
    assert not torch.allclose(context[:, 0], context[:, 1], atol=1e-9)
    context.square().sum().backward()
    for projection in (enc.cross_attn.W_q, enc.cross_attn.W_k):
        assert projection.weight.grad.abs().sum() > 0
    restored = copy.deepcopy(enc)
    restored.load_state_dict(enc.state_dict())
    torch.testing.assert_close(restored.encode(x, *ids), out)


@pytest.mark.parametrize("kind", ["plain", "latent", "identity"])
def test_empty_group_attention_is_zero_finite_and_logged_once(kind, caplog):
    if kind == "plain":
        module = CrossAttention(5, 8)
    elif kind == "latent":
        module = LatentCrossAttention(5, 8, 2)
    else:
        module = IdentityCrossAttention(5, 8, [[0, 0, 0], [1, 1, 1]], 2, 2, 2)
    module.out_norm.bias.data.fill_(2)  # Learned norm bias must not revive padding.
    x = torch.randn(2, 3, 2, 5, requires_grad=True)
    mask = torch.tensor([[0, 0], [1, 0]])
    for _ in range(2):
        out, _ = module(x, mask)
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[0]) == 0
        assert torch.count_nonzero(out[1, :, 1]) == 0
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert sum("all-masked" in r.message for r in caplog.records) == 1


def test_multihead_all_masked_rows_have_finite_gradients():
    m = MultiHeadAttention(8, 2, dropout=0)
    x = torch.randn(2, 3, 8, requires_grad=True)
    out = m(x, x, x, torch.zeros(2, 3, 3))
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_positional_dropout_and_odd_dimension():
    m = PositionalEncoding(7, dropout=1)
    x = torch.ones(2, 5, 7)
    assert torch.count_nonzero(m(x)) == 0
    m.eval()
    torch.testing.assert_close(m(x), x + m.pe[:, :5])


@pytest.mark.parametrize("name", ["lstm", "gru"])
def test_bidirectional_head_uses_final_hidden_states(name):
    m = create_model(name, "embeddings", 5, 2, 2,
                     hidden_size=8, num_layers=2, bidirectional=True, dropout=0)
    x = torch.randn(2, 6, 5)
    ids = torch.tensor([0, 1])
    projected = m.input_proj(m.encode_features(x, ids, ids, ids))
    _, hidden = getattr(m, name)(projected)
    if name == "lstm":
        hidden = hidden[0]
    expected = m.output_proj(torch.cat([hidden[-2], hidden[-1]], dim=-1))
    torch.testing.assert_close(m(x, ids, ids, ids), expected)


@pytest.mark.parametrize("name", ["lstm", "gru"])
def test_identity_variant_preserves_residual_and_trains_identity(name):
    torch.manual_seed(9)
    m = create_model(name, "identity_cross_attention", 5, 2, 2,
                     hidden_size=8, num_layers=1, dropout=0,
                     num_combos=2, features_per_group=5,
                     combo_coords=[[0, 0, 0], [1, 1, 1]])
    x = torch.randn(2, 4, 2, 5, requires_grad=True)
    assert m.requires_combo_loader
    out = m(x)
    assert out.shape == (2, 2, 2)
    out.square().sum().backward()
    assert m.in_proj.weight.grad.abs().sum() > 0
    for emb in m.cross_attention.identity_embeddings:
        assert emb.weight.grad.abs().sum() > 0
    assert m.cross_attention.identity_q.weight.grad.abs().sum() > 0
    assert m.cross_attention.identity_k.weight.grad.abs().sum() > 0


def test_identity_changes_attention_weights():
    torch.manual_seed(19)
    m = IdentityCrossAttention(5, 8, [[0, 0, 0], [1, 1, 1]], 2, 2, 2)
    x = torch.randn(2, 3, 2, 5)
    _, before = m(x)
    with torch.no_grad():
        m.identity_embeddings[0].weight[0].add_(3)
    _, after = m(x)
    assert not torch.allclose(before, after)


def test_lightgbm_subsampling_reaches_fitted_booster():
    from src.models.lightgbm import LightGBMModel
    import numpy as np
    m = LightGBMModel("onehot", 4, 2, 2, n_estimators=3,
                      min_child_samples=2, n_jobs=1)
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, len(m.feature_names)))
    y = rng.normal(size=(40, 2))
    m._extract_numpy = lambda loader: (X, y)
    m.fit(None, verbose=False)
    assert m.model_value.booster_.params["subsample_freq"] == 1
    assert m.model_value.booster_.params["subsample"] == 0.9
    assert np.isfinite(m.predict(X)).all()


def test_training_log_records_architecture_and_attention_guard(tmp_path):
    from src.training.trainer import Trainer
    m = create_model("gru", "identity_cross_attention", 5, 2, 2,
                     hidden_size=8, num_layers=1, num_combos=2,
                     features_per_group=5, combo_coords=[[0, 0, 0], [1, 1, 1]])
    trainer = Trainer({
        "training": {"device": "cpu", "scheduler": "none", "use_amp": False},
        "checkpointing": {"save_dir": str(tmp_path / "checkpoints")},
        "logging": {"log_dir": str(tmp_path / "logs"), "experiment_name": "guard"},
    })
    trainer.setup(m, [], [])
    m(torch.randn(1, 3, 2, 5), group_mask=torch.zeros(1, 2))
    trainer.logger.close(state="finished")
    log = (tmp_path / "logs" / "guard.log").read_text(encoding="utf-8")
    assert "identity_cross_attention_v1" in log
    assert "all-masked" in log
    assert "attention guards=v2" in log


def test_identity_variant_rejects_other_backbones():
    with pytest.raises(ValueError, match="LSTM/GRU"):
        create_model("transformer", "identity_cross_attention", 5, 2, 2)
