"""Architecture contract checks for the flat Exp1 attention arms."""
import pytest
import torch

from src.models.gpt import CausalSelfAttention
from src.models.transformer import TransformerModel


def test_invalid_position_encoding_cannot_silently_change_architecture():
    with pytest.raises(ValueError, match="positional_encoding"):
        TransformerModel(variant="embeddings", num_numeric_features=2,
                         num_states=1, num_commodities=1,
                         positional_encoding="sinusoidl")


def test_gpt_attention_does_not_read_future_positions():
    torch.manual_seed(26)
    attention = CausalSelfAttention(d_model=16, nhead=4, dropout=0.0).eval()
    x = torch.randn(2, 8, 16)
    changed = x.clone()
    changed[:, 4:] += 100.0
    with torch.no_grad():
        original = attention(x)
        perturbed = attention(changed)
    torch.testing.assert_close(original[:, :4], perturbed[:, :4], rtol=0, atol=0)


def test_transformer_examples_do_not_attend_across_batch():
    torch.manual_seed(26)
    model = TransformerModel(variant="embeddings", num_numeric_features=2,
                             num_states=1, num_commodities=1, num_flows=1,
                             hidden_size=16, num_heads=4, num_layers=2,
                             d_ff=32, dropout=0.0).eval()
    x = torch.randn(2, 8, 2)
    changed = x.clone()
    changed[1] += 100.0
    ids = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        original = model(x, ids, ids, ids)
        perturbed = model(changed, ids, ids, ids)
    torch.testing.assert_close(original[0], perturbed[0], rtol=0, atol=0)
