"""The SSMs mark parameters that must be exempt from weight decay.

Handing ``model.parameters()`` straight to the optimizer silently overrides
both conventions, which applies weight decay to state-space parameters their
authors deliberately exempt -- on an SSM that pulls the state matrix toward
zero and degrades the dynamics the model is built on. The hydra pipeline
(external/s4/train.py) honors them, so without this the same S4 trained on
either side is not the same run.
"""

import torch
import torch.nn as nn

from src.training.trainer import Trainer


class _Stub(nn.Module):
    """Carries one parameter of each kind the optimizer has to separate."""

    def __init__(self):
        super().__init__()
        self.plain = nn.Parameter(torch.zeros(2))
        # Mamba convention: external/mamba/.../mamba2.py marks A_log, D, dt_bias
        self.a_log = nn.Parameter(torch.zeros(2))
        self.a_log._no_weight_decay = True
        # S4 convention: S4Kernel.register() attaches an _optim dict
        self.kernel_a = nn.Parameter(torch.zeros(2))
        self.kernel_a._optim = {"lr": 0.001, "weight_decay": 0.0}


def _groups(optimizer_name="adamw", weight_decay=1e-4, lr=1e-3):
    trainer = Trainer.__new__(Trainer)
    trainer.model = _Stub()
    trainer.optimizer_name = optimizer_name
    trainer.lr = lr
    trainer.betas = (0.9, 0.999)
    trainer.eps = 1e-8
    trainer.weight_decay = weight_decay
    trainer.weight_decay_exempt = True
    trainer._create_optimizer()
    return trainer.optimizer.param_groups


def test_no_weight_decay_params_get_their_own_group():
    groups = _groups()
    zero_wd = [g for g in groups if g["weight_decay"] == 0.0]
    assert zero_wd, "_no_weight_decay / _optim params must not inherit wd"
    exempt = {id(p) for g in zero_wd for p in g["params"]}
    # Only the two the authors marked. `plain` is 1-D as well but nothing asked
    # for it to be spared, so it keeps the decay.
    assert len(exempt) == 2


def test_unmarked_models_get_one_plain_group():
    """A model whose authors marked nothing is trained exactly as before: one
    group, everything decayed. The usual AdamW norm/bias exemption is NOT added
    on top -- no author of these models asked for it, and imposing a third
    convention on architectures from different papers is its own distortion."""
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.optimizer_name = "adamw"
    trainer.lr = 1e-3
    trainer.betas = (0.9, 0.999)
    trainer.eps = 1e-8
    trainer.weight_decay = 1e-4
    trainer.weight_decay_exempt = True
    trainer._create_optimizer()
    assert len(trainer.optimizer.param_groups) == 1
    assert len(trainer.optimizer.param_groups[0]["params"]) == 4


def test_s4_optim_dict_overrides_are_applied():
    groups = _groups(lr=5e-3)
    special = [g for g in groups if len(g["params"]) == 1 and g["lr"] == 0.001]
    assert special, "an _optim lr must override the global lr"
    assert special[0]["weight_decay"] == 0.0


def test_every_parameter_lands_in_exactly_one_group():
    groups = _groups()
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == 3, "params must not be dropped or duplicated"


def test_real_s4_kernel_parameters_are_exempt():
    """End-to-end on the genuine S4Block, which runs on CPU here."""
    from src.models.s4 import make_s4

    model = make_s4(variant="embeddings", num_numeric_features=4, num_states=2,
                    num_commodities=2, num_flows=2, d_model=16, n_layers=1,
                    d_state=8)
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.optimizer_name = "adamw"
    trainer.lr = 1e-3
    trainer.betas = (0.9, 0.999)
    trainer.eps = 1e-8
    trainer.weight_decay = 1e-4
    trainer.weight_decay_exempt = True
    trainer._create_optimizer()

    zero_wd = [p for p in model.parameters()
               if getattr(p, "_optim", {}).get("weight_decay") == 0.0]
    assert zero_wd, "the real S4 kernel should tag A/B/dt with weight_decay=0"
    exempt = {id(p) for g in trainer.optimizer.param_groups
              if g["weight_decay"] == 0.0 for p in g["params"]}
    assert all(id(p) in exempt for p in zero_wd), \
        "S4 kernel parameters that ask for wd=0 must not receive weight decay"
    # C is registered with wd=None, so it keeps the global decay -- on BOTH
    # sides. Faithfulness to the hydra pipeline, not an oversight.


def test_the_ablation_switch_decays_everything():
    """weight_decay_exempt: false reproduces the pre-exemption runs -- one group,
    nothing spared. The reported winner was trained that way, and decaying A_log
    is a shrinkage prior rather than a bug, so which is better is an open
    question the switch exists to settle."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = _Stub()
    trainer.optimizer_name = "adamw"
    trainer.lr = 1e-3
    trainer.betas = (0.9, 0.999)
    trainer.eps = 1e-8
    trainer.weight_decay = 1e-4
    trainer.weight_decay_exempt = False
    trainer._create_optimizer()

    groups = trainer.optimizer.param_groups
    assert len(groups) == 1, "the ablation arm must not split parameters"
    assert groups[0]["weight_decay"] == 1e-4
    assert len(groups[0]["params"]) == 3, "including the SSM-marked ones"
