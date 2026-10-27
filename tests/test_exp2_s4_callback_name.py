"""Hydra S4 checkpoints carry callback `_name_` echoes that composed configs lack."""
import copy

from src.utils.exp2_protocol import comparable_s4_contract


def _contract():
    return {"model": "s4", "training_config": {"callbacks": {
        "early_stopping": {"monitor": "val/loss", "patience": 50},
        "model_checkpoint": {"monitor": "val/loss"},
    }, "optimizer": {"lr": 1e-4}}}


def _stamped(contract):
    stamped = copy.deepcopy(contract)
    for key, callback in stamped["training_config"]["callbacks"].items():
        callback["_name_"] = key
    return stamped


def test_self_named_callbacks_compare_equal():
    expected = _contract()
    assert comparable_s4_contract(_stamped(expected)) == comparable_s4_contract(expected)


def test_input_is_not_mutated():
    stamped = _stamped(_contract())
    before = copy.deepcopy(stamped)
    comparable_s4_contract(stamped)
    assert stamped == before


def test_foreign_callback_name_is_still_a_difference():
    stamped = _stamped(_contract())
    stamped["training_config"]["callbacks"]["early_stopping"]["_name_"] = "timer"
    assert comparable_s4_contract(stamped) != comparable_s4_contract(_contract())


def test_recipe_changes_are_still_differences():
    for path, value in ((("callbacks", "early_stopping", "patience"), 49),
                        (("optimizer", "lr"), 1e-3)):
        changed = _stamped(_contract())
        node = changed["training_config"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        assert comparable_s4_contract(changed) != comparable_s4_contract(_contract())


def test_non_contract_values_pass_through():
    assert comparable_s4_contract(None) is None
    assert comparable_s4_contract({"model": "gru"}) == {"model": "gru"}
