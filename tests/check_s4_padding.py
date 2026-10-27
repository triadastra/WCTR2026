"""Exercise real S4ND encoding, layers, loss and checkpoint rejection on CPU.

Run from external/s4: python ../../tests/check_s4_padding.py
No training outputs or production data are modified.
"""
import copy
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path.cwd()))
import train
import src.dataloaders.trade_unified as unified
import src.dataloaders.trade_grid as grid
from s4_benchmark import grid_cell_mask
from s4_test_support import synthetic


def month_panel(*args, **kwargs):
    pipe = synthetic(*args, **kwargs)
    pipe.config.batch_by_target_month = True
    return pipe


def run():
    unified._canonical_pipeline = month_panel
    grid._clean_pipeline = lambda root, length: month_panel(root, length, 2)
    torch.set_num_threads(1)
    for transposed in (False, True):
        for exp in ('s4nd_2d_state', 's4nd_2d', 's4nd_3d', 's4nd_4d'):
            torch.manual_seed(26)
            with initialize_config_dir(config_dir=str(Path.cwd() / 'configs'), version_base=None):
                cfg = compose(config_name='config', overrides=[
                    f'experiment={exp}', 'wandb=null', 'dataset.input_len=3',
                    'model.d_model=8', 'model.n_layers=3', 'model.layer.d_state=4',
                    'model.layer.lr=null', 'model.dropout=0.0',
                    f'model.transposed={str(transposed).lower()}', 'loader.num_workers=0',
                ])
            cfg = train.utils.train.process_config(cfg)
            model = train.SequenceLightningModule(cfg).eval()
            model._state = None
            # Nonzero learned offsets must not reactivate absent cells.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith('bias') and parameter.is_floating_point():
                        parameter.fill_(0.25)
            x, y, extra = next(iter(model.dataset.train_dataloader(
                batch_size=1, shuffle=False, drop_last=False, num_workers=0)))
            valid = grid_cell_mask(y.shape[:-1], model.dataset, extra, device=x.device)
            assert valid.any() and (~valid).any()
            token_mask = valid[..., None].expand(x.shape[:-1])
            checked_layers = []

            def layer_checked(module, args, result):
                output = result[0]
                if transposed:
                    output = output.movedim(1, -1)
                assert torch.count_nonzero(output[~token_mask]) == 0
                checked_layers.append(module)

            handles = [layer.register_forward_hook(layer_checked) for layer in model.model.layers]
            expected = model((x, y, extra))[0].detach()
            for fill in (1e8, float('nan'), float('inf')):
                changed = x.clone()
                changed[~valid] = fill
                actual = model((changed, y, extra))[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
            assert len(checked_layers) == 12  # four forwards through all three layers
            for handle in handles:
                handle.remove()

            # Exact valid-observation mean and gradients through the real step.
            changed = x.clone()
            changed[~valid] = float('nan')
            changed.requires_grad_()
            changed_y = y.clone()
            changed_y[~valid] = float('inf')
            logged = {}
            model.log_dict = lambda values, **kwargs: logged.update(values=values, options=kwargs)
            model._process_state = lambda *args, **kwargs: None
            output = model((changed, changed_y, extra))[0]
            reference = (output[valid] - changed_y[valid]).square().mean()
            reference.backward()
            reference_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            changed.grad = None
            loss = model._shared_step((changed, changed_y, extra), 0, 'train')
            torch.testing.assert_close(loss, reference.detach(), rtol=0, atol=0)
            loss.backward()
            assert torch.count_nonzero(changed.grad[~valid]) == 0
            assert torch.isfinite(changed.grad).all() and changed.grad[valid].abs().sum() > 0
            for name, parameter in model.named_parameters():
                if name in reference_grads:
                    assert torch.isfinite(parameter.grad).all(), name
                    torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=0, atol=0)
            assert logged['options']['batch_size'] == int(valid.sum())
            assert torch.count_nonzero(output[~valid]) == 0

            # No behavior change on a fully occupied grid; also exercise bare
            # backbone calls where the optional mask is not supplied.
            encoded, _ = model.encoder(x, **extra)
            all_valid = torch.ones(encoded.shape[:-1], dtype=torch.bool)
            with torch.no_grad():
                plain = model.model(encoded, state=None)[0]
                masked = model.model(encoded, state=None, grid_mask=all_valid)[0]
            torch.testing.assert_close(plain, masked, rtol=0, atol=0)

            checkpoint = {'state_dict': model.state_dict()}
            model.on_save_checkpoint(checkpoint)
            model.on_load_checkpoint(checkpoint)
            for drift in ('padding', 'loss', 'protocol'):
                old = copy.deepcopy(checkpoint)
                if drift == 'padding':
                    old['run_contract']['model_config'].pop('grid_padding')
                elif drift == 'loss':
                    old['run_contract']['training_config']['loss_reduction'] = 'valid-target-mean-v1'
                else:
                    old['cohort']['benchmark_protocol_version'] = 'fair-comparison-v4'
                try:
                    model.on_load_checkpoint(old)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f'accepted stale {drift} checkpoint')
            print(json.dumps({'experiment': exp, 'transposed': transposed,
                              'invariant_to_finite_nan_inf_padding': True,
                              'masked_layers': 3, 'padded_input_gradient': 0,
                              'valid_loss_and_parameter_gradients_match': True,
                              'stale_checkpoint_rejected': True}), flush=True)
    print('ALL S4ND PADDING CHECKS PASSED', flush=True)


if __name__ == '__main__':
    run()
