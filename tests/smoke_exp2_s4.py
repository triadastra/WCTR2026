"""Small synthetic S4/S4ND fit, early stopping, checkpoint and evaluation check.

Run from external/s4: python ../../tests/smoke_exp2_s4.py
Requires the installed upstream runtime; creates no benchmark artifacts.
"""
import sys, tempfile, importlib.util, json, hashlib
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
sys.path.insert(0, str(Path.cwd()))
import train
import src.dataloaders.trade_unified as unified
import src.dataloaders.trade_grid as grid
root=Path.cwd().parents[1]
sys.path.append(str(root/'src/utils'))

from s4_test_support import synthetic
unified._canonical_pipeline=synthetic
grid._clean_pipeline=lambda data_dir,input_len: synthetic(data_dir,input_len,2)
spec=importlib.util.spec_from_file_location('evaluation',root/'scripts/s4_eval.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
evaluation._percombo_plots=lambda *a,**kw: None
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as out:
 evaluation.RUNS=Path(out)
 for exp in ['s4_embeddings','s4_onehot','s4_embeddings_bidir','s4_onehot_bidir','s4nd_2d_state','s4nd_2d','s4nd_3d','s4nd_4d']:
  directory=f's4_{exp}_seed26'
  path=Path(out)/directory/'hydra/checkpoints/val/loss.ckpt'
  with initialize_config_dir(config_dir=str(Path.cwd()/'configs'),version_base=None):
   config=compose(config_name='config',overrides=[f'experiment={exp}','train.seed=26','wandb=null','dataset.input_len=3',
      'model.d_model=8','model.n_layers=1','model.layer.d_state=4','model.layer.lr=null','loader.batch_size=2','loader.num_workers=0',
      'train.optimizer_param_grouping=null','train.test=false',
      'trainer.accelerator=cpu','trainer.max_epochs=3','optimizer.lr=0.0',
      'model.dropout=0.0', '+trainer.num_sanity_val_steps=0',
      '+trainer.enable_progress_bar=false',
      'callbacks.rich_progress_bar=null','callbacks.rich_model_summary=null',
      f'callbacks.model_checkpoint.dirpath={path.parent.parent}',
      'callbacks.model_checkpoint.filename=val/loss',
      '+callbacks.early_stopping.monitor=val/loss',
      '+callbacks.early_stopping.mode=min', '+callbacks.early_stopping.patience=1'])
  config=train.utils.train.process_config(config)
  # Exercise the actual Lightning lifecycle, not a hand-made completion record.
  # Zero learning rate and dropout make validation stable so early stopping is
  # deterministic and quick without touching the production recipe or data.
  train.train(config)
  checkpoint=torch.load(path, map_location='cpu', weights_only=False)
  hydra_dir = path.parents[2]
  inventory = json.loads((hydra_dir/'checkpoint_manifest.json').read_text())
  assert {entry['role'] for entry in inventory['files']} == {'best', 'resume'}
  for entry in inventory['files']:
   assert hashlib.sha256((hydra_dir/entry['path']).read_bytes()).hexdigest() == entry['sha256']
  assert json.loads((hydra_dir/'logs/status.json').read_text())['state'] == 'finished'
  epochs = [json.loads(line) for line in (hydra_dir/'logs/epochs.jsonl').read_text().splitlines()]
  assert len(epochs) == 2 and all('val/loss' in row['metrics'] for row in epochs)
  assert list((hydra_dir/'logs/lightning').glob('version_*/metrics.csv'))
  result=evaluation.evaluate(exp,directory,26)
  assert result['n_samples']==55
  assert 'mase' in result['metrics'] and result['run_contract']==checkpoint['run_contract']
  assert result['training_completion']['stop_reason']=='early_stopping'
  if exp.startswith('s4_'):
   direction = exp.endswith('_bidir')
   assert result['model'] == ('s4_bidir' if direction else 's4')
   assert result['architecture']['bidirectional'] is direction
   opposite = exp.removesuffix('_bidir') if direction else exp + '_bidir'
   try:
    evaluation.evaluate(opposite,directory,26)
   except RuntimeError as error:
    assert 'direction' in str(error)
   else:
    raise AssertionError('opposite-direction evaluation was accepted')
  print('SMOKE PASS',exp,flush=True)
