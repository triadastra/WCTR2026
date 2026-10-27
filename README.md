# Multidimensional Demand Forecasting using State Space Models

**Cheng-Jui Fan · Yi-Wen Hsu · Emma Jiahui Li · Renjie Qiu · Nikolay Aristov · Sicai Zhou · Elenna R. Dugundji**

World Conference on Transport Research (WCTR 2026), Toulouse · Transportation Research Procedia

This repository holds the code, data, every experiment run and the results for benchmarking statistical baselines, gradient-boosted trees, RNNs, Transformers and state-space models. The task is forecasting monthly U.S. state × commodity × flow trade for eight states along the Mississippi River.

**[Findings](docs/FINDINGS.md)** ·
**[Results snapshot](results/2026-10-26/)** ·
**[Figures](figures/)** ·
**[Paper audit](docs/paper_audit/audit_report.md)** ·
**[Reproduce the numbers](#quick-start-reproduce-every-number-on-a-cpu)**

> This public release was assembled from the authors' private research repository with Claude Code (Claude Opus 5.5). Code, configurations, data and run results are exported unchanged from source commit `0c43935`; [SOURCE.json](SOURCE.json) lists the three documentation files edited afterwards. The findings, figures and paper audit were recomputed from the shipped run files.

## Overview

The benchmark asks whether representing the state, commodity and flow axes explicitly helps forecasting. It compares flat encoders (one-hot or learned embeddings) with Axial Self-Attention (ASA) over 1-D backbones, FiLM conditioning, and native multidimensional SSMs (S4ND, Mamba-ND). The comparison runs at three scales:

1. **Part 1, aggregated series:** all trade summed into one national series.
2. **Part 2, window × lag sweep:** input-window and lag-count sensitivity.
3. **Part 3, multidimensional panel:** all 918 retained state × commodity × flow lanes, in flat, 2D, 3D and 4D tensor layouts.

Evaluation reports MSE (the training objective), MAE, RMSE, sMAPE and MASE, with state-clustered significance tests against Moving Average.

## Results at a glance

| | Result |
|---|---|
| Aggregated series | Moving Average is best on every metric (MSE 0.003362). LSTM is next at +8.38% MSE, although 11 of its 25 seeds individually beat Moving Average. |
| Multidimensional panel | **S4 ASA 3D** is best (MSE 0.048726, 6.54% below Moving Average). SSM 2D–4D arms take the top six MSE ranks. |
| Significance | No arm beats Moving Average after Benjamini–Hochberg correction (smallest q = 0.084; S4 ASA 3D raw p = 0.0078, q = 0.152). |
| Loss dependence | The MSE and MAE rankings barely agree (Spearman ρ = 0.24). LSTM embeddings is 1st on MAE and only 12th on MSE. |
| Windows and lags | The best-to-worst spread within each model is 1.03–1.42%. |

**Best arm per family on the multidimensional panel** (paper Table 2). Δ is the improvement over Moving Average; † marks a raw wild-cluster p < 0.05.

| Family | Best variant | MSE rank | MAE rank | MASE | Δ MSE % | Δ MAE % |
|---|---|---:|---:|---:|---:|---:|
| SSM | S4 · ASA 3D | 1 | 15 | 0.862 | +6.54† | +2.16 |
| Attention | Transformer · FiLM | 13 | 2 | 0.841 | +5.02 | +3.15† |
| RNN | LSTM · flat embeddings | 12 | 1 | 0.830 | +5.05† | +4.08† |
| Tree | XGBoost · flat one-hot | 49 | 5 | 0.843 | +0.67 | +2.88† |
| Baseline | Moving Average | 52 | 43 | 0.862 | — | — |

![Improvement over Moving Average for every arm](figures/nd_improvement_heatmap.png)

![MSE rank vs MAE rank](figures/rank_mse_vs_mae.png)

| Aggregated series, per seed | Window × lag sweep |
|---|---|
| ![Aggregate per-seed spread](figures/aggregate_seed_spread.png) | ![Sweep heatmaps](figures/sweep_heatmaps.png) |

[docs/FINDINGS.md](docs/FINDINGS.md) has every table in full: all 12 aggregate models, the MLP-head comparison, all 64 sweep cells, all 57 panel arms with p and q values, and the SHAP attributions. The paper's original figure exports are in [`results/2026-10-26/analysis/paper_figures_2026-10-23/`](results/2026-10-26/analysis/paper_figures_2026-10-23/).

## Dataset

[`data/processed/hs2_monthly_aggregated.csv`](data/processed/) is the pinned monthly release consumed by every model (SHA-256 `585abc64…`). The raw import and export CSVs it was built from are in [`data/imports/`](data/imports/) and [`data/exports/`](data/exports/). [`data/README.md`](data/README.md) and [`data/processed/README.md`](data/processed/README.md) document the aggregation.

| Property | Benchmark specification |
|---|---|
| Source | U.S. Census Bureau, USA Trade Online: state-level trade by HS2 commodity |
| Period | January 2009 – May 2025; 197 months |
| States | Arkansas, Illinois, Iowa, Louisiana, Minnesota, Mississippi, Missouri, Wisconsin |
| Lanes | 1,542 state × commodity × flow combinations; **918 retained** (≤5% zero months over the training period), 93 commodities, 61.7% lattice density |
| Targets | Next-month value and shipping weight. Imports use generic vessel fields, exports use containerized vessel fields (`legacy_mixed`). |
| Features | Value, weight, 12 value lags, 12 weight lags, month sine/cosine (28 per step); 36-month input window |
| Split | Train ≤ 2023-12 · validation 2024 · test January–May 2025 (918 × 5 = 4,590 targets) |
| Normalization | Per-lane min-max, fitted on training months only |

The Part 2 sweep starts its supervised training targets in January 2015. Part 1 sums the eight-state panel into one series before scaling.

## Quick start: reproduce every number on a CPU

The saved runs are in the repository, so the paper's tables and figures can be rebuilt without a GPU. This takes about a minute:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy==1.26.4 pandas==2.3.0 scipy==1.11.4 matplotlib==3.8.4

python tools/verify_against_manifest.py   # 2,500 result files vs. their SHA-256 manifest
python tools/recompute_results.py         # re-derive metrics, means, ranks, MASE, p/q values, SHAP shares
python tools/build_claims_ledger.py       # compare against every number printed in the paper
python tools/build_findings.py            # regenerate docs/FINDINGS.md
python tools/make_figures.py              # regenerate figures/
```

## Models and naming

| Family | Variants | Implementation |
|---|---|---|
| Statistical baselines | Moving average, seasonal naive, random walk | `src/models/baselines.py` |
| Gradient-boosted trees | XGBoost, LightGBM, Random Forest; `onehot`, `embeddings` | `src/models/{xgboost,lightgbm,random_forest}.py` |
| RNNs | LSTM, GRU; `onehot`, `embeddings`, `cross_attention` (ASA 1-D), `film_attention`, `cross_attention_{2d,3d,4d}` | `src/models/{lstm,gru,combo_attention}.py` |
| Attention | Transformer, GPT; same variants (FiLM and ASA 1-D for Transformer) | `src/models/{transformer,gpt}.py` |
| SSMs | S4, S4 (bidirectional), Mamba-2; `onehot`, `embeddings`, `cross_attention_{2d,3d,4d}` | `external/s4/`, `src/models/{s4,mamba}.py` |
| Native ND SSMs | Mamba-ND `{2d,3d,4d}`; S4ND `2d_state` (state × time), `2d` (state × commodity-flow group × time), `3d`, `4d` | `external/Mamba-ND/`, `external/s4/` grid adapter |

Encodings use embeddings of 8 dims for state, 32 for commodity and 2 for flow (the ASA 1-D and FiLM arms use 4 for flow). All neural models use 4 layers, width 128, dropout 0.1 and AdamW with learning rate and weight decay 1e-4. Parameter counts range from 180,326 to 966,018. [docs/MODEL_DIFFS.md](docs/MODEL_DIFFS.md) records where the wrappers differ from the original papers.

## Experiments and runs

| Part | Description | Runs | Location |
|---|---|---:|---|
| 1 | Aggregated series, linear head: 9 learned models × 25 seeds + 3 baselines | 228 | `results/2026-10-26/runs/aggregate-numeric-october/` |
| 1 | Same with an MLP head (128 → 64 → 2): 7 models × 25 seeds + 3 baselines | 178 | `results/2026-10-26/runs/aggregate-mlphead-october/` |
| 2 | Window × lag sweep: 4 models × 16 cells × seeds 26/105/115 | 192 | `results/2026-10-26/runs/sweep/` |
| 3 | Multidimensional panel: 57 arms × seeds 26/105/115 | 171 | `results/2026-10-26/runs/full/` |

Each run folder holds `metrics.json` (metrics, run contract, parameter inventory) and `predictions.csv`. The snapshot also preserves 453 earlier September aggregate runs, which the paper does not use. Model checkpoints are not included.

## Training and evaluation

Training needs **Linux, an NVIDIA GPU with CUDA, and Triton** for the Mamba arms. The paper's runs used an NVIDIA H20.

```bash
python -m pip install -r requirements-benchmark.txt
python scripts/prepare_exp2.py --data-dir . --check-data --check-runtime

bash scripts/run_all_experiments.sh --print-plan
DATA_DIR=. GPUS=0 RESUME=1 bash scripts/run_all_experiments.sh

python scripts/collect_results.py --runs-dir outputs/full
```

Single runs:

```bash
python scripts/train.py --model lstm --variant embeddings --seed 26 --data-dir .
python scripts/inference.py --model lstm --variant embeddings \
    --checkpoint outputs/full/lstm_embeddings_seed26/best.pth --normalized --data-dir .
python scripts/run_baseline.py --model moving_average --normalized --data-dir .
cd external/s4 && python train.py experiment=s4_embeddings train.seed=26 wandb=null loader.num_workers=0
```

The regular `python -m pytest -q` suite runs in the benchmark environment, including synthetic S4/S4ND lifecycles. The statistical protocol is described in [docs/STATISTICAL_PROTOCOL.md](docs/STATISTICAL_PROTOCOL.md), and leakage guarantees in [docs/LEAKAGE_AND_REPORTING_AUDIT.md](docs/LEAKAGE_AND_REPORTING_AUDIT.md).

## Results and verification

- **Recomputation:** all 769 runs behind the paper were re-derived from their predictions. Every stored metric agrees within 1e-14. MASE was rebuilt from the raw dataset, and every significance test with an independent implementation.
- **Paper check:** [docs/paper_audit/](docs/paper_audit/) compares all 424 numbers and claims in the submitted paper with the recomputed evidence: **408 supported**. Every value in Tables 1–2 and Figs. 1 and 4, and every dagger and encoder ratio, reproduces. Corrections are listed below.
- **Training-machine copy:** a fresh copy of the October runs taken from the training machine on 27 October 2026 matched the manifest byte-for-byte (820 of 820 files).
- **Tests:** on GitHub Actions (Ubuntu, Python 3.11, CPU), the suite passes with **769 tests and 3 skipped**. The same workflow re-verifies the manifest and rebuilds the claim ledger from the saved runs.
- **Earlier audit:** the snapshot's own numerical audit is in [`results/2026-10-26/audit/AUDIT.md`](results/2026-10-26/audit/AUDIT.md).

These checks establish that the saved results are internally consistent and correctly reported. They are not a fresh training run.

## Corrections to the submitted paper

| Where | Correction |
|---|---|
| Fig. 3, S4ND "2D" | The printed +4.30 / +1.01 are the 3-axis `s4nd_2d` layout. The state × time layout described in Sec. 4.3 (`s4nd_2d_state`) gives **+4.16 / +1.05**. |
| Fig. 2 | Four annotations are one digit off in the last place (4.947 → 4.946, 4.920 → 4.921, 4.888 → 4.889, 4.906 → 4.907). The minima are unchanged. |
| Sec. 4.1, 4.5 | The sweep trains from January 2015. Import targets are generic vessel trade, not containerized. Hyperparameters, not parameter counts, are shared. |
| Abstract, Sec. 6.2 | "Baselines do not outperform DL" holds for MSE only. The "~6.5% lower MSE" is the best ND arm (median 4.14%). The SHAP dollar figures are mean absolute attributions, not shifts. |

[docs/paper_audit/text_fixes.md](docs/paper_audit/text_fixes.md) has replacement wording for each.

## Repository layout

```text
src/            Models, encoders, data pipeline, training, inference and metrics
config/         Shared protocol, model and variant configurations
scripts/        Data preparation, training, evaluation and reporting commands
tests/          Regression, integrity and leakage tests
external/       Upstream S4, Mamba and Mamba-ND implementations with their licenses
ext/            Statistical evaluation helpers (wild-cluster bootstrap, DM tests)
util/           Paper table and figure builders, diagnostics
data/           Census monthly trade data: raw imports/exports and the processed release
results/        Dated results snapshot: every run's metrics and predictions, manifest, audit
docs/           Findings, protocol and audit documents, and the paper-claim audit
figures/        Figures generated by tools/make_figures.py
tools/          Recompute, verify, ledger, findings and figure scripts for this release
SOURCE.json     Source commit, export scope, edits, and dataset/manifest checksums
```

## Citation

```bibtex
@inproceedings{fan2026multidimensional,
  title     = {Multidimensional Demand Forecasting using State Space Models},
  author    = {Fan, Cheng-Jui and Hsu, Yi-Wen and Li, Emma Jiahui and Qiu, Renjie and
               Aristov, Nikolay and Zhou, Sicai and Dugundji, Elenna R.},
  booktitle = {World Conference on Transport Research (WCTR 2026)},
  year      = {2026},
  address   = {Toulouse, France}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). The follow-up study, *Are Multidimensional Models Worth Their Computational Cost in Demand Forecasting?* (IEEE HPEC 2026), is at [triadastra/HPEC2026](https://github.com/triadastra/HPEC2026). It uses a different dataset and protocol.

## Declaration on the use of generative AI

As declared in the paper, Claude Code and ChatGPT Codex were used for code scaffolding, source research, table typesetting, metrics graphing and proofreading, and Grammarly Pro for grammar checking. The research code was rewritten with assistance from Anthropic Claude Code (Opus 4.7 / 4.8, Fable 5), OpenAI Codex (GPT-5.5) and Moonshot Kimi (k2.6). All experiments and reported results were run and checked by the authors.

This public release, its findings document, figures and paper audit were assembled with Claude Code (Claude Opus 5.5). No experiments, configurations or run results were altered.

## License

The benchmark code is released under the [MIT License](LICENSE). Upstream code under `external/` keeps its original licenses. The trade data comes from the U.S. Census Bureau's USA Trade Online.
