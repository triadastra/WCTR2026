#!/usr/bin/env python
"""Regenerate the Celsia/WCTR2026 model-card index from the local box mirror.

The HF repo's README is a NAVIGABLE INDEX of every checkpoint, generated from
the directory tree at ~/Desktop/Safe/box_mirror/outputs (the complete copy of
the GPU box's outputs; the box itself cannot reach huggingface.co, so all HF
operations run from the Mac). Re-run after the HF repo contents change.
Requires a logged-in `hf` CLI; HfApi picks up the stored token.

Usage:  python util/make_hf_card.py
"""
import collections
import io
import pathlib
import re

ROOT = pathlib.Path.home() / "Desktop/Safe/box_mirror/outputs"
REPO = "Celsia/WCTR2026"


def runs_in(sub):
    d = ROOT / sub
    if not d.exists():
        return []
    out = []
    for r in sorted(d.iterdir()):
        if r.is_dir() and ((r / "best.pth").exists()
                           or list(r.glob("hydra/checkpoints/val/loss.ckpt"))):
            out.append(r.name)
    return out


def build_card():
    lines = []
    agg = runs_in("agg")
    arms = collections.defaultdict(list)
    for name in agg:
        m = re.match(r"(.+)_seed(\d+)$", name)
        if m:
            arms[m.group(1)].append(int(m.group(2)))
    lines.append("## Experiment 1 — aggregated series (`outputs/agg/`)\n")
    lines.append("Path pattern: `outputs/agg/{arm}_seed{S}/best.pth`\n")
    lines.append("| arm | seeds with checkpoints |")
    lines.append("|---|---|")
    for a in sorted(arms):
        s = sorted(arms[a])
        lines.append(f"| `{a}` | {len(s)}: {', '.join(map(str, s))} |")

    lines.append("\n## Experiment 2 — per-combination arms "
                 "(`outputs/full/`, `outputs/fixed/`)\n")
    lines.append("Only these arms have persisted weights (all others are "
                 "exactly reproducible from pinned seeds; predictions and "
                 "metrics for every arm are in the GitHub repo).\n")
    lines.append("| path |")
    lines.append("|---|")
    for sub in ("full", "fixed"):
        for name in runs_in(sub):
            lines.append(f"| `outputs/{sub}/{name}/best.pth` |")

    lines.append("\n## Experiment 3 — window x lag sweep (`outputs/sweep/`)\n")
    lines.append("Path pattern: `outputs/sweep/{model}_{variant}_w{window}_"
                 "l{lags}_seed{S}/best.pth` (S4-family runs keep hydra "
                 "checkpoints at `.../hydra/checkpoints/val/loss.ckpt`). "
                 "`dss_*` runs are the retired DSS model, kept for the "
                 "record.\n")
    pat = re.compile(r"(.+?)_(embeddings|onehot)_w(\d+)_l(\d+)_seed(\d+)$")
    table = collections.defaultdict(list)
    other = []
    for name in runs_in("sweep"):
        m = pat.match(name)
        if m:
            table[(m.group(1), m.group(2))].append(
                (int(m.group(3)), int(m.group(4)), int(m.group(5))))
        else:
            other.append(name)
    lines.append("| model / variant | window x lag cells (seed) |")
    lines.append("|---|---|")
    for (mod, var), cells in sorted(table.items()):
        cs = ", ".join(f"w{w}xl{l}(s{s})" for w, l, s in sorted(cells))
        lines.append(f"| `{mod}_{var}` | {cs} |")
    if other:
        lines.append("\nOther sweep dirs: "
                     + ", ".join(f"`{o}`" for o in other))

    lines.append("\n## Weight-decay ablation (`outputs/wd_ablation/`)\n")
    lines.append("| path |")
    lines.append("|---|")
    for name in runs_in("wd_ablation"):
        lines.append(f"| `outputs/wd_ablation/{name}/best.pth` |")
    lines.append("\n## Historical (`outputs/_buggy_combo_backup/`)\n")
    lines.append("Pre-fix combo-encoder runs (reshape-bug era) — superseded, "
                 "kept for auditability: "
                 + ", ".join(f"`{n}`"
                             for n in runs_in("_buggy_combo_backup")) + "\n")

    head = """---
tags: [time-series-forecasting, state-space-models, transport]
---
# WCTR 2026 — Multidimensional Demand Forecasting: trained models

Checkpoints for *Multidimensional Demand Forecasting using State Space Models*
(WCTR 2026). Code, data pipeline, and full results:
https://github.com/triadastra/Multidimensional-Demand-Forecasting

Every run directory carries its `metrics.json`. Loading a checkpoint:

```python
from huggingface_hub import hf_hub_download
import torch
p = hf_hub_download("Celsia/WCTR2026", "outputs/agg/gru_onehot_seed26/best.pth")
state = torch.load(p, map_location="cpu")
```

Model definitions live in the GitHub repo (`src/models/`, `scripts/train.py`).

<!-- TODO(author): choose a license and confirm this card's wording. -->

"""
    return head + "\n".join(lines) + "\n"


def main():
    from huggingface_hub import HfApi
    card = build_card()
    HfApi().upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md", repo_id=REPO,
        commit_message="Model card: regenerate checkpoint index from tree")
    print(f"[hf-card] uploaded ({card.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
