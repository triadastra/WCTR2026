#!/usr/bin/env python3
"""Build the compact workbook from freshly validated experiment artifacts."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_paper_results import _exp1_rows, _exp2_rows, _sweep_rows

OUT = ROOT / "results_compact.xlsx"

def condense(df, keys, smape_col="smape"):
    g = df.groupby(keys, as_index=False).agg(
        MAE=("mae", "mean"),
        MSE=("norm_mse_combined", "mean"),   # the model's actual (combined, normalized) training loss
        RMSE=("rmse", "mean"),
        sMAPE_pct=(smape_col, "mean"),
        n_seeds=("seed", "nunique"),
    )
    for c in ["MAE", "MSE", "RMSE"]:
        g[c] = g[c].round(6)
    g["sMAPE_pct"] = g["sMAPE_pct"].round(2)
    return g.sort_values("MSE").reset_index(drop=True)


def main():
    # The collectors include validated_exp2_artifacts and the corresponding
    # Exp 1/3 completion/source gates. No cached workbook is read.
    exp1_rows, _ = _exp1_rows(ROOT)
    exp1 = condense(pd.DataFrame(exp1_rows), ["model", "variant"])
    exp2 = condense(pd.DataFrame(_exp2_rows(ROOT)), ["model", "variant"])
    sweep_rows = _sweep_rows(ROOT / "outputs/sweep")
    exp3 = condense(pd.DataFrame(sweep_rows), ["model", "variant", "window", "lag_count"])
    readme = pd.DataFrame({"definition": [
        "Source-validated artifacts; complete declared seed rosters and completion receipts.",
        "Primary MSE averages normalized squared errors across valid series-time observations and both targets.",
        "MAE and RMSE are normalized; sMAPE is in physical units, with equal channel weights.",
        "Seed-mean metrics; these values do not establish untouched-holdout performance."]})
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for name, frame in (("README", readme), ("Exp1_Aggregated", exp1),
                            ("Exp2_PerCombo", exp2), ("Exp3_Sweep", exp3)):
            frame.to_excel(writer, sheet_name=name, index=False)
    print(f"Wrote {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
