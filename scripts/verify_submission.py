"""Verify the three deliverables for the MLPS final project.

Checks:
  1. results/ensemble_pred_24h.csv  format / row count / columns / NaN / sign
  2. results/ensemble_pred_48h.csv  same
  3. results/recommended_counties.txt  format = [fips, fips, fips, fips, fips]
  4. Re-runs greedy on the 48h CSV and confirms the chosen counties match the .txt
  5. Compares timestamp+location ordering against submission_template_*.csv
  6. Prints sanity stats: per-horizon peak, total predicted outage-hours, mitigation %

Usage:
    python scripts/verify_submission.py
    python scripts/verify_submission.py --pred-dir results --templates .

Exits with non-zero status if any hard check fails.
"""
from __future__ import annotations
import argparse, ast, sys
from pathlib import Path
import numpy as np
import pandas as pd

CAP = 1000
NUM_GEN = 5
EXPECTED_24H_ROWS = 24 * 83   # 1992
EXPECTED_48H_ROWS = 48 * 83   # 3984
EXPECTED_COLS = ["timestamp", "location", "pred"]


def _hard(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK:   {msg}")


def _load_pred(path: Path, expected_rows: int, label: str, errors: list[str]) -> pd.DataFrame | None:
    print(f"\n[{label}] {path}")
    if not path.exists():
        errors.append(f"missing: {path}"); print(f"  FAIL: file not found"); return None
    df = pd.read_csv(path)
    _hard(list(df.columns) == EXPECTED_COLS, f"columns == {EXPECTED_COLS}, got {list(df.columns)}", errors)
    _hard(len(df) == expected_rows, f"row count == {expected_rows} (got {len(df)})", errors)
    _hard(df["pred"].notna().all(), "no NaN in pred", errors)
    _hard((df["pred"] >= 0).all(), "pred >= 0", errors)
    n_loc = df["location"].nunique()
    _hard(n_loc == 83, f"83 unique counties (got {n_loc})", errors)
    n_ts = df["timestamp"].nunique()
    expected_ts = expected_rows // 83
    _hard(n_ts == expected_ts, f"{expected_ts} unique timestamps (got {n_ts})", errors)
    return df


def _check_template(pred_df: pd.DataFrame, template_path: Path, label: str, errors: list[str]) -> None:
    if not template_path.exists():
        print(f"  WARN: template {template_path} missing, skipping order check"); return
    tpl = pd.read_csv(template_path)
    _hard(len(tpl) == len(pred_df), f"{label} length matches template", errors)
    same_ts  = (tpl["timestamp"].astype(str).values == pred_df["timestamp"].astype(str).values).all()
    same_loc = (tpl["location"].astype(int).values   == pred_df["location"].astype(int).values).all()
    _hard(same_ts,  f"{label} timestamp order matches template", errors)
    _hard(same_loc, f"{label} location order matches template", errors)


def _stats(label: str, df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean").values
    print(f"  {label} stats: peak={pivot.max():.1f}  total outage-hours={pivot.sum():.0f}  "
          f"mean={pivot.mean():.1f}  p99={np.percentile(pivot, 99):.1f}")


def _greedy(P: np.ndarray) -> tuple[list[int], float]:
    L = P.shape[1]; alloc = {}
    def m(a):
        s = 0.0
        for i,k in a.items():
            if k>0: s += np.minimum(P[:,i], k*CAP).sum()
        return s
    for _ in range(NUM_GEN):
        cur = m(alloc); best_gain, best_i = -1.0, None
        for i in range(L):
            tr = dict(alloc); tr[i] = tr.get(i,0)+1
            g = m(tr) - cur
            if g > best_gain: best_gain, best_i = g, i
        alloc[best_i] = alloc.get(best_i, 0) + 1
    seq = []
    for i,k in alloc.items(): seq += [i]*k
    return seq, m(alloc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results")
    ap.add_argument("--templates", default=".")
    ap.add_argument("--p24", default=None, help="override 24h CSV path")
    ap.add_argument("--p48", default=None, help="override 48h CSV path")
    ap.add_argument("--counties", default=None, help="override counties.txt path")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir); tpl_dir = Path(args.templates)
    p24_path = Path(args.p24) if args.p24 else pred_dir / "ensemble_pred_24h.csv"
    p48_path = Path(args.p48) if args.p48 else pred_dir / "ensemble_pred_48h.csv"
    counties_path = Path(args.counties) if args.counties else pred_dir / "recommended_counties.txt"

    errors: list[str] = []

    # 1. CSVs
    df24 = _load_pred(p24_path, EXPECTED_24H_ROWS, "Prediction 24h", errors)
    df48 = _load_pred(p48_path, EXPECTED_48H_ROWS, "Prediction 48h", errors)

    # 2. Template ordering
    if df24 is not None:
        _check_template(df24, tpl_dir / "submission_template_24h.csv", "24h", errors)
    if df48 is not None:
        _check_template(df48, tpl_dir / "submission_template_48h.csv", "48h", errors)

    # 3. Stats
    if df24 is not None: _stats("24h", df24)
    if df48 is not None: _stats("48h", df48)

    # 4. Counties .txt
    print(f"\n[Counties] {counties_path}")
    if not counties_path.exists():
        errors.append(f"missing: {counties_path}"); print("  FAIL: file not found")
    else:
        raw = counties_path.read_text().strip()
        try:
            parsed = ast.literal_eval(raw)
            ok = isinstance(parsed, list) and len(parsed) == NUM_GEN and all(isinstance(x,int) for x in parsed)
        except Exception:
            parsed, ok = None, False
        _hard(ok, f"valid Python list of {NUM_GEN} integers, got: {raw!r}", errors)

        # 5. Re-run greedy on 48h CSV and confirm same picks
        if df48 is not None and parsed is not None:
            piv = df48.pivot_table(index="timestamp", columns="location", values="pred", aggfunc="mean")
            piv = piv.sort_index()
            P = piv.values.astype(np.float64)
            cols = list(piv.columns.astype(int))
            seq_idx, mit = _greedy(P)
            recomputed = [cols[i] for i in seq_idx]
            print(f"  recomputed greedy   : {recomputed}")
            print(f"  declared (from .txt): {parsed}")
            print(f"  predicted total     : {P.sum():.0f}")
            print(f"  greedy mitigation   : {mit:.0f}  ({100*mit/max(P.sum(),1):.1f}% of forecast)")
            _hard(sorted(recomputed) == sorted(parsed),
                  "greedy re-derived from 48h CSV matches declared counties", errors)

    # Summary
    print("\n" + "="*60)
    if errors:
        print(f"VERIFY: {len(errors)} FAILURE(S)")
        for e in errors: print(f"  - {e}")
        return 1
    print("VERIFY: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
