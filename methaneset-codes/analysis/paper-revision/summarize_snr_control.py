#!/usr/bin/env python
"""
Resume snr_samples_*.csv del control de mascara aleatoria.

Genera por (sensor, banda):
  n, mediana real (IQR), mediana del control (IQR), ratio real/control,
  mediana del ratio pareado real/control, fraccion de muestras real > control,
  p-valor de signo (binomial exacta, dos colas) sobre comparaciones pareadas.

Tambien un diagnostico de contexto superficial: para la banda de mayor SNR
termico (L89 B10/B11) compara el valor medio del target dentro/ fuera de pluma
(en K), y lo mismo para SWIR-2.

Uso: python summarize_snr_control.py   (reads $METHANESET_WORK/snr_control)
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def summarize(df, tag):
    rows = []
    for band_index, g in df.groupby("band_index"):
        g = g.sort_values("sample_id")
        real = g["snr_real"].dropna()
        ctrl = g["snr_ctrl_med"].dropna()
        paired = g.dropna(subset=["snr_real", "snr_ctrl_med"])
        ratio_s = (paired["snr_real"] / paired["snr_ctrl_med"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        q_r = np.percentile(real, [25, 50, 75]) if len(real) else [np.nan] * 3
        q_c = np.percentile(ctrl, [25, 50, 75]) if len(ctrl) else [np.nan] * 3
        n_gt = int((paired["snr_real"] > paired["snr_ctrl_med"]).sum())
        n_lt = int((paired["snr_real"] < paired["snr_ctrl_med"]).sum())
        n_eq = len(paired) - n_gt - n_lt
        n_pair = n_gt + n_lt
        p_sign = (binomtest(n_gt, n_pair, 0.5, alternative="two-sided").pvalue
                  if n_pair > 0 else np.nan)
        try:
            p_wil = (wilcoxon(paired["snr_real"], paired["snr_ctrl_med"]).pvalue
                     if len(paired) >= 10 else np.nan)
        except ValueError:
            p_wil = np.nan
        rows.append({
            "tag": tag,
            "sensor": g["sensor"].iloc[0],
            "band_index": band_index,
            "band": g["band"].iloc[0],
            "wl": g["wl"].iloc[0],
            "n": len(paired),
            "med_real": q_r[1], "q1_real": q_r[0], "q3_real": q_r[2],
            "med_ctrl": q_c[1], "q1_ctrl": q_c[0], "q3_ctrl": q_c[2],
            "ratio": q_r[1] / q_c[1] if q_c[1] else np.nan,
            "med_ratio_paired": float(ratio_s.median()) if len(ratio_s) else np.nan,
            "frac_real_gt_ctrl": n_gt / n_pair if n_pair else np.nan,
            "p_sign": p_sign,
            "p_wilcoxon": p_wil,
        })
    return pd.DataFrame(rows)


def markdown_table(tab):
    lines = []
    for tag, g in tab.groupby("tag", sort=False):
        lines.append(f"\n### {tag}\n")
        lines.append("| banda | wl (nm) | n | SNR real (med [Q1-Q3]) | "
                     "SNR control (med [Q1-Q3]) | ratio | real>ctrl | p (signo) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(
                f"| {r['band']} | {int(r['wl'])} | {int(r['n'])} | "
                f"{r['med_real']:.3f} [{r['q1_real']:.3f}-{r['q3_real']:.3f}] | "
                f"{r['med_ctrl']:.3f} [{r['q1_ctrl']:.3f}-{r['q3_ctrl']:.3f}] | "
                f"{r['ratio']:.2f} | {r['frac_real_gt_ctrl']:.2f} | "
                f"{r['p_sign']:.1e} |")
    return "\n".join(lines)


def context_diagnostic(df, tag):
    """Diferencia media del target pluma-fondo (contexto de superficie)."""
    rows = []
    for band_index, g in df.groupby("band_index"):
        d = (g["target_mean_plume"] - g["target_mean_bg"]).dropna()
        if not len(d):
            continue
        rows.append({
            "tag": tag, "band": g["band"].iloc[0], "wl": g["wl"].iloc[0],
            "n": len(d),
            "med_target_diff": float(d.median()),
            "q1_target_diff": float(np.percentile(d, 25)),
            "q3_target_diff": float(np.percentile(d, 75)),
            "frac_plume_hotter": float((d > 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get(
        "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")),
        "snr_control"))
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    frames, ctxs = [], []
    files = sorted(out.glob("snr_samples_*.csv"))
    for f in files:
        tag = f.stem.replace("snr_samples_", "")
        if args.tags and tag not in args.tags:
            continue
        df = pd.read_csv(f)
        frames.append(summarize(df, tag))
        ctxs.append(context_diagnostic(df, tag))
    tab = pd.concat(frames, ignore_index=True)
    ctx = pd.concat(ctxs, ignore_index=True)
    tab.to_csv(out / "snr_table.csv", index=False)
    ctx.to_csv(out / "target_context.csv", index=False)
    md = markdown_table(tab)
    (out / "snr_table.md").write_text(md)
    print(md)
    print("\n### contexto target (pluma - fondo)\n")
    print(ctx.to_string(index=False))


if __name__ == "__main__":
    main()
