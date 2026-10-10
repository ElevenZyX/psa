"""
"Camino 2" for choosing the clip bound C: leave-one-out over datasets, so the C used on dataset k is chosen
without looking at k (for tofu-forget10 = D_priv: chosen on the 7 public PSA datasets only).

others(k)  = the 7 public PSA datasets if k = tofu-forget10, else the 7 public ones minus k.
             tofu-forget10 never enters others(·). Same rule as test_letter_transfer.py:275.

Step 1  window W_k = the 5 consecutive layers with the highest mean R̄ over others(k), read from measure_rbar.py's
        csv (mean over the (dataset, layer) cells, i.e. every dataset of others(k) weighs the same). Printed with
        the margin to the second-best window; checked against EXPECTED_WINDOWS (AssertionError on mismatch).
Step 2  grid check: mean_norm (csv) of others(k) on the layers of W_k must lie in [min C_GRID, max C_GRID];
        SystemExit otherwise.
Step 4  sweep on the sweep_epsilon.py caches (<diffs-dir>/<dataset>.pt; this script never loads a model, so
        --require-cache is always in force), every dataset on every distinct window of the model:
          PSA        v_l = mean_i clip_C(d_i^l)  + C   * sigma(eps/S, delta/S, 2/n) * z_l
          PSA-prima  v   = mean_i clip_Cg(D_i)   + Cg  * sigma(eps,   delta,   2/n) * z,   Cg = C * sqrt(S)
        clip_C(x) = x * min(1, C/‖x‖), i.e. C * x / max{C, ‖x‖}: the run.py priv_mean convention (normalise by the
        bound, add noise_multiplier * N(0, I), multiply back), via sweep_epsilon.clip_rows / noisy_vectors.
        sigma = Balle-Wang analytic calibration (sweep_epsilon.Accountant), sensitivity 2/n with the real n.
        Fidelity = cos(v_l, mean_i d_i^l) per layer, against the Mean Steering vector (mean of the d_i^l, NOT clipped,
        NOT normalised; run.py's scaled_mean only rescales it, run.py:37-41). The reference is the same for every C.
        'mean' = mean of the S per-layer cosines. Seeds are paired: seed s draws one z in R^{S x h}
        (torch.Generator(s), float64, built exactly as sweep_epsilon.main), shared by both mechanisms, every C,
        every eps, every dataset and every window.
Step 5  F_k(C) = mean over (j in others(k), eps in SELECT_EPS) of the PSA fidelity of j measured in W_k
        (mean over seeds -> mean over layers -> equal weight over datasets and eps). C_k = argmax_C F_k(C).
        Ties (|F - max F| <= TIE_TOL) are NOT resolved: c_selection.csv gets C empty, empate=True and the tied Cs.
        TIE_TOL is not 0 because a mathematically exact tie (paired z, every row of others clipped at both Cs, so
        the privatised direction does not depend on C) only agrees to float rounding.

Checks before the sweep: R̄ recomputed from every cache == csv (cache_alignment.crosscheck_rbar, SystemExit),
sweep_epsilon.check_priv_mean_replication, and sweep_epsilon.check_tight_equals_prima on every (dataset, window).

    python sweep_c.py --selftest                       # synthetic, in memory, writes nothing
    python sweep_c.py --model-tag 1b --check-only      # steps 1-2 only (needs only the R̄ csv)
    python sweep_c.py --model-tag 1b                   # sweep_1b/diffs, resultados_rbar_1b -> sweep_c_1b/
    python sweep_c.py --table sweep_c_1b sweep_c_3b sweep_c_8b   # 3 models x 8 datasets table

Outputs (in --out-dir, which must not exist or be empty): windows.csv, grid_check.csv, sweep_c_results.csv (per seed),
sweep_c_summary.csv (over seeds), clip_stats.csv, c_selection.csv, c_selection_curves.csv, c_selected_fidelity.csv,
resumen.txt.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cache_alignment import DEFAULT_DATASETS, OWN_DATASET, PSA_DATASETS, crosscheck_rbar
from sweep_epsilon import (Accountant, check_priv_mean_replication, check_tight_equals_prima, ci95, clip_rows,
                           cosine, load_cache, noisy_vectors, sigma_analytic)

MODELS = {
    '1b': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_full',
    '3b': 'open-unlearning/tofu_Llama-3.2-3B-Instruct_full',
    '8b': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
}
# (default window, {dataset: exception}) — Diego's expected step-1 result, asserted on the real csv
EXPECTED_WINDOWS = {
    '1b': ((7, 11), {'refusal': (8, 12)}),
    '3b': ((11, 15), {'sycophancy': (10, 14)}),
    '8b': ((12, 16), {'hallucination': (13, 17), 'refusal': (13, 17)}),
}
WINDOW_SIZE = 5
C_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
EPSILONS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
SELECT_EPS = [1, 2, 4, 8]
MECHS = ['psa', 'prima']
TIE_TOL = 1e-9
PRIMA_TEST_TOL = 1e-9


def log(msg: str = '') -> None:
    print(msg, flush=True)


def others(k: str) -> list[str]:
    """The 7 public PSA datasets minus k (all 7 for tofu-forget10). D_priv never enters."""
    return [x for x in PSA_DATASETS if x != k]


def wlabel(w: tuple[int, int]) -> str:
    return f"{w[0]}-{w[1]}"


def wlayers(w: tuple[int, int]) -> list[int]:
    return list(range(w[0], w[1] + 1))


# --------------------------------------------------------------------------------------
# Steps 1-2 (R̄ csv only)
# --------------------------------------------------------------------------------------
def rbar_for_model(csv: pd.DataFrame, model: str, label: str) -> pd.DataFrame:
    sub = csv[csv['model'] == model]
    if sub.empty:
        raise SystemExit(f"{label}: no rows for model {model!r} (models in the csv: {sorted(csv['model'].unique())})")
    missing = [d for d in DEFAULT_DATASETS if d not in set(sub['dataset'])]
    if missing:
        raise SystemExit(f"{label}: datasets {missing} missing for {model!r}")
    layers = sorted(sub['layer'].unique())
    if layers != list(range(len(layers))):
        raise SystemExit(f"{label}: layers are {layers}, expected 0..L-1")
    counts = sub.groupby('dataset')['layer'].nunique()
    dup = sub.duplicated(['dataset', 'layer']).any()
    if dup or (counts[DEFAULT_DATASETS] != len(layers)).any():
        raise SystemExit(f"{label}: not exactly one row per (dataset, layer) for {model!r}")
    return sub


def window_table(rbar: pd.DataFrame, size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Step 1 for every dataset k. rbar: rows (dataset, layer, rbar) of one model, every dataset on every layer."""
    piv = rbar.pivot(index='layer', columns='dataset', values='rbar').sort_index()
    layers = piv.index.to_numpy()
    starts = layers[:len(layers) - size + 1]
    rows = []
    for k in DEFAULT_DATASETS:
        oth = others(k)
        per_layer = piv[oth].mean(axis=1).to_numpy()          # equal weight per dataset of others(k)
        score = np.array([per_layer[i:i + size].mean() for i in range(len(starts))])
        order = np.argsort(-score, kind='stable')
        b, s = order[0], order[1]
        if score[b] == score[s]:
            raise SystemExit(f"step 1, {k}: windows {starts[b]}-{starts[b] + size - 1} and {starts[s]}-{starts[s] + size - 1} "
                             f"have exactly the same mean R̄ ({score[b]}); the rule does not define a winner. Stopping.")
        rows.append({'dataset': k, 'window_start': int(starts[b]), 'window_end': int(starts[b] + size - 1),
                     'ventana': f"{starts[b]}-{starts[b] + size - 1}", 'rbar_mean_others': float(score[b]),
                     'second_window': f"{starts[s]}-{starts[s] + size - 1}", 'rbar_second': float(score[s]),
                     'margin': float(score[b] - score[s]), 'margin_rel': float((score[b] - score[s]) / score[b]),
                     'others': ' '.join(oth)})
    return pd.DataFrame(rows)


def check_expected_windows(wt: pd.DataFrame, tag: str) -> None:
    default, exceptions = EXPECTED_WINDOWS[tag]
    bad = []
    for _, r in wt.iterrows():
        exp = exceptions.get(r['dataset'], default)
        if (r['window_start'], r['window_end']) != exp:
            bad.append(f"{r['dataset']}: got {r['ventana']}, expected {wlabel(exp)}")
    if bad:
        raise AssertionError(f"step 1 ({tag}): windows differ from the expected ones: " + "; ".join(bad))
    log(f"  assert OK: windows match the expected ones for {tag} (default {wlabel(default)}, "
        f"exceptions {', '.join(f'{d} {wlabel(w)}' for d, w in exceptions.items())})")


def grid_check(rbar: pd.DataFrame, wt: pd.DataFrame, grid: list[float]) -> pd.DataFrame:
    """Step 2: mean_norm of others(k) on the layers of W_k, per k."""
    rows = []
    for _, r in wt.iterrows():
        sub = rbar[rbar['dataset'].isin(others(r['dataset'])) & rbar['layer'].between(r['window_start'], r['window_end'])]
        assert len(sub) == len(others(r['dataset'])) * WINDOW_SIZE
        lo, hi = sub.loc[sub['mean_norm'].idxmin()], sub.loc[sub['mean_norm'].idxmax()]
        rows.append({'dataset': r['dataset'], 'ventana': r['ventana'],
                     'mean_norm_min': float(lo['mean_norm']), 'argmin': f"{lo['dataset']} L{int(lo['layer'])}",
                     'mean_norm_max': float(hi['mean_norm']), 'argmax': f"{hi['dataset']} L{int(hi['layer'])}",
                     'inside_grid': bool(lo['mean_norm'] >= min(grid) and hi['mean_norm'] <= max(grid))})
    return pd.DataFrame(rows)


def steps_1_2(csv: pd.DataFrame, model: str, tag: str | None, grid: list[float], label: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rbar = rbar_for_model(csv, model, label)
    log("=" * 100)
    log(f"STEP 1 — windows (leave-one-out, {WINDOW_SIZE} consecutive layers, max mean R̄ over others(k)) — {model}")
    wt = window_table(rbar)
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        log(wt[['dataset', 'ventana', 'rbar_mean_others', 'second_window', 'rbar_second', 'margin', 'margin_rel']]
            .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if tag is not None:
        check_expected_windows(wt, tag)
    log("=" * 100)
    log(f"STEP 2 — grid check: mean_norm of others(k) on the layers of W_k inside [{min(grid):g}, {max(grid):g}]")
    gc = grid_check(rbar, wt, grid)
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        log(gc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not gc['inside_grid'].all():
        raise SystemExit(f"STEP 2 FAILED: mean_norm outside [{min(grid):g}, {max(grid):g}] for "
                         f"{gc.loc[~gc['inside_grid'], 'dataset'].tolist()}. Stopping before the sweep.")
    log(f"  grid check OK for all {len(gc)} datasets")
    return wt, gc


# --------------------------------------------------------------------------------------
# Step 4: sweep (in memory)
# --------------------------------------------------------------------------------------
def paired_draws(seeds: list[int], S: int, h: int) -> torch.Tensor:
    """Same construction as sweep_epsilon.main: one N(0, I_{S x h}) per seed, float64, CPU."""
    return torch.stack([torch.randn(S, h, generator=torch.Generator(device='cpu').manual_seed(s), dtype=torch.float64)
                        for s in seeds])


def sweep_one(d: torch.Tensor, Z: torch.Tensor, acc: Accountant, grid: list[float], epsilons: list[float]
              ) -> tuple[np.ndarray, list[dict]]:
    """d: (n, S, h) float64 diffs of one dataset on one window. Returns
    cos[c, e, mech, seed, layer] against the unclipped mean, and per-(C, mech, layer) clip stats."""
    n, S, h = d.shape
    K = Z.shape[0]
    ref = d.mean(dim=0)                                     # Mean Steering vector, the same for every C
    D = d.reshape(n, S * h)
    cos = np.empty((len(grid), len(epsilons), len(MECHS), K, S))
    stats = []
    for c, C in enumerate(grid):
        Cg = C * math.sqrt(S)
        m_psa = torch.empty(S, h, dtype=torch.float64)
        frac_psa = np.empty(S)
        for j in range(S):
            clipped, frac_psa[j] = clip_rows(d[:, j, :], C)
            m_psa[j] = clipped.mean(dim=0)
        clipped_cat, frac_prima = clip_rows(D, Cg)
        m = {'psa': m_psa, 'prima': clipped_cat.mean(dim=0).reshape(S, h)}
        bound = {'psa': np.full(S, float(C)), 'prima': np.full(S, Cg)}
        frac = {'psa': frac_psa, 'prima': np.full(S, frac_prima)}
        for mech in MECHS:
            cn = cosine(m[mech], ref).numpy()
            for j in range(S):
                stats.append({'mechanism': mech, 'C': C, 'Cg': Cg, 'layer_pos': j, 'clip_bound': bound[mech][j],
                              'frac_clipped': frac[mech][j], 'cos_noiseless_vs_mean_steering': float(cn[j])})
        for e, eps in enumerate(epsilons):
            for t, mech in enumerate(MECHS):
                V, _ = noisy_vectors(m[mech], bound[mech], acc.sigma(mech, eps), Z)   # (K, S, h)
                cos[c, e, t] = cosine(V, ref[None]).numpy()
    return cos, stats


def run_sweep(caches: dict, windows: list[tuple[int, int]], grid: list[float], epsilons: list[float],
              seeds: list[int], delta: float, calibration: str, model: str, real_checks: bool
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """caches: {dataset: cache dict as written by sweep_epsilon.extract_diffs}. Every dataset x every window."""
    S = WINDOW_SIZE
    h = next(iter(caches.values()))['hidden']
    Z = paired_draws(seeds, S, h)
    log(f"noise draws Z: {tuple(Z.shape)} float64, seeds {seeds[0]}..{seeds[-1]} (paired across mechanisms, C, eps, datasets, windows)")
    res_rows, clip_out = [], []
    for name in DEFAULT_DATASETS:
        cache = caches[name]
        assert cache['hidden'] == h, f"{name}: hidden {cache['hidden']} != {h}"
        n = cache['n']
        acc = Accountant(calibration, n, delta, S)
        if real_checks:
            check_priv_mean_replication(cache['diffs'][windows[0][0]], C=20.0, sigma=0.02)
        for w in windows:
            t0 = time.time()
            d = torch.stack([cache['diffs'][l] for l in wlayers(w)], dim=1).double()   # (n, S, h)
            if real_checks:
                check_tight_equals_prima(d, Z, epsilons, acc, f"{name} {wlabel(w)}")
            cos, stats = sweep_one(d, Z, acc, grid, epsilons)
            for st in stats:
                clip_out.append({'model': model, 'ventana': wlabel(w), 'dataset': name, 'n': n,
                                 'layer': wlayers(w)[st.pop('layer_pos')], **st})
            for c, C in enumerate(grid):
                for e, eps in enumerate(epsilons):
                    for t, mech in enumerate(MECHS):
                        base = {'model': model, 'ventana': wlabel(w), 'dataset': name, 'n': n, 'mechanism': mech,
                                'C': C, 'Cg': C * math.sqrt(S), 'epsilon': eps}
                        for k_, seed in enumerate(seeds):
                            for j, l in enumerate(wlayers(w)):
                                res_rows.append({**base, 'seed': seed, 'layer': l, 'cosine': cos[c, e, t, k_, j]})
                            res_rows.append({**base, 'seed': seed, 'layer': 'mean', 'cosine': cos[c, e, t, k_].mean()})
            log(f"  [{name} {wlabel(w)}] n={n} swept {len(grid)} C x {len(epsilons)} eps x {len(MECHS)} mechanisms x "
                f"{len(seeds)} seeds in {time.time() - t0:.1f}s")
    return pd.DataFrame(res_rows), pd.DataFrame(clip_out)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ['model', 'ventana', 'dataset', 'n', 'mechanism', 'C', 'Cg', 'epsilon', 'layer']
    g = results.groupby(keys, sort=False)['cosine']
    out = g.agg(n_seeds='size', cos_mean='mean', cos_std=lambda x: x.std(ddof=1)).reset_index()
    from scipy.stats import t as t_dist
    half = t_dist.ppf(0.975, out['n_seeds'] - 1) * out['cos_std'] / np.sqrt(out['n_seeds'])
    out['cos_ci95_lo'], out['cos_ci95_hi'] = out['cos_mean'] - half, out['cos_mean'] + half
    return out


# --------------------------------------------------------------------------------------
# Step 5: leave-one-out selection
# --------------------------------------------------------------------------------------
def selection_curves(summary: pd.DataFrame, window_of: dict[str, str], grid: list[float], select_eps: list[float]
                     ) -> pd.DataFrame:
    """F_k(C) for every k. Uses ONLY rows of others(k) measured in W_k (asserted)."""
    psa = summary[(summary['mechanism'] == 'psa') & (summary['layer'] == 'mean') & summary['epsilon'].isin(select_eps)]
    rows = []
    for k in DEFAULT_DATASETS:
        oth = others(k)
        sub = psa[(psa['ventana'] == window_of[k]) & psa['dataset'].isin(oth)]
        assert k not in set(sub['dataset']), f"selection for {k} would use {k}"
        assert set(sub['ventana']) == {window_of[k]}
        cells = sub.groupby(['C', 'dataset', 'epsilon']).size()
        if len(sub) != len(oth) * len(grid) * len(select_eps) or (cells != 1).any():
            raise SystemExit(f"selection for {k}: expected one row per (C, dataset in others, eps in {select_eps}) in window "
                             f"{window_of[k]}, got {len(sub)} rows")
        F = sub.groupby('C')['cos_mean'].mean()             # equal weight over (dataset, eps)
        for C in grid:
            rows.append({'dataset': k, 'ventana': window_of[k], 'C': C, 'F': float(F.loc[C]),
                         'n_cells': int((sub['C'] == C).sum())})
    return pd.DataFrame(rows)


def select_c(curves: pd.DataFrame, model: str, S: int = WINDOW_SIZE, tie_tol: float = TIE_TOL) -> pd.DataFrame:
    rows = []
    for k in DEFAULT_DATASETS:
        cu = curves[curves['dataset'] == k].set_index('C')['F']
        best = cu.max()
        tied = sorted(cu.index[cu >= best - tie_tol].tolist())
        rest = cu.drop(index=tied)
        tie = len(tied) > 1
        rows.append({'model': model, 'dataset': k, 'ventana': curves.loc[curves['dataset'] == k, 'ventana'].iloc[0],
                     'C': float('nan') if tie else tied[0], 'Cg': float('nan') if tie else tied[0] * math.sqrt(S),
                     'empate': tie, 'C_empatados': ' '.join(f"{c:g}" for c in tied) if tie else '',
                     'F_best': float(best), 'C_second': float(rest.idxmax()) if len(rest) else float('nan'),
                     'F_second': float(rest.max()) if len(rest) else float('nan'),
                     'margin_F': float(best - rest.max()) if len(rest) else float('nan')})
    return pd.DataFrame(rows)


def selected_fidelity(results: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    """PSA and PSA-prima on k itself, in W_k, at C_k (every tied C if there is a tie): mean over seeds of the layer-mean
    cosine and the paired (same z) difference prima - PSA with a t 95% interval."""
    r = results[results['layer'] == 'mean']
    rows = []
    for _, s in sel.iterrows():
        Cs = [float(c) for c in s['C_empatados'].split()] if s['empate'] else [s['C']]
        for C in Cs:
            sub = r[(r['dataset'] == s['dataset']) & (r['ventana'] == s['ventana']) & (r['C'] == C)]
            wide = sub.pivot_table(index=['epsilon', 'seed'], columns='mechanism', values='cosine').reset_index()
            for eps, we in wide.groupby('epsilon', sort=True):
                m, sd, hw = ci95((we['prima'] - we['psa']).to_numpy())
                rows.append({'model': s['model'], 'dataset': s['dataset'], 'ventana': s['ventana'], 'C': C,
                             'Cg': C * math.sqrt(WINDOW_SIZE), 'empate': s['empate'], 'epsilon': eps,
                             'n_seeds': len(we), 'cos_psa': we['psa'].mean(), 'cos_prima': we['prima'].mean(),
                             'diff_prima_psa': m, 'diff_ci95_lo': m - hw, 'diff_ci95_hi': m + hw,
                             'paired_separated': bool(m - hw > 0 or m + hw < 0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Summary text
# --------------------------------------------------------------------------------------
def write_resumen(args, wt: pd.DataFrame, gc: pd.DataFrame, sel: pd.DataFrame, curves: pd.DataFrame,
                  clip: pd.DataFrame, fid: pd.DataFrame, n_by: dict[str, int], grid: list[float]) -> str:
    L = []
    P = L.append
    P("=" * 100)
    P(f"RESUMEN — selección de C dejando uno fuera — {args.model}")
    P("=" * 100)
    P(f"grilla C {grid} | Cg = C·√{WINDOW_SIZE} | ε {args.epsilons} | δ total {args.delta:g} | calibración {args.calibration} | "
      f"{args.n_seeds} semillas apareadas | selección sobre ε ∈ {args.select_eps}")
    P("PSA: recorte por capa a C, un gaussiano por capa a (ε/5, δ/5). PSA-prima: recorte del concatenado a Cg, un gaussiano a (ε, δ).")
    P("Fidelidad = coseno contra el vector de Mean Steering (media de los d sin recortar ni normalizar), por capa y media sobre las 5 capas.")
    small = {d: n for d, n in n_by.items() if n < args.n_samples}
    if small:
        P(f"AVISO: datasets con n < {args.n_samples}: {small}; su σ usa su n real (más ruido al mismo ε) y entran a 'otros' con el mismo peso.")
    P("")
    P("PASO 1 — ventanas (margen = R̄ medio de la mejor ventana − la segunda, sobre 'otros'):")
    for _, r in wt.iterrows():
        P(f"  {r['dataset']:24s} {r['ventana']:>6s}  R̄ {r['rbar_mean_others']:.5f} | 2ª {r['second_window']:>6s} "
          f"{r['rbar_second']:.5f} | margen {r['margin']:.2e} ({100 * r['margin_rel']:.2f}%)")
    P("")
    P(f"PASO 2 — mean_norm de 'otros' en la ventana de k (debe caer en [{min(grid):g}, {max(grid):g}]):")
    for _, r in gc.iterrows():
        P(f"  {r['dataset']:24s} {r['ventana']:>6s}  mín {r['mean_norm_min']:.3f} ({r['argmin']}) | máx {r['mean_norm_max']:.3f} ({r['argmax']})")
    P("")
    P("PASO 5 — C_k = argmax_C F_k(C), F = fidelidad de PSA media sobre 'otros' (medidos en la ventana de k) y ε ∈ "
      f"{args.select_eps}:")
    psa_clip = clip[clip['mechanism'] == 'psa'].groupby(['ventana', 'dataset', 'C'])['frac_clipped'].min()
    for _, s in sel.iterrows():
        k, w = s['dataset'], s['ventana']
        if s['empate']:
            tied = [float(c) for c in s['C_empatados'].split()]
            all_clipped = all(psa_clip.loc[(w, j, c)] >= 1.0 for j in others(k) for c in tied)
            P(f"  {k:24s} {w:>6s}  EMPATE entre C ∈ {{{s['C_empatados']}}} (F = {s['F_best']:.6f}; no se elige)"
              + (" — empate matemático: con esos C se recorta el 100% de las filas de 'otros' en todas las capas, "
                 "la dirección privatizada no depende de C" if all_clipped else ""))
            continue
        edge = " | AVISO: C en el borde de la grilla, el óptimo puede estar fuera" if s['C'] in (min(grid), max(grid)) else ""
        P(f"  {k:24s} {w:>6s}  C={s['C']:g}  Cg={s['Cg']:.3f}  F={s['F_best']:.5f} | 2º C={s['C_second']:g} "
          f"F={s['F_second']:.5f} (margen {s['margin_F']:.2e}){edge}")
    n_tie = int(sel['empate'].sum())
    P(f"  => {n_tie} empate(s)" + (": revisar c_selection.csv (columna C_empatados)" if n_tie else ""))
    P("  Curvas F_k(C) completas en c_selection_curves.csv.")
    P("")
    P("PSA vs PSA-prima en el propio k, ventana de k, C = C_k (coseno medio sobre capas; Δ = prima − PSA apareado, IC 95% t):")
    for (k, C), sub in fid.groupby(['dataset', 'C'], sort=False):
        parts = [f"ε={r['epsilon']:g}: {r['cos_psa']:.3f}/{r['cos_prima']:.3f} Δ{r['diff_prima_psa']:+.3f}"
                 f"{'*' if r['paired_separated'] else ''}" for _, r in sub.iterrows()]
        P(f"  {k:24s} C={C:g}  " + " | ".join(parts))
    P("  (PSA/prima; * = IC apareado excluye 0)")
    text = "\n".join(L)
    log(text)
    return text


# --------------------------------------------------------------------------------------
# Real run
# --------------------------------------------------------------------------------------
def load_caches(diffs_dir: Path, model: str, n_samples: int, seed: int, layers: list[int], csv: pd.DataFrame,
                csv_label: str) -> dict:
    caches = {}
    for name in DEFAULT_DATASETS:
        path = diffs_dir / f"{name}.pt"
        if not path.is_file():
            raise SystemExit(f"no cache {path} (this script never loads the model; run sweep_epsilon.py --extract-only first)")
        cache = load_cache(path, model, n_samples, seed, layers)       # model / seed / read_token_index / n / layers
        log(f"[{name}] cache {path} | n={cache['n']} of {cache['total']} | hidden {cache['hidden']} | {len(cache['layers'])} layers")
        crosscheck_rbar(cache, csv, csv_label)                          # strict: SystemExit on mismatch
        cache['diffs'] = {l: cache['diffs'][l] for l in layers}        # keep only the window layers in memory
        caches[name] = cache
    return caches


def run(args) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} exists and is not empty; existing results are never overwritten. Use a new folder.")
    csv = pd.read_csv(args.rbar_csv)
    grid = [float(c) for c in args.c_grid]
    wt, gc = steps_1_2(csv, args.model, args.model_tag, grid, args.rbar_csv)
    if args.check_only:
        log("--check-only: steps 1-2 done, nothing written.")
        return
    windows = sorted({(int(r['window_start']), int(r['window_end'])) for _, r in wt.iterrows()})
    window_of = dict(zip(wt['dataset'], wt['ventana']))
    layers = sorted({l for w in windows for l in wlayers(w)})
    log("=" * 100)
    log(f"STEP 4 — sweep on {args.diffs_dir}, windows {[wlabel(w) for w in windows]} (every dataset on every window)")
    caches = load_caches(Path(args.diffs_dir), args.model, args.n_samples, args.seed, layers, csv, args.rbar_csv)
    seeds = list(range(args.n_seeds))
    results, clip = run_sweep(caches, windows, grid, sorted(args.epsilons), seeds, args.delta, args.calibration,
                              args.model, real_checks=True)
    summary = summarize(results)
    curves = selection_curves(summary, window_of, grid, args.select_eps)
    sel = select_c(curves, args.model)
    fid = selected_fidelity(results, sel)

    out_dir.mkdir(parents=True, exist_ok=True)
    wt.to_csv(out_dir / 'windows.csv', index=False)
    gc.to_csv(out_dir / 'grid_check.csv', index=False)
    results.to_csv(out_dir / 'sweep_c_results.csv', index=False)
    summary.to_csv(out_dir / 'sweep_c_summary.csv', index=False)
    clip.to_csv(out_dir / 'clip_stats.csv', index=False)
    curves.to_csv(out_dir / 'c_selection_curves.csv', index=False)
    sel[['model', 'dataset', 'ventana', 'C', 'Cg', 'empate', 'C_empatados']].to_csv(out_dir / 'c_selection.csv', index=False)
    fid.to_csv(out_dir / 'c_selected_fidelity.csv', index=False)
    log("=" * 100)
    log("STEP 5 — c_selection.csv")
    log(sel[['model', 'dataset', 'ventana', 'C', 'Cg', 'empate', 'C_empatados']].to_string(index=False))
    text = write_resumen(args, wt, gc, sel, curves, clip, fid, {d: c['n'] for d, c in caches.items()}, grid)
    (out_dir / 'resumen.txt').write_text(text + "\n", encoding='utf-8')
    log(f"\nwrote windows.csv, grid_check.csv, sweep_c_results.csv, sweep_c_summary.csv, clip_stats.csv, c_selection.csv, "
        f"c_selection_curves.csv, c_selected_fidelity.csv, resumen.txt in {out_dir}")


def table(dirs: list[str]) -> None:
    """Models x datasets table from several c_selection.csv."""
    frames = []
    for d in dirs:
        p = Path(d) / 'c_selection.csv'
        if not p.is_file():
            raise SystemExit(f"{p} not found")
        frames.append(pd.read_csv(p, keep_default_na=False, na_values=['']))
    df = pd.concat(frames, ignore_index=True)
    df['cell'] = [f"EMPATE {{{r['C_empatados']}}} [{r['ventana']}]" if str(r['empate']) == 'True'
                  else f"C={r['C']:g} Cg={r['Cg']:.2f} [{r['ventana']}]" for _, r in df.iterrows()]
    tab = df.pivot(index='dataset', columns='model', values='cell').reindex(DEFAULT_DATASETS)
    tab.columns = [Path(c).name for c in tab.columns]
    with pd.option_context('display.width', 300, 'display.max_columns', None, 'display.max_colwidth', 60):
        log(tab.to_string())


# --------------------------------------------------------------------------------------
# Self-test (synthetic, in memory, no files)
# --------------------------------------------------------------------------------------
ST_H, ST_LAYERS = 64, 9
ST_WINDOWS = {'default': (2, 6), 'refusal': (3, 7)}      # two windows, as in the real models
ST_R_PUBLIC, ST_R_ODD = 2.5, 14.0                       # norm scale of the public datasets / of the odd ones
ST_ODD = ('sycophancy', OWN_DATASET)                     # one public dataset and D_priv with a very different optimum
ST_N = {'coordinate-other-ais': 360, 'corrigible-neutral-HHH': 290}


def st_diffs(rng: np.random.Generator, n: int, R: float, L: int, h: int) -> dict[int, torch.Tensor]:
    """Two equal groups: norms ~2R (direction u_A) and ~R/2 (direction u_B), lognormal spread, small direction noise.
    Clipping below ~2R pulls the mean towards u_B (bias), a large C keeps the direction but adds noise ∝ C:
    the optimum C scales with R."""
    out = {}
    for l in range(L):
        uA, uB = np.linalg.qr(rng.standard_normal((h, 2)))[0].T
        group = rng.random(n) < 0.5
        dirs = np.where(group[:, None], uA, uB) + 0.25 * rng.standard_normal((n, h)) / math.sqrt(h)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(group, 2.0 * R, 0.5 * R) * np.exp(0.15 * rng.standard_normal(n))
        out[l] = torch.from_numpy((dirs * norms[:, None]).astype(np.float32))
    return out


def st_caches(seed: int) -> dict:
    rng = np.random.default_rng([seed, 7])
    caches = {}
    for name in DEFAULT_DATASETS:
        n = ST_N.get(name, 400)
        R = ST_R_ODD if name in ST_ODD else ST_R_PUBLIC
        caches[name] = {'model': 'synthetic/selftest', 'dataset': name, 'n': n, 'total': n, 'seed': 42,
                        'read_token_index': -2, 'hidden': ST_H, 'layers': list(range(ST_LAYERS)),
                        'diffs': st_diffs(rng, n, R, ST_LAYERS, ST_H)}
    return caches


def own_optimum(summary: pd.DataFrame, k: str, w: str, select_eps: list[float]) -> float:
    s = summary[(summary['dataset'] == k) & (summary['ventana'] == w) & (summary['mechanism'] == 'psa')
                & (summary['layer'] == 'mean') & summary['epsilon'].isin(select_eps)]
    return float(s.groupby('C')['cos_mean'].mean().idxmax())


def brute_force_selection(results: pd.DataFrame, k: str, w: str, grid: list[float], select_eps: list[float]) -> float:
    """Independent recomputation of C_k from the per-seed rows: loops, no groupby, seeds -> layers -> (dataset, eps)."""
    r = results[(results['ventana'] == w) & (results['mechanism'] == 'psa') & results['epsilon'].isin(select_eps)
                & (results['layer'] != 'mean')]
    best, best_c = -np.inf, None
    for C in grid:
        vals = []
        for j in others(k):
            for eps in select_eps:
                sub = r[(r['dataset'] == j) & (r['C'] == C) & (r['epsilon'] == eps)]
                assert len(sub) == WINDOW_SIZE * results['seed'].nunique()
                per_seed = sub.groupby('seed')['cosine'].mean()          # mean over the 5 layers
                vals.append(per_seed.mean())
        F = float(np.mean(vals))
        if F > best:
            best, best_c = F, C
    return best_c


def selftest(args) -> None:
    t0 = time.time()
    grid = [float(c) for c in C_GRID]
    eps_grid = sorted(args.epsilons)
    log("#" * 100)
    log("SELFTEST A — step 1 windows: leave-one-out never reads R̄ of the evaluated dataset")
    rng = np.random.default_rng(0)
    rows = []
    for k_, name in enumerate(DEFAULT_DATASETS):
        prof = np.exp(-0.5 * ((np.arange(12) - 5.0) / 2.0) ** 2) * (0.3 + 0.01 * k_) + 0.002 * rng.random(12)
        if name == 'refusal':
            prof = prof + 3.0 * np.exp(-0.5 * ((np.arange(12) - 9.0) / 1.0) ** 2)   # pulls everyone else's window up
        for l in range(12):
            rows.append({'model': 'synthetic', 'dataset': name, 'layer': l, 'rbar': prof[l], 'norm_mean': prof[l] * 5,
                         'mean_norm': 5.0 + 0.1 * l, 'n': 400})
    rb = pd.DataFrame(rows)
    wt = window_table(rb)
    log(wt[['dataset', 'ventana', 'second_window', 'margin']].to_string(index=False))
    # independent brute force
    for k in DEFAULT_DATASETS:
        sc = {s: np.mean([rb[(rb['dataset'] == j) & (rb['layer'] == l)]['rbar'].iloc[0] for j in others(k) for l in range(s, s + 5)])
              for s in range(12 - 4)}
        bs = max(sc, key=sc.get)
        assert wt.set_index('dataset').loc[k, 'window_start'] == bs, (k, bs)
    w = wt.set_index('dataset')['ventana']
    assert w['refusal'] != w['sycophancy'], "the synthetic R̄ must give refusal a different window"
    for k in DEFAULT_DATASETS:   # corrupting k's own R̄ must not move W_k
        bad = rb.copy()
        bad.loc[bad['dataset'] == k, 'rbar'] = np.where(bad.loc[bad['dataset'] == k, 'layer'] < 3, 50.0, 0.0)
        assert window_table(bad).set_index('dataset').loc[k, 'ventana'] == w[k], k
    log(f"  OK: windows match a brute-force recomputation; refusal {w['refusal']} vs the rest {w['sycophancy']}; "
        f"corrupting R̄ of k never moves W_k")
    try:
        check_expected_windows(wt.assign(window_start=wt['window_start'] + 1), '1b')
    except AssertionError as e:
        log(f"  (expected) the expected-windows assert fires on a mismatch: {str(e)[:80]}...")
    else:
        raise AssertionError("check_expected_windows accepted wrong windows")
    try:
        grid_out = rb.copy()
        grid_out.loc[(grid_out['dataset'] == 'hallucination') & (grid_out['layer'] == 4), 'mean_norm'] = 25.0
        steps_1_2(grid_out, 'synthetic', None, grid, 'synthetic csv')
    except SystemExit as e:
        log(f"  (expected) step 2 stops when a mean_norm leaves the grid: {str(e)[:90]}...")
    else:
        raise AssertionError("step 2 accepted a mean_norm outside the grid")

    log("#" * 100)
    log("SELFTEST B — PSA-prima without clipping == one Gaussian on the per-layer normalised vectors")
    n, S, h, C = 400, WINDOW_SIZE, ST_H, 1000.0
    d = torch.from_numpy(np.random.default_rng(1).standard_normal((n, S, h)) * 0.5)      # norms ~4 << C
    assert float(d.norm(dim=2).max()) < C and float(d.reshape(n, -1).norm(dim=1).max()) < C * math.sqrt(S)
    seeds = list(range(args.n_seeds))
    Z = paired_draws(seeds, S, h)
    acc = Accountant('analytic', n, args.delta, S)
    cos, stats = sweep_one(d, Z, acc, [C], eps_grid)
    assert all(s['frac_clipped'] == 0.0 for s in stats), "C was supposed to clip nothing"
    assert all(abs(s['cos_noiseless_vs_mean_steering'] - 1.0) < 1e-12 for s in stats), \
        "without clipping the noiseless vectors must coincide in direction with the Mean Steering reference"
    ref = d.mean(dim=0)
    x = d / C                                                      # per layer: d / max{C, ‖d‖} = d / C
    mu = x.reshape(n, -1).mean(dim=0)                              # concatenated, norm of each row <= sqrt(S)
    worst, worst_std = 0.0, 0.0
    for e, eps in enumerate(eps_grid):
        sig = sigma_analytic(eps, args.delta, math.sqrt(S) * 2.0 / n)   # ONE mechanism, sensitivity sqrt(S) * 2/n
        V = C * (mu[None] + sig * Z.reshape(len(seeds), -1))           # back to original units
        direct = cosine(V.reshape(len(seeds), S, h), ref[None]).numpy()
        worst = max(worst, float(np.abs(direct - cos[0, e, MECHS.index('prima')]).max()))
        worst_std = max(worst_std, abs(C * math.sqrt(S) * acc.sigma_prima(eps) - C * sig) / (C * sig))
    log(f"  max |cos(prima, Cg=C√5) − cos(single Gaussian on d/C)| = {worst:.2e} (tol {PRIMA_TEST_TOL:g}); "
        f"max relative gap of the noise std Cg·σ(ε,δ,2/n) vs C·σ(ε,δ,√5·2/n) = {worst_std:.2e}")
    assert worst < PRIMA_TEST_TOL and worst_std < 1e-8
    check_tight_equals_prima(d, Z, eps_grid, acc, 'synthetic (sweep_epsilon test reused)')

    log("#" * 100)
    log(f"SELFTEST C — leave-one-out selection of C ({', '.join(ST_ODD)} have norm scale {ST_R_ODD} vs {ST_R_PUBLIC})")
    caches = st_caches(0)
    window_of = {k: wlabel(ST_WINDOWS.get(k, ST_WINDOWS['default'])) for k in DEFAULT_DATASETS}
    windows = sorted(set(ST_WINDOWS.values()))
    results, clip = run_sweep(caches, windows, grid, eps_grid, seeds, args.delta, 'analytic', 'synthetic', real_checks=False)
    summary = summarize(results)
    curves = selection_curves(summary, window_of, grid, SELECT_EPS)
    sel = select_c(curves, 'synthetic').set_index('dataset')
    log(sel[['ventana', 'C', 'Cg', 'empate', 'C_empatados', 'F_best', 'C_second', 'margin_F']].to_string())
    own = {k: own_optimum(summary, k, window_of[k], SELECT_EPS) for k in DEFAULT_DATASETS}
    log(f"  own (in-sample) optimum per dataset: {own}")
    assert not sel['empate'].any(), "unexpected tie in the synthetic selection"
    for k in ST_ODD:
        assert own[k] >= 12 and sel.loc[k, 'C'] <= 5, (k, own[k], sel.loc[k, 'C'])
    for k in DEFAULT_DATASETS:
        assert sel.loc[k, 'C'] == brute_force_selection(results, k, window_of[k], grid, SELECT_EPS), k
    log("  OK: the odd datasets get the C of the others, not their own optimum; every C_k matches a brute-force recomputation")
    # corrupting the evaluated dataset (every window) never changes C_k; corrupting a member of others(k) in W_k can
    pick = {C: (1.0 if C == grid[-1] else 0.0) for C in grid}
    changed_by_member = 0
    for k in DEFAULT_DATASETS:
        bad = summary.copy()
        bad.loc[bad['dataset'] == k, 'cos_mean'] = bad.loc[bad['dataset'] == k, 'C'].map(pick)
        s2 = select_c(selection_curves(bad, window_of, grid, SELECT_EPS), 'synthetic').set_index('dataset')
        assert s2.loc[k, 'C'] == sel.loc[k, 'C'], f"C_{k} changed when only {k}'s own rows were corrupted"
        member = others(k)[0]
        bad = summary.copy()
        hit = (bad['dataset'] == member) & (bad['ventana'] == window_of[k])
        bad.loc[hit, 'cos_mean'] = bad.loc[hit, 'C'].map(lambda c: 50.0 if c == grid[-1] else 0.0)
        changed_by_member += int(select_c(selection_curves(bad, window_of, grid, SELECT_EPS), 'synthetic')
                                 .set_index('dataset').loc[k, 'C'] != sel.loc[k, 'C'])
        wrong = [x for x in summary['ventana'].unique() if x != window_of[k]][0]
        bad = summary.copy()
        hit = bad['dataset'].isin(others(k)) & (bad['ventana'] == wrong)
        bad.loc[hit, 'cos_mean'] = bad.loc[hit, 'C'].map(pick)
        assert select_c(selection_curves(bad, window_of, grid, SELECT_EPS), 'synthetic').set_index('dataset').loc[k, 'C'] \
            == sel.loc[k, 'C'], f"C_{k} changed when others(k) were corrupted in a window other than W_k"
    assert changed_by_member == len(DEFAULT_DATASETS), "the corruption test is not sensitive to others(k)"
    log("  OK: corrupting k (all windows) never moves C_k; corrupting others(k) outside W_k never moves C_k; "
        "corrupting a member of others(k) in W_k does (the test is sensitive)")

    log("#" * 100)
    log("SELFTEST D — ties are reported, not resolved")
    tie_curves = curves.copy()
    k0 = DEFAULT_DATASETS[0]
    top = tie_curves.loc[tie_curves['dataset'] == k0, 'F'].max()
    tie_curves.loc[(tie_curves['dataset'] == k0) & (tie_curves['C'].isin([4.0, 6.0])), 'F'] = top + 0.01
    st_ = select_c(tie_curves, 'synthetic').set_index('dataset')
    assert st_.loc[k0, 'empate'] and np.isnan(st_.loc[k0, 'C']) and st_.loc[k0, 'C_empatados'] == '4 6', st_.loc[k0]
    d_big =torch.stack([caches['hallucination']['diffs'][l] for l in range(2, 7)], dim=1).double() * 100   # every norm >> 3
    cos2, st2 = sweep_one(d_big, Z, Accountant('analytic', d_big.shape[0], args.delta, S), [2.0, 3.0], eps_grid)
    assert all(s['frac_clipped'] == 1.0 for s in st2)
    gap = float(np.abs(cos2[0, :, 0] - cos2[1, :, 0]).max())
    log(f"  every row clipped at C=2 and C=3: max |cos_PSA(C=2) − cos_PSA(C=3)| = {gap:.1e} (exact tie up to rounding, "
        f"TIE_TOL = {TIE_TOL:g})")
    assert 0.0 <= gap < TIE_TOL * 1e-2
    log("  OK: a tie leaves C empty, sets empate=True and lists the tied Cs")
    log("#" * 100)
    log(f"SELFTEST PASSED in {time.time() - t0:.1f} s (nothing written)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", choices=list(MODELS), help="sets --model, --rbar-csv, --diffs-dir, --out-dir defaults")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rbar-csv", default=None, help="default resultados_rbar_<tag>/rbar_results.csv")
    parser.add_argument("--diffs-dir", default=None, help="default sweep_<tag>/diffs (sweep_epsilon.py caches)")
    parser.add_argument("--out-dir", default=None, help="default sweep_c_<tag>; must not exist or be empty")
    parser.add_argument("--c-grid", type=float, nargs='+', default=C_GRID)
    parser.add_argument("--epsilons", type=float, nargs='+', default=EPSILONS)
    parser.add_argument("--select-eps", type=float, nargs='+', default=SELECT_EPS)
    parser.add_argument("--delta", type=float, default=1e-5, help="total delta")
    parser.add_argument("--calibration", default='analytic', choices=['analytic', 'classical'])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="dataset shuffle seed the caches were extracted with")
    parser.add_argument("--require-cache", action='store_true',
                        help="accepted for symmetry with sweep_epsilon.py; always in force (this script never loads a model)")
    parser.add_argument("--check-only", action='store_true', help="steps 1-2 only, from the R̄ csv; writes nothing")
    parser.add_argument("--table", nargs='+', metavar='DIR', help="print the models x datasets table from these output dirs")
    parser.add_argument("--selftest", action='store_true', help="synthetic data in memory, asserts, writes nothing")
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
        return
    if args.table:
        table(args.table)
        return
    if args.model_tag is None and (args.model is None or args.rbar_csv is None):
        parser.error("give --model-tag, or --model and --rbar-csv (and --diffs-dir / --out-dir)")
    tag = args.model_tag
    args.model = args.model or MODELS[tag]
    if tag is not None and args.model != MODELS[tag]:
        parser.error(f"--model {args.model!r} does not match --model-tag {tag} ({MODELS[tag]!r})")
    args.rbar_csv = args.rbar_csv or f"resultados_rbar_{tag}/rbar_results.csv"
    args.diffs_dir = args.diffs_dir or (f"sweep_{tag}/diffs" if tag else None)
    args.out_dir = args.out_dir or (f"sweep_c_{tag}" if tag else None)
    if not args.check_only and (args.diffs_dir is None or args.out_dir is None):
        parser.error("--diffs-dir and --out-dir are required without --model-tag")
    assert args.n_seeds >= 2
    if sorted(args.select_eps) != sorted(set(args.select_eps)) or not set(args.select_eps) <= set(args.epsilons):
        parser.error(f"--select-eps {args.select_eps} must be distinct values of --epsilons")
    log("=" * 100)
    log(f"model {args.model} | rbar csv {args.rbar_csv} | diffs {args.diffs_dir} | out {args.out_dir}")
    log(f"C grid {args.c_grid} | eps {args.epsilons} | select eps {args.select_eps} | delta {args.delta:g} | "
        f"{args.calibration} | {args.n_seeds} seeds")
    run(args)


if __name__ == '__main__':
    main()
