"""
Does the answer-letter direction estimated on the 7 public PSA datasets clean a dataset that took no part
in the estimate? If it does, the projection applied to tofu-forget10 (D_priv) can use a v that never
touches D_priv: a fixed, non-expansive map applied before PSA, so PSA's sensitivity is unchanged.

Every d_i = h(prompt, pos letter) - h(prompt, neg letter) is read at the letter token (read_token_index=-2).
Rows with answer_matching_behavior = (A) carry +(e_A - e_B), rows with (B) carry -(e_A - e_B), so
v_k = M_A - M_B (M_X = mean of the d_i with answer_matching_behavior = X) has the same sign in every
dataset and the per-dataset directions can be averaged. The sign rests on (A) -> (B) and (B) -> (A) being
the only (pos, neg) combinations of the A/B rows, which step 0 asserts (SystemExit otherwise).

Inputs: the sweep_epsilon.py caches <diffs-dir>/<dataset>.pt, the A/B jsons, measure_rbar.py's csv and,
optionally, test_letter_effect.py's csv. Step 0 is cache_alignment.align (permutation, independent
verification, R̄ cross-check); any dataset that does not verify stops the run. No model, no GPU; all
numerics in float64, one layer at a time.

Definitions, per layer
  v̂_k            (M_A - M_B)/‖·‖ over the rows with pos in {(A),(B)} of dataset k. Rows with other letters
                  (survival-instinct has (C)..(H), all with neg (A)) do not enter any v but do enter R̄ and
                  every variance.
  v_pool^(-k)     unit(mean of the unit v̂_j, j in PSA, j != k): primary, pre-registered. tofu-forget10
                  never enters a pool. For a non-PSA target (tofu) the pool is all the PSA datasets given.
                  cos(v̂_j, v_pool) > 0 is checked for every member j; violations are logged.
  concat pool     M_A - M_B over the concatenated A/B rows of the same members (sensitivity only).
  projection      d_clean = d - (d·v) v, rank 1.

Analyses
  1  8x8 cosine matrix between the v̂_k (tofu included, marked diagnostic).
  2  Split-half reliability: --n-splits random halvings stratified by letter (every letter, extra letters
     included); r_half = cos(v_half1, v_half2); SB = 2 r/(1+r) with r = mean r_half; ceiling_k = √SB.
     Under isotropic noise √SB approximates the expected cosine between v̂_k (full n) and its population
     direction: an approximation, not a bound (anisotropic noise or a very weak letter signal break it;
     SB <= 0 gives ceiling = NaN).
  3  Transfer, target k: cos_transfer = cos(v_pool^(-k), v̂_k), cos_ratio = cos_transfer / ceiling_k.
     R̄_none (no projection), R̄_oracle (own v̂_k, in-sample), R̄_oracle_crossfit (each half cleaned with
     the v of the other half, R̄ over all n rows, mean over the same splits), R̄_transfer (v_pool^(-k)).
     recovered = (R̄_transfer - R̄_none)/(R̄_oracle - R̄_none), NaN when |denominator| < 1e-6; also
     against the crossfit oracle. residual_mean_gap = ‖P(M_A - M_B)‖/‖M_A - M_B‖ with P the transferred
     projection (0 = the letter means coincide after projecting: exact linear erasure in the sample, the
     LEACE criterion). letter_var_before/after = variance of the centred d along v̂_k over the total
     variance of the original d, before / after projecting with v_pool^(-k). mean_energy_in_letter_before/
     after = ‖(mean d)·v̂_k‖²/‖mean d‖² before / after (test_letter_effect's metric with the own v̂_k);
     mean_energy_removed_transfer = the same share along v_pool (what the transfer takes from the numerator).
     Secondary: cos_transfer_concat, rbar_transfer_concat, recovered_concat with the concat pool.
  4  Permutation null: --n-perm permutations of the letters inside every pool dataset, v_pool_perm
     recomputed, cos(v_pool_perm, v̂_k); p2.5, p97.5 and max, next to 1/√hidden.
  5  tofu-forget10 as an oracle diagnostic: v_pool^(PSA) = pool of the 7, same metrics against tofu's own
     v̂ (used only to measure). R̄_transfer of tofu is what the deployed mechanism reaches without D_priv.
     role = 'loo_public' for the PSA datasets, 'oracle_diagnostic' otherwise; only loo_public enters the
     verdict.
  6  Option format: cached questions listing the options as "\\n (A)" (space) vs "\\n(A)" (no space).

Cross-check (optional): with --letter-effect-csv present, for every dataset whose letters_used is exactly
"(A) (B)", R̄_oracle must equal rbar_projected (and R̄_none rbar_global, n equal) to 1e-6 on every layer,
else SystemExit.

Criteria (constants below, printed with the verdict, not tuned after looking at results): a (dataset,
primary layer) pair passes when recovered >= RECOVERED_MIN. "transfiere" = every PSA dataset passes on
every primary layer; "no transfiere" = most pairs fail; "parcial" otherwise. COS_RATIO_MIN is reported only.

Outputs (in --out-dir): letter_transfer.csv, letter_transfer_cos_matrix.csv, letter_transfer_splithalf.csv,
letter_transfer_verdict.txt, letter_transfer_cos_heatmap.png, letter_transfer_cos_by_layer.png,
letter_transfer_rbar_by_layer.png

    python test_letter_transfer.py --selftest
    python test_letter_transfer.py --diffs-dir sweep_1b/diffs --rbar-csv resultados_rbar_1b/rbar_results.csv \\
        --letter-effect-csv letter_effect/letter_effect.csv --out-dir letter_transfer
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from cache_alignment import (DEFAULT_DATASETS, OWN_DATASET, PSA_DATASETS, align, load_cache, load_dataset_json,
                             load_rbar_csv, log, rbar_float32_as_measure_rbar, shuffle_permutation)
from test_letter_effect import (C_A, C_B, C_CONTROL, C_GLOBAL, C_GRID, C_SURFACE, C_TEXT, C_TEXT2, CI_PERCENTILES,
                                describe_letters, grid_figure, letters_of, project_out, rbar_of, style_axis)

# ---- pre-registered criteria (printed with the verdict; do not tune after looking at results) ----------
RECOVERED_MIN = 0.90   # primary: a (dataset, primary layer) pair passes when recovered >= this
COS_RATIO_MIN = 0.90   # secondary: cos_transfer / ceiling, reported only
# ---------------------------------------------------------------------------------------------------------
DENOM_EPS = 1e-6           # recovered = NaN when |R̄_oracle - R̄_none| < DENOM_EPS
LETTER_EFFECT_TOL = 1e-6   # R̄_oracle vs test_letter_effect's rbar_projected
MIN_LETTER_N = 5           # same as test_letter_effect --min-letter-n (describe_letters needs it)
ROLE_LOO, ROLE_DIAG = 'loo_public', 'oracle_diagnostic'
FMT_SPACE, FMT_NOSPACE = '\n (A)', '\n(A)'

# Diverging pair of the dataviz reference palette: blue <-> red with a grey midpoint (cos = 0).
C_DIV_NEG, C_DIV_MID, C_DIV_POS = '#2a78d6', '#f0efec', '#e34948'


# --------------------------------------------------------------------------------------
# Step 0 and per-dataset index structures (shared by every layer)
# --------------------------------------------------------------------------------------
def check_ab_signs(name: str, pos: np.ndarray, neg: np.ndarray) -> None:
    """The common sign of v = M_A - M_B needs (A)->(B) and (B)->(A) to be the only A/B-row combinations."""
    bad_a = sorted(set(neg[pos == '(A)'].tolist()) - {'(B)'})
    bad_b = sorted(set(neg[pos == '(B)'].tolist()) - {'(A)'})
    if bad_a or bad_b:
        raise SystemExit(f"[{name}] rows with pos (A) have neg {bad_a or '(B)'} and rows with pos (B) have neg "
                         f"{bad_b or '(A)'}: d_i of an A/B row is not ±(e_A - e_B), the sign of M_A - M_B is not "
                         f"comparable across datasets. Stopping.")


def option_format(name: str, ab: list[dict], perm: list[int]) -> dict:
    q = [ab[j]['question'] for j in perm]
    fmt = {'fmt_space': sum(FMT_SPACE in s for s in q), 'fmt_nospace': sum(FMT_NOSPACE in s for s in q),
           'json_space': sum(FMT_SPACE in r['question'] for r in ab),
           'json_nospace': sum(FMT_NOSPACE in r['question'] for r in ab)}
    expected = 'space' if name in PSA_DATASETS else 'nospace'
    log(f"  option format: {fmt['fmt_space']} cached questions with '\\n (A)', {fmt['fmt_nospace']} with '\\n(A)' "
        f"(json: {fmt['json_space']} / {fmt['json_nospace']} of {len(ab)})")
    if fmt[f'fmt_{expected}'] != len(q):
        log(f"  NOTE: the documented assumption is PSA = '\\n (A)' and {OWN_DATASET} = '\\n(A)'; {name} does not follow "
            f"it ({expected} expected). Recorded, not an error.")
    return fmt


def step0(name: str, cache: dict, ab: list[dict], csv: pd.DataFrame, csv_label: str) -> tuple[np.ndarray, dict]:
    log(f"  n={cache['n']} of total={cache['total']} | shuffle seed {cache['seed']} | read_token_index "
        f"{cache['read_token_index']} | hidden {cache['hidden']} | {len(cache['layers'])} layers")
    perm = align(cache, ab, csv, csv_label)
    pos, neg = letters_of(ab, perm)
    describe_letters(name, pos, neg, MIN_LETTER_N)
    check_ab_signs(name, pos, neg)
    return pos, option_format(name, ab, perm)


def diff_weights(is_a: np.ndarray, is_b: np.ndarray) -> np.ndarray:
    """(..., n) masks -> (..., n) weights w with w @ d = M_A - M_B."""
    n_a = is_a.sum(axis=-1, keepdims=True)
    n_b = is_b.sum(axis=-1, keepdims=True)
    assert (n_a > 0).all() and (n_b > 0).all(), "a half / permutation without (A) or (B) rows"
    return is_a / n_a - is_b / n_b


def stratified_halves(pos: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """(n_splits, n) bool, True = half 1. Every letter is split in two; an odd letter count sends its extra
    row to a random half."""
    n = len(pos)
    h1 = np.zeros((n_splits, n), dtype=bool)
    groups = [np.flatnonzero(pos == x) for x in sorted(set(pos.tolist()))]
    for s in range(n_splits):
        for idx in groups:
            k = len(idx) // 2 + (int(rng.integers(2)) if len(idx) % 2 else 0)
            h1[s, rng.permutation(idx)[:k]] = True
    return h1


def prepare(name: str, pos: np.ndarray, ds_index: int, seed: int, n_splits: int, n_perm: int, in_pool: bool) -> dict:
    is_a, is_b = pos == '(A)', pos == '(B)'
    h1 = stratified_halves(pos, n_splits, np.random.default_rng([seed, 1, ds_index]))
    prep = {'name': name, 'n': len(pos), 'pos': pos, 'is_a': is_a, 'is_b': is_b,
            'n_a': int(is_a.sum()), 'n_b': int(is_b.sum()), 'w': diff_weights(is_a, is_b), 'h1': h1,
            'w1': diff_weights(h1 & is_a, h1 & is_b), 'w2': diff_weights(~h1 & is_a, ~h1 & is_b), 'wperm': None}
    if in_pool:  # letters permuted inside the dataset (counts preserved), same permutations on every layer
        rng = np.random.default_rng([seed, 2, ds_index])
        perms = np.stack([rng.permutation(len(pos)) for _ in range(n_perm)])
        prep['wperm'] = diff_weights(is_a[perms], is_b[perms])
    return prep


# --------------------------------------------------------------------------------------
# Numerics (float64)
# --------------------------------------------------------------------------------------
def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def rbar_crossfit(d: np.ndarray, h1: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """(n_splits,) R̄ over all n rows when half-1 rows are cleaned with v2 and half-2 rows with v1 (unit rows).
    ‖P_v d_i‖² = ‖d_i‖² - (d_i·v)², mean(d_clean) = mean(d) - (Σ_{h1} (d_i·v2) v2 + Σ_{h2} (d_i·v1) v1)/n."""
    n = d.shape[0]
    p1, p2 = d @ v1.T, d @ v2.T                           # (n, S)
    in1 = h1.T
    proj = np.where(in1, p2, p1)
    norms = np.sqrt(np.clip((d ** 2).sum(axis=1)[:, None] - proj ** 2, 0.0, None)).mean(axis=0)
    a = (p2 * in1).sum(axis=0)
    b = (p1 * ~in1).sum(axis=0)
    m_clean = d.mean(axis=0)[None, :] - (a[:, None] * v2 + b[:, None] * v1) / n
    return np.linalg.norm(m_clean, axis=1) / norms


def dataset_layer_stats(prep: dict, d: np.ndarray) -> dict:
    g = prep['w'] @ d                                     # M_A - M_B
    vhat = g / np.linalg.norm(g)
    v1, v2 = unit(prep['w1'] @ d), unit(prep['w2'] @ d)
    r_half = (v1 * v2).sum(axis=1)
    cf = rbar_crossfit(d, prep['h1'], v1, v2)
    # vectorised crossfit vs a direct projection, split 0 (guards the algebra above on every real layer)
    h = prep['h1'][0]
    dc = np.empty_like(d)
    dc[h] = project_out(d[h], v2[0][None])
    dc[~h] = project_out(d[~h], v1[0][None])
    assert abs(rbar_of(dc) - cf[0]) < 1e-9, f"crossfit R̄ mismatch {rbar_of(dc)} vs {cf[0]}"
    return {'g': g, 'vhat': vhat, 'r_half': r_half, 'rbar_cf': cf,
            'vperm': unit(prep['wperm'] @ d) if prep['wperm'] is not None else None,
            'sum_a': d[prep['is_a']].sum(axis=0), 'sum_b': d[prep['is_b']].sum(axis=0),
            'rbar_none': rbar_of(d), 'rbar_oracle': rbar_of(project_out(d, vhat[None]))}


def recovered_of(t: float, none: float, oracle: float) -> float:
    den = oracle - none
    return float('nan') if abs(den) < DENOM_EPS else (t - none) / den


def transfer_metrics(d: np.ndarray, st: dict, v: np.ndarray) -> dict:
    """Metrics of projecting the unit direction v out of d, measured against the dataset's own v̂."""
    vhat, g = st['vhat'], st['g']
    n = d.shape[0]
    m = d.mean(axis=0)
    dc = d - m
    total = float((dc ** 2).sum() / n)
    dcc = project_out(dc, v[None])
    m_clean = m - (m @ v) * v
    g_res = g - (g @ v) * v
    cos_t = float(v @ vhat)
    gap = float(np.linalg.norm(g_res) / np.linalg.norm(g))
    assert abs(gap - np.sqrt(max(0.0, 1.0 - cos_t ** 2))) < 1e-8, "residual_mean_gap != sqrt(1 - cos_transfer^2)"
    rbar_t = rbar_of(project_out(d, v[None]))
    return {'cos_transfer': cos_t, 'rbar_transfer': rbar_t,
            'recovered': recovered_of(rbar_t, st['rbar_none'], st['rbar_oracle']),
            'recovered_crossfit': recovered_of(rbar_t, st['rbar_none'], float(st['rbar_cf'].mean())),
            'residual_mean_gap': gap,
            'letter_var_before': float(((dc @ vhat) ** 2).sum() / n) / total,
            'letter_var_after': float(((dcc @ vhat) ** 2).sum() / n) / total,
            'mean_energy_in_letter_before': float((m @ vhat) ** 2 / (m @ m)),
            'mean_energy_in_letter_after': float((m_clean @ vhat) ** 2 / (m_clean @ m_clean)),
            'mean_energy_removed_transfer': float((m @ v) ** 2 / (m @ m))}


# --------------------------------------------------------------------------------------
# Analysis over all datasets and layers
# --------------------------------------------------------------------------------------
def analyse(datasets: list[str], caches: dict, preps: dict, fmts: dict, primary: list[int]
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layers = caches[datasets[0]]['layers']
    hidden = caches[datasets[0]]['hidden']
    pool_all = [x for x in datasets if x in PSA_DATASETS]
    rows, cos_rows, split_rows, warnings = [], [], [], []
    for layer in layers:
        d64 = {x: caches[x]['diffs'][layer].numpy().astype(np.float64) for x in datasets}
        st = {x: dataset_layer_stats(preps[x], d64[x]) for x in datasets}
        V = np.stack([st[x]['vhat'] for x in datasets])
        C = V @ V.T
        for i, a in enumerate(datasets):
            for j, b in enumerate(datasets):
                cos_rows.append({'layer': layer, 'ds_i': a, 'ds_j': b, 'cos': float(C[i, j]),
                                 'diagnostic': a not in PSA_DATASETS or b not in PSA_DATASETS})
        for x in datasets:
            r = st[x]['r_half']
            r_mean = float(r.mean())
            sb = 2 * r_mean / (1 + r_mean)
            ceiling = float(np.sqrt(sb)) if sb > 0 else float('nan')
            if not sb > 0:
                warnings.append(f"layer {layer} {x}: mean r_half = {r_mean:.3f} -> SB = {sb:.3f} <= 0, ceiling = NaN")
            lo, hi = np.percentile(r, CI_PERCENTILES)
            cf = st[x]['rbar_cf']
            cf_lo, cf_hi = np.percentile(cf, CI_PERCENTILES)
            split_rows.append({'dataset': x, 'layer': layer, 'n_splits': len(r), 'r_half_mean': r_mean,
                               'r_half_p2_5': float(lo), 'r_half_p97_5': float(hi), 'r_half_min': float(r.min()),
                               'sb': sb, 'ceiling': ceiling, 'rbar_oracle_crossfit': float(cf.mean()),
                               'rbar_oracle_crossfit_p2_5': float(cf_lo), 'rbar_oracle_crossfit_p97_5': float(cf_hi)})
        split = {r_['dataset']: r_ for r_ in split_rows if r_['layer'] == layer}

        for x in datasets:
            members = [j for j in pool_all if j != x]
            vpool = unit(np.mean([st[j]['vhat'] for j in members], axis=0))
            member_cos = {j: float(st[j]['vhat'] @ vpool) for j in members}
            for j, c in member_cos.items():
                if not c > 0:
                    warnings.append(f"layer {layer} pool for {x}: cos(v̂_{j}, v_pool) = {c:.3f} <= 0")
            null = unit(np.sum([st[j]['vperm'] for j in members], axis=0)) @ st[x]['vhat']
            n_lo, n_hi = np.percentile(null, CI_PERCENTILES)
            va = sum(st[j]['sum_a'] for j in members) / sum(preps[j]['n_a'] for j in members)
            vb = sum(st[j]['sum_b'] for j in members) / sum(preps[j]['n_b'] for j in members)
            vconcat = unit(va - vb)
            tm = transfer_metrics(d64[x], st[x], vpool)
            rbar_tc = rbar_of(project_out(d64[x], vconcat[None]))
            sp = split[x]
            p = preps[x]
            rows.append({
                'dataset': x, 'role': ROLE_LOO if x in PSA_DATASETS else ROLE_DIAG, 'layer': layer,
                'primary_layer': layer in primary,
                'rbar_none': st[x]['rbar_none'], 'rbar_oracle': st[x]['rbar_oracle'],
                'rbar_oracle_crossfit': sp['rbar_oracle_crossfit'], 'rbar_transfer': tm['rbar_transfer'],
                'recovered': tm['recovered'], 'recovered_crossfit': tm['recovered_crossfit'],
                'rbar_gain_oracle': st[x]['rbar_oracle'] - st[x]['rbar_none'],   # denominator of recovered
                'passes': bool(tm['recovered'] >= RECOVERED_MIN),
                'cos_transfer': tm['cos_transfer'], 'ceiling': sp['ceiling'],
                'cos_ratio': tm['cos_transfer'] / sp['ceiling'],
                'cos_ratio_ok': bool(tm['cos_transfer'] / sp['ceiling'] >= COS_RATIO_MIN),
                'r_half_mean': sp['r_half_mean'], 'sb': sp['sb'],
                'null_cos_p2_5': float(n_lo), 'null_cos_p97_5': float(n_hi), 'null_cos_max': float(null.max()),
                'cos_above_null_p97_5': bool(tm['cos_transfer'] > n_hi), 'inv_sqrt_hidden': 1 / np.sqrt(hidden),
                'residual_mean_gap': tm['residual_mean_gap'],
                'letter_var_before': tm['letter_var_before'], 'letter_var_after': tm['letter_var_after'],
                'mean_energy_in_letter_before': tm['mean_energy_in_letter_before'],
                'mean_energy_in_letter_after': tm['mean_energy_in_letter_after'],
                'mean_energy_removed_transfer': tm['mean_energy_removed_transfer'],
                'cos_transfer_concat': float(vconcat @ st[x]['vhat']), 'rbar_transfer_concat': rbar_tc,
                'recovered_concat': recovered_of(rbar_tc, st[x]['rbar_none'], st[x]['rbar_oracle']),
                'pool_size': len(members), 'pool_min_member_cos': min(member_cos.values()),
                'pool_datasets': ' '.join(members),
                'n': p['n'], 'n_letterA': p['n_a'], 'n_letterB': p['n_b'], 'n_other_letters': p['n'] - p['n_a'] - p['n_b'],
                'hidden': hidden, 'fmt_space': fmts[x]['fmt_space'], 'fmt_nospace': fmts[x]['fmt_nospace'],
            })
    for w in warnings:
        log(f"  WARNING: {w}")
    if not warnings:
        log("  pool check OK: cos(v̂_j, v_pool) > 0 for every member of every pool on every layer; SB > 0 everywhere")
    return pd.DataFrame(rows), pd.DataFrame(cos_rows), pd.DataFrame(split_rows)


def crosscheck_letter_effect(df: pd.DataFrame, le: pd.DataFrame, label: str) -> None:
    """SystemExit unless R̄_oracle == rbar_projected (and R̄_none == rbar_global, same n) to 1e-6 for every
    dataset whose letters_used is exactly '(A) (B)' in test_letter_effect's csv."""
    for col in ('dataset', 'layer', 'n', 'rbar_global', 'rbar_projected', 'letters_used'):
        if col not in le.columns:
            raise SystemExit(f"{label} has no column {col!r}; it is not a test_letter_effect.py csv")
    checked, skipped = [], []
    for name in df['dataset'].unique():
        ref = le[le['dataset'] == name]
        if ref.empty:
            skipped.append(f"{name} (absent)")
            continue
        used = set(ref['letters_used'].astype(str))
        if used != {'(A) (B)'}:
            skipped.append(f"{name} (letters_used {sorted(used)})")
            continue
        ref = ref.set_index('layer')
        worst = 0.0
        for _, r in df[df['dataset'] == name].iterrows():
            layer = r['layer']
            if layer not in ref.index:
                raise SystemExit(f"[{name}] layer {layer} is not in {label}")
            if int(ref.loc[layer, 'n']) != int(r['n']):
                raise SystemExit(f"[{name}] layer {layer}: {label} n={int(ref.loc[layer, 'n'])} != cache n={int(r['n'])}")
            diffs = (abs(r['rbar_oracle'] - float(ref.loc[layer, 'rbar_projected'])),
                     abs(r['rbar_none'] - float(ref.loc[layer, 'rbar_global'])))
            worst = max(worst, *diffs)
            if max(diffs) > LETTER_EFFECT_TOL:
                raise SystemExit(f"[{name}] layer {layer}: R̄_oracle {r['rbar_oracle']:.8f} vs rbar_projected "
                                 f"{float(ref.loc[layer, 'rbar_projected']):.8f}, R̄_none {r['rbar_none']:.8f} vs rbar_global "
                                 f"{float(ref.loc[layer, 'rbar_global']):.8f} (tolerance {LETTER_EFFECT_TOL:g}) in {label}. "
                                 f"The two scripts disagree on the same cache; stopping.")
        checked.append(f"{name} ({worst:.1e})")
    log(f"  cross-check vs {label} OK (max |diff| per dataset): {', '.join(checked) or 'none'}"
        + (f" | skipped: {', '.join(skipped)}" if skipped else ""))


# --------------------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------------------
def build_verdict(df: pd.DataFrame, datasets: list[str], primary: list[int], fmts: dict) -> tuple[str, str]:
    psa = [x for x in datasets if x in PSA_DATASETS]
    diag = [x for x in datasets if x not in PSA_DATASETS]
    pairs = df[(df['role'] == ROLE_LOO) & df['layer'].isin(primary)]
    n_pairs, n_pass = len(pairs), int(pairs['passes'].sum())
    fails = pairs[~pairs['passes']]
    if n_pass == n_pairs:
        label = "transfiere"
    elif len(fails) > n_pairs / 2:
        label = "no transfiere"
    else:
        label = "parcial"
    if len(fails) <= 10:
        fail_txt = "; ".join(f"{r['dataset']} L{int(r['layer'])} ({r['recovered']:.3f}; R̄_oracle − R̄_none = "
                             f"{r['rbar_gain_oracle']:.4f}, cos_ratio {r['cos_ratio']:.3f})" for _, r in fails.iterrows())
    else:  # grouped by dataset, detail in the csv
        fail_txt = "; ".join(f"{x} capas {sorted(int(l) for l in g['layer'])} (recovered máx {g['recovered'].max():.3f})"
                             for x, g in fails.groupby('dataset', sort=False))
    lines = ["VEREDICTO",
             f"  ¿la dirección de letra estimada con los otros datasets de PSA limpia un dataset que no participó?   {label}"
             + ("" if len(psa) == 7 else f"   (solo {len(psa)} datasets de PSA analizados)"),
             f"  pares (dataset, capa primaria) que pasan: {n_pass}/{n_pairs}; capas primarias {primary}"]
    if len(fails):
        lines.append(f"  fallan (recovered): {fail_txt}")
    rat = pairs['cos_ratio']
    gain = pairs['rbar_gain_oracle'].abs()
    lines += [
        f"  denominador de recovered |R̄_oracle − R̄_none| en capas primarias: mín {gain.min():.4f} "
        f"({pairs.loc[gain.idxmin(), 'dataset'] if n_pairs else '-'}), mediana {gain.median():.4f}; con denominador chico "
        f"(letras desbalanceadas) recovered amplifica diferencias pequeñas, mirar cos_ratio junto a él",
        f"  recovered en capas primarias (PSA, LOO):  mediana {pairs['recovered'].median():.3f}, mín {pairs['recovered'].min():.3f}"
        f" | vs oráculo crossfit: mediana {pairs['recovered_crossfit'].median():.3f}, mín {pairs['recovered_crossfit'].min():.3f}",
        f"  secundario (solo se reporta): cos_transfer mediana {pairs['cos_transfer'].median():.3f}; cos_ratio >= {COS_RATIO_MIN} "
        f"en {int((rat >= COS_RATIO_MIN).sum())}/{n_pairs} (mediana {rat.median():.3f}); pool concatenado: cos mediana "
        f"{pairs['cos_transfer_concat'].median():.3f}, recovered mediana {pairs['recovered_concat'].median():.3f}",
        f"  nulo por permutación: cos_transfer > p97.5 del nulo en {int(pairs['cos_above_null_p97_5'].sum())}/{n_pairs} pares; "
        f"p97.5 del nulo mediana {pairs['null_cos_p97_5'].median():.3f}, máx {pairs['null_cos_max'].max():.3f}; "
        f"1/√hidden = {pairs['inv_sqrt_hidden'].iloc[0] if n_pairs else float('nan'):.3f}",
        f"  brecha residual de medias ‖P(M_A−M_B)‖/‖M_A−M_B‖ (0 = borrado lineal exacto): mediana "
        f"{pairs['residual_mean_gap'].median():.3f}, máx {pairs['residual_mean_gap'].max():.3f}",
        f"  criterio: recovered = (R̄_transfer − R̄_none)/(R̄_oracle − R̄_none) >= {RECOVERED_MIN} (NaN si |denominador| < "
        f"{DENOM_EPS:g} cuenta como falla); 'transfiere' = pasan los {len(psa)} datasets de PSA en todas las capas primarias; "
        f"'no transfiere' = falla la mayoría de los pares; 'parcial' = el resto. COS_RATIO_MIN = {COS_RATIO_MIN} solo se reporta."]
    for x in diag:
        sub = df[(df['dataset'] == x) & df['layer'].isin(primary)].sort_values('layer')
        lines.append(f"  DIAGNÓSTICO {x} (no entra al veredicto; pool = los {int(sub['pool_size'].iloc[0])} de PSA, "
                     f"v̂ propio solo para medir):")
        for _, r in sub.iterrows():
            lines.append(f"    L{int(r['layer'])}: R̄ none {r['rbar_none']:.3f} → transfer {r['rbar_transfer']:.3f} "
                         f"(oracle {r['rbar_oracle']:.3f}, crossfit {r['rbar_oracle_crossfit']:.3f}) | recovered "
                         f"{r['recovered']:.3f} | cos {r['cos_transfer']:.3f}, techo {r['ceiling']:.3f}, ratio {r['cos_ratio']:.3f} | "
                         f"nulo p97.5 {r['null_cos_p97_5']:.3f}")
    lines.append("  formato de opciones (filas cacheadas, '\\n (A)' / '\\n(A)'): "
                 + ", ".join(f"{x} {fmts[x]['fmt_space']}/{fmts[x]['fmt_nospace']}" for x in datasets))
    off = [x for x in psa if fmts[x]['fmt_space'] != preps_n(df, x)] + \
          [x for x in diag if fmts[x]['fmt_nospace'] != preps_n(df, x)]
    if off:
        lines.append(f"  (no siguen el supuesto 'PSA con espacio, {OWN_DATASET} sin espacio': {', '.join(off)})")
    if label == "transfiere":
        concl = (f"la dirección de letra es compartida por el protocolo: para {OWN_DATASET} se puede proyectar con "
                 f"v_pool^(PSA), que no toca D_priv (proyección fija y no expansiva: la sensibilidad de PSA no cambia)")
    elif label == "no transfiere":
        concl = ("la dirección de letra no transfiere entre datasets: quitarla de D_priv requeriría estimarla con D_priv "
                 "(y pagar esa estimación en privacidad)")
    else:
        concl = "transferencia parcial: revisar los pares que fallan antes de usar v_pool en D_priv"
    lines.append(f"  => {concl}")
    return "\n".join(lines), label


def preps_n(df: pd.DataFrame, x: str) -> int:
    return int(df.loc[df['dataset'] == x, 'n'].iloc[0])


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def display_name(x: str) -> str:
    return x if x in PSA_DATASETS else f"{x} (diagnostic)"


def plot_heatmap(cos_df: pd.DataFrame, datasets: list[str], layer: int, header: str, path: Path) -> None:
    sub = cos_df[cos_df['layer'] == layer].pivot(index='ds_i', columns='ds_j', values='cos').loc[datasets, datasets]
    cmap = LinearSegmentedColormap.from_list('div', [C_DIV_NEG, C_DIV_MID, C_DIV_POS])
    k = len(datasets)
    fig, ax = plt.subplots(figsize=(1.05 * k + 3.2, 0.9 * k + 2.2))
    fig.patch.set_facecolor(C_SURFACE)
    im = ax.imshow(sub.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sub.iat[i, j]:.2f}", ha='center', va='center', fontsize=8, color=C_TEXT)
    names = [display_name(x) for x in datasets]
    ax.set_xticks(range(k), names, rotation=35, ha='right', fontsize=8, color=C_TEXT2)
    ax.set_yticks(range(k), names, fontsize=8, color=C_TEXT2)
    for i, x in enumerate(datasets):  # frame the diagnostic row/column instead of recolouring it
        if x not in PSA_DATASETS:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), k, 1, fill=False, edgecolor=C_TEXT2, linewidth=1.2, linestyle='--'))
            ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, k, fill=False, edgecolor=C_TEXT2, linewidth=1.2, linestyle='--'))
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=C_TEXT2, labelsize=8)
    cb.outline.set_visible(False)
    cb.set_label('cos(v̂_i, v̂_j)', color=C_TEXT2, fontsize=9)
    ax.set_title(f"Cosine between the per-dataset letter directions v̂ = (M_A − M_B)/‖·‖, layer {layer}",
                 loc='left', fontsize=10, color=C_TEXT)
    fig.suptitle(header, x=0.01, ha='left', fontsize=9, color=C_TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def shade_primary(ax, primary: list[int]) -> None:
    ax.axvspan(min(primary) - 0.5, max(primary) + 0.5, color=C_GRID, alpha=0.45, linewidth=0, zorder=0)


def legend_union(fig, axes) -> None:
    handles, labels = {}, []
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in handles:
                handles[l_] = h_
                labels.append(l_)
    fig.legend([handles[l_] for l_ in labels], labels, frameon=False, fontsize=9, loc='lower center', ncol=len(labels))


def plot_cos_grid(df: pd.DataFrame, datasets: list[str], primary: list[int], note_layer: int, header: str, path: Path) -> None:
    layers = sorted(df['layer'].unique())
    fig, axes = grid_figure(datasets)
    lo = min(-0.2, float(df[['null_cos_p2_5', 'cos_transfer', 'cos_transfer_concat']].min().min()) - 0.05)
    for ax, x in zip(axes, datasets):
        sub = df[df['dataset'] == x].sort_values('layer')
        shade_primary(ax, primary)
        ax.axhline(0, color=C_GRID, linewidth=1)
        ax.fill_between(sub['layer'], sub['null_cos_p2_5'], sub['null_cos_p97_5'], color=C_CONTROL, alpha=0.5, linewidth=0,
                        label='permutation null (95% interval)')
        ax.plot(sub['layer'], sub['null_cos_max'], color=C_TEXT2, linewidth=1, linestyle=':', label='permutation null max')
        ax.plot(sub['layer'], sub['ceiling'], color=C_GLOBAL, linewidth=1.6, linestyle='--', label='ceiling √SB (split-half)')
        ax.plot(sub['layer'], sub['cos_transfer_concat'], color=C_B, linewidth=1.4, marker='s', markersize=3.5,
                label='cos(concat pool, v̂) (secondary)')
        ax.plot(sub['layer'], sub['cos_transfer'], color=C_A, linewidth=2, marker='o', markersize=4,
                label='cos(v_pool, v̂) (LOO pool of unit v̂)')
        r = sub.set_index('layer').loc[note_layer]
        style_axis(ax, display_name(x), layers,
                   note=f"L{note_layer}: cos {r['cos_transfer']:.2f} · ceiling {r['ceiling']:.2f} · ratio {r['cos_ratio']:.2f}\n"
                        f"null p97.5 {r['null_cos_p97_5']:.2f} · pool of {int(r['pool_size'])}")
        ax.set_ylim(lo, 1.05)
    for ax in axes[-4:]:
        ax.set_xlabel('layer index', color=C_TEXT2, fontsize=9)
    axes[0].set_ylabel('cosine', color=C_TEXT2, fontsize=9)
    legend_union(fig, axes)
    fig.suptitle(f"Transfer of the letter direction: pooled direction vs each dataset's own v̂ (shaded: primary layers) — {header}",
                 x=0.01, ha='left', fontsize=11, color=C_TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rbar_grid(df: pd.DataFrame, datasets: list[str], primary: list[int], note_layer: int, header: str, path: Path) -> None:
    layers = sorted(df['layer'].unique())
    fig, axes = grid_figure(datasets)
    for ax, x in zip(axes, datasets):
        sub = df[df['dataset'] == x].sort_values('layer')
        shade_primary(ax, primary)
        ax.plot(sub['layer'], sub['rbar_none'], color=C_GLOBAL, linewidth=2.2, marker='o', markersize=4, label='R̄ none')
        ax.plot(sub['layer'], sub['rbar_oracle'], color=C_A, linewidth=1.8, marker='o', markersize=4,
                label='R̄ oracle (own v̂, in-sample)')
        ax.plot(sub['layer'], sub['rbar_transfer'], color=C_B, linewidth=1.8, marker='s', markersize=4, linestyle='--',
                label='R̄ transfer (pooled v)')
        r = sub.set_index('layer').loc[note_layer]
        style_axis(ax, display_name(x), layers,
                   note=f"L{note_layer}: recovered {r['recovered']:.3f} (crossfit {r['recovered_crossfit']:.3f})\n"
                        f"none {r['rbar_none']:.2f} · oracle {r['rbar_oracle']:.2f} · transfer {r['rbar_transfer']:.2f}")
        ax.set_ylim(0, 1)
    for ax in axes[-4:]:
        ax.set_xlabel('layer index', color=C_TEXT2, fontsize=9)
    axes[0].set_ylabel('R̄', color=C_TEXT2, fontsize=9)
    legend_union(fig, axes)
    fig.suptitle(f"R̄ without projection, projecting the own letter direction, and projecting the pooled one (shaded: primary layers) — {header}",
                 x=0.01, ha='left', fontsize=11, color=C_TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Pipeline (shared by the real run and --selftest)
# --------------------------------------------------------------------------------------
def run_pipeline(datasets: list[str], caches: dict, abs_: dict, csv: pd.DataFrame, csv_label: str,
                 le: pd.DataFrame | None, le_label: str, args, out_dir: Path) -> tuple[pd.DataFrame, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    psa = [x for x in datasets if x in PSA_DATASETS]
    if len(psa) < 2:
        raise SystemExit(f"need at least 2 PSA datasets for a leave-one-out pool, have {psa}")
    model, layers, hidden = None, None, None
    preps, fmts = {}, {}
    log("=" * 100)
    for i, x in enumerate(datasets):
        cache = caches[x]
        log(f"[{x}] STEP 0")
        if model is None:
            model, layers, hidden = cache['model'], list(cache['layers']), cache['hidden']
        elif cache['model'] != model or list(cache['layers']) != layers or cache['hidden'] != hidden:
            raise SystemExit(f"[{x}] cache model/layers/hidden ({cache['model']!r}, {cache['layers']}, {cache['hidden']}) "
                             f"differ from {model!r}, {layers}, {hidden}: mixed caches")
        pos, fmts[x] = step0(x, cache, abs_[x], csv, csv_label)
        preps[x] = prepare(x, pos, i, args.seed, args.n_splits, args.n_perm, in_pool=x in PSA_DATASETS)
        log("=" * 100)
    primary = list(args.primary_layers)
    missing = [l for l in primary + [args.heatmap_layer] if l not in layers]
    if missing:
        raise SystemExit(f"layers {missing} (--primary-layers / --heatmap-layer) are not in the caches ({layers})")

    log(f"ANALYSIS: {len(layers)} layers, pool = {psa} (LOO for each), {args.n_splits} splits, {args.n_perm} permutations, "
        f"seed {args.seed}")
    df, cos_df, split_df = analyse(datasets, caches, preps, fmts, primary)
    if le is not None:
        crosscheck_letter_effect(df, le, le_label)
    else:
        log(f"  cross-check vs {le_label}: skipped (file not found)")
    df.to_csv(out_dir / 'letter_transfer.csv', index=False)
    cos_df.to_csv(out_dir / 'letter_transfer_cos_matrix.csv', index=False)
    split_df.to_csv(out_dir / 'letter_transfer_splithalf.csv', index=False)

    log(f"PRIMARY LAYERS {primary}")
    cols = ['dataset', 'layer', 'rbar_none', 'rbar_oracle', 'rbar_oracle_crossfit', 'rbar_transfer', 'rbar_gain_oracle', 'recovered',
            'recovered_crossfit', 'cos_transfer', 'ceiling', 'cos_ratio', 'null_cos_p97_5', 'residual_mean_gap',
            'letter_var_before', 'letter_var_after', 'recovered_concat']
    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.float_format', '{:.4f}'.format):
        log(df[df['layer'].isin(primary)][cols].to_string(index=False))
    log(f"cosine matrix at layer {args.heatmap_layer}")
    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        log(cos_df[cos_df['layer'] == args.heatmap_layer].pivot(index='ds_i', columns='ds_j', values='cos')
            .loc[datasets, datasets].to_string())

    verdict, label = build_verdict(df, datasets, primary, fmts)
    log("=" * 100)
    log(verdict)
    (out_dir / 'letter_transfer_verdict.txt').write_text(verdict + "\n", encoding='utf-8')

    header = f"{Path(model).name} | {args.n_splits} splits, {args.n_perm} permutations, seed {args.seed}"
    plot_heatmap(cos_df, datasets, args.heatmap_layer, header, out_dir / 'letter_transfer_cos_heatmap.png')
    plot_cos_grid(df, datasets, primary, args.heatmap_layer, header, out_dir / 'letter_transfer_cos_by_layer.png')
    plot_rbar_grid(df, datasets, primary, args.heatmap_layer, header, out_dir / 'letter_transfer_rbar_by_layer.png')
    log("=" * 100)
    log(f"wrote {', '.join(str(out_dir / f) for f in ('letter_transfer.csv', 'letter_transfer_cos_matrix.csv', 'letter_transfer_splithalf.csv', 'letter_transfer_verdict.txt', 'letter_transfer_cos_heatmap.png', 'letter_transfer_cos_by_layer.png', 'letter_transfer_rbar_by_layer.png'))}")
    return df, label


# --------------------------------------------------------------------------------------
# Self-test: synthetic caches in memory, through the same pipeline (step 0 included)
# --------------------------------------------------------------------------------------
# name: (n, {letter: count}), counts over the cached rows
SELFTEST_SPEC = {
    'sycophancy': (400, {'(A)': 200, '(B)': 200}),
    'hallucination': (380, {'(A)': 187, '(B)': 193}),
    'refusal': (340, {'(A)': 160, '(B)': 180}),
    'myopic-reward': (360, {'(A)': 180, '(B)': 180}),
    'survival-instinct': (400, {'(A)': 180, '(B)': 150, '(C)': 40, '(D)': 20, '(E)': 10}),   # extra letters
    'coordinate-other-ais': (356, {'(A)': 230, '(B)': 126}),                                 # unbalanced
    'corrigible-neutral-HHH': (300, {'(A)': 150, '(B)': 150}),
    'tofu-forget10': (400, {'(A)': 200, '(B)': 200}),
}
SELFTEST_HIDDEN, SELFTEST_LAYERS = 256, 12
# Letter amplitude chosen so the letter carries ~2/3 of ‖d‖² (real layer-8 caches: cos same letter ≈ 0.72-0.80).
# Known property of recovered, visible here: with unbalanced letters the mean carries a letter part (m·u) next to
# the behaviour part, so (m·v)² moves at FIRST order in the angle between v and u and the in-sample oracle can beat
# even the population direction u. In the 230/126 dataset projecting the true u gives recovered ≈ 0.90 on some
# layers (0.66 with a = 12). The self-test therefore compares the transfer against the population direction instead
# of asserting 'transfiere'. rbar_gain_oracle (the denominator) is in the csv for the same reason.
SELFTEST_LETTER, SELFTEST_BEHAVIOUR, SELFTEST_NOISE = 24.0, 6.0, 1.0


def synthetic_inputs(shared_letter: bool, seed: int) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """d_i = b_k + (e_pos - e_neg) + noise, e_A = +a u/2, e_B = -a u/2 (so A/B rows carry s_i a u, s_i = ±1),
    extra letters e_X = a w_X (random). u is one per layer (shared) or one per layer and dataset (not shared)."""
    import torch
    from test_letter_effect import analyse_dataset
    rng = np.random.default_rng([seed, 99])
    h, L = SELFTEST_HIDDEN, SELFTEST_LAYERS
    u_shared = unit(rng.standard_normal((L, h)))
    caches, abs_, csv_rows, le_rows = {}, {}, [], []
    for k, (name, (n, counts)) in enumerate(SELFTEST_SPEC.items()):
        assert sum(counts.values()) == n
        total, cache_seed = n + 25, 7
        perm = shuffle_permutation(total, n, cache_seed)
        letters = rng.permutation(np.concatenate([[x] * c for x, c in counts.items()]))
        pos_json = rng.choice(['(A)', '(B)'], size=total).astype(object)
        pos_json[perm] = letters
        sep = FMT_NOSPACE if name in (OWN_DATASET, 'refusal') else FMT_SPACE    # mirrors the real jsons
        ab = [{'question': f"[{name}] synthetic question {j}?{sep} yes{sep.replace('A', 'B')} no",
               'answer_matching_behavior': p, 'answer_not_matching_behavior': '(B)' if p == '(A)' else '(A)'}
              for j, p in enumerate(pos_json)]
        u = u_shared if shared_letter else unit(rng.standard_normal((L, h)))
        b = SELFTEST_BEHAVIOUR * unit(rng.standard_normal((L, h)))
        extra = {x: unit(rng.standard_normal((L, h))) for x in counts if x not in ('(A)', '(B)')}
        diffs = {}
        for l in range(L):
            e = {'(A)': SELFTEST_LETTER * u[l] / 2, '(B)': -SELFTEST_LETTER * u[l] / 2}
            e.update({x: SELFTEST_LETTER * w[l] for x, w in extra.items()})
            letter_part = np.stack([e[p] - e['(B)' if p == '(A)' else '(A)'] for p in letters])
            d = b[l] + letter_part + SELFTEST_NOISE * rng.standard_normal((n, h))
            diffs[l] = torch.from_numpy(d.astype(np.float32))
        q0, first = ab[perm[0]]['question'], ab[perm[0]]
        cache = {'model': 'synthetic/selftest', 'dataset': name, 'n': n, 'total': total, 'seed': cache_seed,
                 'read_token_index': -2, 'hidden': h, 'layers': list(range(L)), 'diffs': diffs,
                 'example_pos': f"<|user|>{q0}<|assistant|>{first['answer_matching_behavior']}",
                 'example_neg': f"<|user|>{q0}<|assistant|>{first['answer_not_matching_behavior']}",
                 '_u_population': u}   # selftest only: the true letter direction per layer (not in real caches)
        caches[name], abs_[name] = cache, ab
        for l in range(L):
            rbar, nm, mn = rbar_float32_as_measure_rbar(diffs[l])
            csv_rows.append({'model': cache['model'], 'dataset': name, 'layer': l, 'n': n, 'rbar': rbar})
        pos_c = np.array([ab[j]['answer_matching_behavior'] for j in perm])
        used = [x for x in sorted(counts) if counts[x] >= MIN_LETTER_N]
        rows, _ = analyse_dataset(name, cache, pos_c, used, n_control=3, seed=0)   # test_letter_effect, independently
        le_rows.extend(rows)
    return caches, abs_, pd.DataFrame(csv_rows), pd.DataFrame(le_rows)


def recovered_population(df: pd.DataFrame, caches: dict) -> pd.Series:
    """recovered obtained by projecting the TRUE letter direction u (selftest only): the best any transfer can do."""
    out = []
    for _, r in df.iterrows():
        cache = caches[r['dataset']]
        d = cache['diffs'][r['layer']].numpy().astype(np.float64)
        u = cache['_u_population'][r['layer']]
        out.append(recovered_of(rbar_of(project_out(d, u[None])), r['rbar_none'], r['rbar_oracle']))
    return pd.Series(out, index=df.index)


def selftest(args) -> None:
    t0 = time.time()
    out_root = Path(args.out_dir) / 'selftest'
    datasets = list(SELFTEST_SPEC)
    psa = [x for x in datasets if x in PSA_DATASETS]
    args.primary_layers = [l for l in args.primary_layers if l < SELFTEST_LAYERS]
    for case, shared in (('positive', True), ('negative', False)):
        log("#" * 100)
        log(f"SELFTEST {case}: letter direction {'shared by' if shared else 'different in'} every dataset")
        caches, abs_, csv, le = synthetic_inputs(shared, args.seed)
        df, label = run_pipeline(datasets, caches, abs_, csv, 'synthetic rbar csv', le, 'synthetic letter_effect csv',
                                 args, out_root / case)
        prim = df[df['layer'].isin(args.primary_layers)]
        loo = df[df['role'] == ROLE_LOO]
        tofu = df[df['dataset'] == OWN_DATASET]
        # identities that must hold whatever the data
        assert set(loo['dataset']) == set(psa) and (loo['pool_size'] == len(psa) - 1).all()
        assert (tofu['role'] == ROLE_DIAG).all() and (tofu['pool_size'] == len(psa)).all()
        assert not df['pool_datasets'].str.contains(OWN_DATASET).any(), "tofu entered a pool"
        if case == 'positive':
            rec_pop = recovered_population(df, caches)
            gap_pop = (df['recovered'] - rec_pop).abs()
            log(f"  recovered of the transfer vs of the population direction u: max |diff| {gap_pop.max():.3f}; "
                f"population-u recovered min {rec_pop.min():.3f} ({df.loc[rec_pop.idxmin(), 'dataset']} "
                f"L{int(df.loc[rec_pop.idxmin(), 'layer'])})")
            assert gap_pop.max() < 0.05, gap_pop.describe()            # transfer as good as the true direction
            balanced = df['dataset'] != 'coordinate-other-ais'
            assert df.loc[balanced, 'recovered'].between(RECOVERED_MIN, 1.10).all(), df.loc[balanced, 'recovered'].describe()
            assert df.loc[balanced, 'recovered_crossfit'].between(RECOVERED_MIN, 1.10).all()
            assert df['recovered'].between(0.80, 1.20).all(), df['recovered'].describe()
            fails = prim[(prim['role'] == ROLE_LOO) & ~prim['passes']]
            assert label in ('transfiere', 'parcial'), label
            # any failure must be one the population direction itself (almost) fails: a property of the metric
            assert (rec_pop[fails.index] < RECOVERED_MIN + 0.05).all(), fails[['dataset', 'layer', 'recovered']]
            if len(fails):
                log(f"  NOTE: verdict '{label}' only because of {', '.join(f'{r.dataset} L{r.layer}' for r in fails.itertuples())}, "
                    f"where projecting the TRUE u gives recovered {', '.join(f'{rec_pop[i]:.3f}' for i in fails.index)}")
            assert loo['cos_ratio'].min() > 0.95 and loo['cos_ratio'].max() < 1.05, loo['cos_ratio'].describe()
            assert (df['cos_transfer'] > df['null_cos_max']).all()
            assert (df['residual_mean_gap'] < 0.2).all() and (df['letter_var_after'] < 0.01).all()
            assert (df['pool_min_member_cos'] > 0.9).all()
        else:
            assert label == 'no transfiere', label
            assert loo['recovered'].abs().max() < 0.10, loo['recovered'].describe()
            assert tofu['recovered'].abs().max() < 0.10, tofu['recovered'].describe()
            assert df['cos_transfer'].abs().max() < 0.35, df['cos_transfer'].describe()
            assert df['cos_above_null_p97_5'].mean() <= 0.15, df['cos_above_null_p97_5'].mean()
            assert (df['residual_mean_gap'] > 0.9).all()
        # every A/B-only dataset was cross-checked against test_letter_effect; a 1e-3 shift must stop the run
        bad = le.copy()
        bad.loc[bad['dataset'] == 'sycophancy', 'rbar_projected'] += 1e-3
        try:
            crosscheck_letter_effect(df, bad, 'corrupted synthetic letter_effect csv')
        except SystemExit as e:
            log(f"  (expected) corrupted cross-check stopped the run: {str(e)[:90]}...")
        else:
            raise AssertionError("the corrupted letter_effect cross-check did not stop the run")
        log(f"SELFTEST {case}: OK | verdict '{label}' | primary-layer recovered PSA median {prim[prim['role'] == ROLE_LOO]['recovered'].median():.3f}, "
            f"tofu median {prim[prim['dataset'] == OWN_DATASET]['recovered'].median():.3f} | cos_transfer median "
            f"{prim['cos_transfer'].median():.3f}, null p97.5 median {prim['null_cos_p97_5'].median():.3f}")
    # the alignment step must still refuse a wrong mapping
    caches, abs_, csv, _ = synthetic_inputs(True, args.seed)
    caches['sycophancy']['seed'] = 8
    try:
        step0('sycophancy', caches['sycophancy'], abs_['sycophancy'], csv, 'synthetic rbar csv')
    except SystemExit as e:
        log(f"(expected) wrong shuffle seed stopped step 0: {str(e)[:90]}...")
    else:
        raise AssertionError("step 0 accepted a wrong shuffle seed")
    log("#" * 100)
    log(f"SELFTEST PASSED in {time.time() - t0:.1f} s (outputs in {out_root})")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diffs-dir", default="sweep_1b/diffs", help="directory with the sweep_epsilon.py caches <dataset>.pt")
    parser.add_argument("--datasets", nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument("--rbar-csv", default="resultados_rbar_1b/rbar_results.csv", help="measure_rbar.py output (mandatory cross-check)")
    parser.add_argument("--letter-effect-csv", default="letter_effect/letter_effect.csv",
                        help="test_letter_effect.py output (optional cross-check, used when the file exists)")
    parser.add_argument("--primary-layers", type=int, nargs='+', default=[7, 8, 9, 10, 11])
    parser.add_argument("--n-splits", type=int, default=200, help="split-half partitions per dataset")
    parser.add_argument("--n-perm", type=int, default=200, help="letter permutations per pool dataset")
    parser.add_argument("--seed", type=int, default=0, help="seed of the splits and permutations")
    parser.add_argument("--heatmap-layer", type=int, default=8, help="layer of the cosine heatmap and of the plot notes")
    parser.add_argument("--out-dir", default="letter_transfer")
    parser.add_argument("--selftest", action='store_true', help="run on synthetic caches in memory (no input files) and assert")
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
        return

    csv = load_rbar_csv(Path(args.rbar_csv))
    le_path = Path(args.letter_effect_csv)
    le = pd.read_csv(le_path) if le_path.is_file() else None
    caches = {x: load_cache(Path(args.diffs_dir) / f"{x}.pt") for x in args.datasets}
    abs_ = {x: load_dataset_json(x) for x in args.datasets}
    run_pipeline(args.datasets, caches, abs_, csv, args.rbar_csv, le, args.letter_effect_csv, args, Path(args.out_dir))


if __name__ == '__main__':
    main()
