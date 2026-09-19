"""
Do the shared refusal templates of tofu-forget10 inflate R̄?

The positive option (c_pos) of every tofu-forget10 pair is one of K fixed refusal templates
(build_pairs.py TEMPLATES), while the negative option is a distinct biography answer. If the
coherence of the difference vectors d_i = h_pos_i - h_neg_i comes from the positive branch taking
only K values, R̄ = ‖mean d‖ / mean ‖d‖ is inflated by the dataset construction rather than by the
task. This script tests that on the cached diffs, without GPU and without loading the model.

Inputs
  --diffs         sweep_epsilon.py cache with d_i^l for all layers (sweep_1b/diffs/tofu-forget10.pt)
  --meta          pares/to_caa_format.py metadata, one line per A/B pair in file order
                  (row_idx, author_id, question_idx, template_id, pos_letter, len_pos_chars, len_neg_chars)
  --dataset-json  the A/B json the cache was extracted from (datasets/generate/tofu-forget10/generate_dataset.json)
  --rbar-csv      measure_rbar.py output, mandatory cross-check that the cache is what was measured

Step 0 — index alignment (the only place this analysis can go silently wrong)
  The cached rows are NOT in json order: measure_rbar.load_pairs (measure_rbar.py:129-139, also used
  by sweep_epsilon.extract_diffs at sweep_epsilon.py:170) does
        data = json.load(...); random.Random(seed).shuffle(data); data[:n_samples]
  and steering_vectors.extract_activations keeps that order (batchify yields data[i:i+batch_size]
  sequentially, utils.py:9-23; torch.concat(pos_acts[layer]) concatenates in call order,
  sweep_epsilon.py:203-207). random.shuffle only calls randbelow(i+1) for i = len-1 .. 1, so it depends
  on len(data) and the seed only: shuffling range(total) with the same seed reproduces the permutation
  exactly. The mapping is then verified independently: the cache stores the formatted prompt of the
  first shuffled pair (example_pos / example_neg), which must contain the question of json[perm[0]]
  and no other question, and must end in that pair's answer letters. R̄ recomputed from the cache with
  measure_rbar's float32 arithmetic must match the csv to 1e-4 on every layer. Any failure stops the run.

Analysis 1 — pairwise-cosine decomposition (sample-size free)
  u_i = d_i/‖d_i‖. Identity: ‖mean u‖² = mean_{i,j} cos(u_i,u_j) (i=j included). The off-diagonal pairs
  split into same-template (intra) and different-template (inter):
        cos_all = w_intra·cos_intra + (1-w_intra)·cos_inter,   w_intra = #intra pairs / #pairs
  The share of cos_all due to the intra excess is  w_intra·(cos_intra - cos_inter) / cos_all.
  The same identity holds for R̄ itself with weights w_i = ‖d_i‖/mean‖d‖:  R̄² = (1/n²) Σ_ij w_i w_j cos_ij,
  which gives an exact counterfactual R̄: the value R̄ would take if every intra pair were as coherent as
  the (norm-weighted) average inter pair. Both identities are asserted numerically.

Analysis 2 — R̄ per template vs random subsets of the same size
  R̄ is biased upwards for small n, so R̄ over the ~n/K pairs of one template is compared with R̄ over
  random subsets of the same size (templates mixed), --n-control repetitions, percentile 95% interval.

Analysis 3 — positive branch alone: NOT POSSIBLE with this cache. sweep_epsilon.extract_diffs stores
  (pos - neg) only (sweep_epsilon.py:207); h_pos and h_neg are not recoverable from the difference.
  The function is implemented and runs only if the cache carries 'pos' and 'neg' dicts (layer -> (n,h)).

Analysis 4 — confounds
  Length: correlation between cos(d_i, mean_{j≠i} d_j) (leave-one-out mean, so d_i is not correlated
  with itself) and len_pos_chars - len_neg_chars (also each length alone: len_pos is a function of the
  template, len_neg is the biography length).
  Letter: mean cosine in the 2x2 {intra, inter} x {same pos_letter, different pos_letter}. d_i is
  ±(h_A - h_B) with the sign set by pos_letter, so a letter component makes same-letter pairs more
  similar and different-letter pairs less similar; it cancels in the mean if the letters are balanced.

Outputs (in --out-dir)
  template_effect.csv, template_effect_by_template.csv, template_effect_letter.csv,
  template_effect_confounds.csv, template_effect.png, template_effect_verdict.txt

    python test_template_effect.py
    python test_template_effect.py --diffs sweep_1b/diffs/tofu-forget10.pt --layer 8 --out-dir template_effect
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

# Separator of the A/B questions written by pares/to_caa_format.py:41-43 (CAA format).
CHOICES_SEP = "\n\nChoices:\n"
OPTION_A = "(A) "
OPTION_B = "\n(B) "

RBAR_TOL = 1e-4          # cross-check tolerance vs rbar_results.csv (same as sweep_epsilon.crosscheck_rbar)
IDENTITY_TOL = 1e-9      # numerical tolerance for the exact decomposition identities (float64)
CI_PERCENTILES = (2.5, 97.5)
# Verdict thresholds (printed with the verdict). The counterfactual R̄ is what R̄ would be if the intra-
# template pairs were as coherent as the inter-template pairs; a relative drop below SUBSTANTIAL_DROP is
# "detectable but small".
SUBSTANTIAL_DROP = 0.10

# Palette: categorical slots 1 (blue) and 2 (orange) of the dataviz reference palette, same hexes as
# measure_rbar.SERIES_COLORS; black for the global R̄ as in measure_rbar's plots.
C_INTRA, C_INTER, C_ALL, C_GLOBAL, C_CONTROL, C_GRID = '#2a78d6', '#eb6834', '#7a7975', '#0b0b0b', '#b5b4ae', '#e5e4e0'
C_TEXT, C_TEXT2 = '#0b0b0b', '#52514e'


def log(msg: str = '') -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# Step 0: inputs and index alignment
# --------------------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    # weights_only=True is enough: the cache holds tensors, dicts, lists, ints and strs (sweep_epsilon.py:219).
    cache = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('model', 'dataset', 'n', 'total', 'seed', 'read_token_index', 'hidden', 'layers', 'diffs',
                'example_pos', 'example_neg'):
        if key not in cache:
            raise SystemExit(f"cache {path} has no key {key!r}; it is not a sweep_epsilon.py diffs cache")
    n, h = cache['n'], cache['hidden']
    for layer in cache['layers']:
        d = cache['diffs'][layer]
        if tuple(d.shape) != (n, h) or d.dtype != torch.float32:
            raise SystemExit(f"layer {layer}: diffs are {tuple(d.shape)} {d.dtype}, expected ({n}, {h}) float32")
    return cache


def shuffle_permutation(total: int, n: int, seed: int) -> list[int]:
    """perm[k] = json row index of cached row k. Replicates measure_rbar.load_pairs (measure_rbar.py:134,
    139): random.Random(seed).shuffle over a list of length `total`, then the first n."""
    idx = list(range(total))
    random.Random(seed).shuffle(idx)
    return idx[:n]


def split_options(question: str) -> tuple[str, str]:
    """Inverse of to_caa_format.build_question: returns (text of (A), text of (B))."""
    assert question.count(CHOICES_SEP + OPTION_A) == 1 and question.count(OPTION_B) == 1, \
        f"question does not have the CAA layout: {question[-120:]!r}"
    options = question.split(CHOICES_SEP + OPTION_A, 1)[1]
    text_a, text_b = options.split(OPTION_B, 1)
    return text_a, text_b


def verify_alignment(cache: dict, ab: list[dict], meta: list[dict], perm: list[int]) -> None:
    problems = []
    total, n = cache['total'], cache['n']
    if len(ab) != total:
        problems.append(f"dataset json has {len(ab)} pairs but the cache says total={total}")
    if len(meta) != len(ab):
        problems.append(f"meta has {len(meta)} rows but the dataset json has {len(ab)}")
    if len(perm) != n or len(set(perm)) != n:
        problems.append(f"permutation has {len(perm)} entries ({len(set(perm))} distinct), expected n={n}")
    if problems:
        raise SystemExit("index alignment FAILED before the content checks: " + "; ".join(problems))

    # meta <-> json consistency, in file order (to_caa_format.py:102-123 writes both from the same loop)
    pos_text_by_template: dict[int, set[str]] = {}
    for i, (row, m) in enumerate(zip(ab, meta)):
        if m['row_idx'] != i:
            problems.append(f"meta line {i} has row_idx={m['row_idx']}")
            break
        if row['answer_matching_behavior'] != m['pos_letter']:
            problems.append(f"row {i}: json answer_matching_behavior={row['answer_matching_behavior']!r} "
                            f"!= meta pos_letter={m['pos_letter']!r}")
            break
        text_a, text_b = split_options(row['question'])
        pos_text, neg_text = (text_a, text_b) if m['pos_letter'] == '(A)' else (text_b, text_a)
        if len(pos_text) != m['len_pos_chars'] or len(neg_text) != m['len_neg_chars']:
            problems.append(f"row {i}: option lengths in the json ({len(pos_text)}, {len(neg_text)}) != meta "
                            f"({m['len_pos_chars']}, {m['len_neg_chars']})")
            break
        pos_text_by_template.setdefault(m['template_id'], set()).add(pos_text)
    # template_id must label the c_pos text of the json one-to-one (independent of build_pairs.TEMPLATES)
    multi = {t: len(v) for t, v in pos_text_by_template.items() if len(v) != 1}
    if multi:
        problems.append(f"meta template_id does not identify a single c_pos text in the json: {multi}")
    texts = [next(iter(v)) for v in pos_text_by_template.values()]
    if len(set(texts)) != len(texts):
        problems.append("two meta template_ids share the same c_pos text in the json")
    if not problems:
        log(f"  meta <-> json OK on {len(ab)} rows: row_idx, pos_letter, option lengths; template_id <-> c_pos text is one-to-one:")
        for t in sorted(pos_text_by_template):
            log(f"    template {t:2d}: {next(iter(pos_text_by_template[t]))[:80]!r}")

    # Independent check of the permutation: the cache stores the formatted prompts of cached row 0
    # (sweep_epsilon.py:211, pairs[0]); the Llama-3 chat template embeds the question verbatim (trimmed)
    # and utils._format_llama3 asserts the string ends exactly in the answer, e.g. "(A)".
    first = ab[perm[0]]
    q0 = first['question'].strip()
    ex_pos, ex_neg = cache['example_pos'], cache['example_neg']
    if q0 not in ex_pos or q0 not in ex_neg:
        problems.append(f"question of json row perm[0]={perm[0]} is not in the cache's example prompts: "
                        f"the shuffle was NOT reproduced")
    else:
        hits = [j for j, row in enumerate(ab) if row['question'].strip() in ex_pos]
        if hits != [perm[0]]:
            problems.append(f"example_pos matches json rows {hits}, expected exactly [{perm[0]}]")
    if not ex_pos.endswith(first['answer_matching_behavior']):
        problems.append(f"example_pos ends in {ex_pos[-6:]!r}, expected {first['answer_matching_behavior']!r}")
    if not ex_neg.endswith(first['answer_not_matching_behavior']):
        problems.append(f"example_neg ends in {ex_neg[-6:]!r}, expected {first['answer_not_matching_behavior']!r}")

    if problems:
        raise SystemExit("index alignment FAILED: " + "; ".join(problems) +
                         ". Stopping: with a wrong mapping every number below would be plausible and wrong.")
    log(f"  alignment OK: cached row 0 = json row {perm[0]} (author {meta[perm[0]]['author_id']}, "
        f"question {meta[perm[0]]['question_idx']}, template {meta[perm[0]]['template_id']}, "
        f"pos_letter {meta[perm[0]]['pos_letter']}); example prompts contain that question only and end in "
        f"{first['answer_matching_behavior']} / {first['answer_not_matching_behavior']}")
    log(f"  example_pos tail: {ex_pos[-70:]!r}")


def rbar_float32_as_measure_rbar(d32: torch.Tensor) -> tuple[float, float, float]:
    """Same arithmetic as measure_rbar.RbarAggregator / sweep_epsilon.rbar_from_cache (float32 torch)."""
    mean_vec = torch.mean(d32, dim=0)
    norm_mean = torch.norm(mean_vec).item()
    mean_norm = torch.norm(d32, dim=1).mean().item()
    return norm_mean / mean_norm, norm_mean, mean_norm


def crosscheck_rbar(cache: dict, csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"rbar csv {csv_path} not found; the R̄ cross-check is mandatory (pass --rbar-csv)")
    csv = pd.read_csv(csv_path)
    ref = csv[(csv['model'] == cache['model']) & (csv['dataset'] == cache['dataset'])].set_index('layer')
    if ref.empty:
        raise SystemExit(f"no rows for model={cache['model']!r} dataset={cache['dataset']!r} in {csv_path}")
    worst, rows = 0.0, []
    for layer in cache['layers']:
        rbar, norm_mean, mean_norm = rbar_float32_as_measure_rbar(cache['diffs'][layer])
        if layer not in ref.index:
            raise SystemExit(f"layer {layer} is in the cache but not in {csv_path}")
        diff = abs(rbar - float(ref.loc[layer, 'rbar']))
        worst = max(worst, diff)
        if int(ref.loc[layer, 'n']) != cache['n']:
            raise SystemExit(f"layer {layer}: csv n={int(ref.loc[layer, 'n'])} != cache n={cache['n']}")
        rows.append({'layer': layer, 'rbar_cache_f32': rbar, 'rbar_csv': float(ref.loc[layer, 'rbar']), 'abs_diff': diff})
    if worst > RBAR_TOL:
        bad = [r for r in rows if r['abs_diff'] > RBAR_TOL]
        raise SystemExit(f"R̄ recomputed from the cache differs from {csv_path} by up to {worst:.2e} (> {RBAR_TOL:g}) "
                         f"on layers {[r['layer'] for r in bad]}. The cache is not what was measured; stopping.")
    log(f"  R̄ cross-check vs {csv_path} OK on {len(rows)} layers (max |diff| = {worst:.1e}, n = {cache['n']})")
    return csv


def check_template_counts(templates: np.ndarray, expected_k: int | None) -> np.ndarray:
    ids, counts = np.unique(templates, return_counts=True)
    n, k = len(templates), len(ids)
    log(f"  templates in the cached rows: {k} distinct ids {ids.tolist()}, counts {counts.tolist()} (n = {n}, n/k = {n / k:.1f})")
    if expected_k is not None and k != expected_k:
        log(f"  NOTE: --expected-templates says {expected_k} templates (~{n / expected_k:.0f} each) but the meta has "
            f"{k} (~{n / k:.0f} each). The analysis uses the {k} ids found in the meta; the intra share of pairs "
            f"is ~1/{k}, not ~1/{expected_k}.")
    lo, hi = math.floor(n / k) - 1, math.ceil(n / k) + 1
    if not (ids.tolist() == list(range(k))):
        log(f"  WARNING: template ids are not 0..{k - 1}: {ids.tolist()}")
    if counts.min() < lo or counts.max() > hi:
        log(f"  WARNING: template counts are not balanced (min {counts.min()}, max {counts.max()}, expected "
            f"{lo}..{hi}); the per-template R̄ control uses one random-subset distribution per group size.")
    return ids


# --------------------------------------------------------------------------------------
# R̄ helpers (float64)
# --------------------------------------------------------------------------------------
def rbar_of(d: np.ndarray) -> float:
    return float(np.linalg.norm(d.mean(axis=0)) / np.linalg.norm(d, axis=1).mean())


def rbar_u_of(d: np.ndarray) -> float:
    """‖mean u‖ with u_i = d_i/‖d_i‖ (the quantity the pairwise identity is exact for)."""
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.linalg.norm(u.mean(axis=0)))


# --------------------------------------------------------------------------------------
# Analysis 1: pairwise cosine decomposition
# --------------------------------------------------------------------------------------
def cosine_decomposition(d: np.ndarray, groups: np.ndarray) -> dict:
    n = d.shape[0]
    norms = np.linalg.norm(d, axis=1)
    U = d / norms[:, None]
    G = U @ U.T                                        # (n, n) cosines, float64
    off = ~np.eye(n, dtype=bool)
    same = groups[:, None] == groups[None, :]
    intra, inter = same & off, ~same
    assert intra.any() and inter.any(), "need at least one intra-template and one inter-template pair"
    cos_all, cos_intra, cos_inter = G[off].mean(), G[intra].mean(), G[inter].mean()
    w_intra = intra.sum() / off.sum()

    # identity 1: ‖mean u‖² = mean of all n² cosines (diagonal included)
    mean_u_sq = float(np.sum(U.mean(axis=0) ** 2))
    err1 = abs(mean_u_sq - G.mean())
    assert err1 < IDENTITY_TOL, f"pairwise identity violated: {err1:.2e}"
    # identity 2: cos_all is the pair-count weighted mean of cos_intra and cos_inter
    err2 = abs(cos_all - (w_intra * cos_intra + (1 - w_intra) * cos_inter))
    assert err2 < IDENTITY_TOL, f"intra/inter split does not add up: {err2:.2e}"
    rbar_u = math.sqrt(mean_u_sq)
    rbar_u_cf = math.sqrt(1.0 / n + (n - 1) / n * cos_inter)   # every off-diagonal pair inter-like

    # identity 3 (exact for R̄): R̄² = (1/n²) Σ_ij w_i w_j cos_ij with w_i = ‖d_i‖ / mean‖d‖
    w = norms / norms.mean()
    WG = np.outer(w, w) * G
    rbar_direct = rbar_of(d)
    err3 = abs(WG.sum() / n ** 2 - rbar_direct ** 2)
    assert err3 < IDENTITY_TOL * max(1.0, rbar_direct ** 2), f"weighted identity violated: {err3:.2e}"
    W = np.outer(w, w)
    cos_inter_w = WG[inter].sum() / W[inter].sum()
    cos_intra_w = WG[intra].sum() / W[intra].sum()
    rbar_cf = math.sqrt((WG[~off].sum() + WG[inter].sum() + W[intra].sum() * cos_inter_w) / n ** 2)

    frac = w_intra * (cos_intra - cos_inter) / cos_all
    return {
        'cos_intra': cos_intra, 'cos_inter': cos_inter, 'cos_all': cos_all, 'ratio': cos_intra / cos_inter,
        'w_intra': w_intra, 'frac_intra_excess': frac,
        'rbar_global': rbar_direct, 'rbar_counterfactual': rbar_cf,
        'rbar_u': rbar_u, 'rbar_u_counterfactual': rbar_u_cf,
        'cos_intra_weighted': cos_intra_w, 'cos_inter_weighted': cos_inter_w,
        'identity_err': max(err1, err2, err3),
        '_G': G, '_intra': intra, '_inter': inter, '_off': off,
    }


# --------------------------------------------------------------------------------------
# Analysis 2: R̄ per template vs random subsets of the same size
# --------------------------------------------------------------------------------------
def control_subsets(n: int, sizes: list[int], n_control: int, seed: int) -> dict[int, np.ndarray]:
    """One fixed set of random subsets per group size, shared by every layer (deterministic)."""
    rng = np.random.default_rng(seed)
    return {m: np.stack([rng.choice(n, size=m, replace=False) for _ in range(n_control)]) for m in sizes}


def rbar_by_group(d: np.ndarray, groups: np.ndarray, ids: np.ndarray, subsets: dict[int, np.ndarray],
                  G: np.ndarray, intra: np.ndarray) -> tuple[list[dict], dict]:
    norms = np.linalg.norm(d, axis=1)
    ctrl_by_size = {}
    for m, S in subsets.items():
        vals = np.empty(len(S))
        for r, idx in enumerate(S):
            vals[r] = np.linalg.norm(d[idx].mean(axis=0)) / norms[idx].mean()
        ctrl_by_size[m] = vals
    rows = []
    for t in ids:
        idx = np.flatnonzero(groups == t)
        m = len(idx)
        r_t = float(np.linalg.norm(d[idx].mean(axis=0)) / norms[idx].mean())
        ctrl = ctrl_by_size[m]
        lo, hi = np.percentile(ctrl, CI_PERCENTILES)
        sub_intra = intra[np.ix_(idx, idx)]
        rows.append({
            'template_id': int(t), 'n': m, 'rbar': r_t, 'rbar_u': rbar_u_of(d[idx]),
            'cos_within': float(G[np.ix_(idx, idx)][sub_intra].mean()),
            'control_mean': float(ctrl.mean()), 'control_ci_low': float(lo), 'control_ci_high': float(hi),
            'control_percentile': float((ctrl <= r_t).mean() * 100.0),
            'above_ci': bool(r_t > hi), 'below_ci': bool(r_t < lo),
        })
    return rows, ctrl_by_size


# --------------------------------------------------------------------------------------
# Analysis 3: branches alone (only if the cache carries pos and neg)
# --------------------------------------------------------------------------------------
def rbar_branches(pos: np.ndarray, neg: np.ndarray, groups: np.ndarray, ids: np.ndarray) -> dict:
    out = {'rbar_pos': rbar_of(pos), 'rbar_neg': rbar_of(neg)}
    for t in ids:
        idx = groups == t
        out[f'rbar_pos_t{int(t)}'] = rbar_of(pos[idx])
        out[f'rbar_neg_t{int(t)}'] = rbar_of(neg[idx])
    return out


# --------------------------------------------------------------------------------------
# Analysis 4: confounds
# --------------------------------------------------------------------------------------
def length_confound(d: np.ndarray, len_pos: np.ndarray, len_neg: np.ndarray) -> dict:
    n = d.shape[0]
    m = d.mean(axis=0)
    M_loo = (n * m[None, :] - d) / (n - 1)                       # leave-one-out mean for every i
    cos_i = (d * M_loo).sum(axis=1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(M_loo, axis=1))
    out = {}
    for label, x in (('dlen', len_pos - len_neg), ('len_pos', len_pos), ('len_neg', len_neg)):
        out[f'pearson_{label}'], out[f'pearson_{label}_p'] = map(float, pearsonr(cos_i, x))
        out[f'spearman_{label}'], out[f'spearman_{label}_p'] = map(float, spearmanr(cos_i, x))
    return out


def letter_table(G: np.ndarray, intra: np.ndarray, inter: np.ndarray, letters: np.ndarray) -> dict:
    same_letter = letters[:, None] == letters[None, :]
    out = {}
    for grp_name, grp in (('intra', intra), ('inter', inter), ('all', intra | inter)):
        for let_name, let in (('same_letter', same_letter), ('diff_letter', ~same_letter)):
            mask = grp & let
            out[f'cos_{grp_name}_{let_name}'] = float(G[mask].mean()) if mask.any() else float('nan')
            out[f'n_pairs_{grp_name}_{let_name}'] = int(mask.sum())
    return out


# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
def style_axis(ax, title: str, ylabel: str, layers: list[int]) -> None:
    ax.set_title(title, loc='left', fontsize=11, color=C_TEXT)
    ax.set_xlabel('layer index', color=C_TEXT2)
    ax.set_ylabel(ylabel, color=C_TEXT2)
    ax.set_xticks(layers)
    ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(C_GRID)


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', fontsize=8.5, color=color, va='center')


def plot(df: pd.DataFrame, dataset: str, model: str, n: int, k: int, m: int, peak: int, path: Path) -> None:
    layers = df['layer'].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor('#fcfcfb')
    for ax in (ax1, ax2):
        ax.set_facecolor('#fcfcfb')

    # (a) pairwise cosines by layer, one y axis
    ax1.plot(layers, df['cos_intra'], color=C_INTRA, linewidth=2, marker='o', markersize=5, label='cos_intra (same template)')
    ax1.plot(layers, df['cos_inter'], color=C_INTER, linewidth=2, marker='s', markersize=5, label='cos_inter (different template)')
    ax1.plot(layers, df['cos_all'], color=C_ALL, linewidth=1.5, linestyle='--', label='cos_all (every pair)')
    ax1.axvline(peak, color=C_GRID, linewidth=1.2, zorder=0)
    y_intra, y_inter = df['cos_intra'].iloc[-1], df['cos_inter'].iloc[-1]
    nudge = 5.0 if abs(y_intra - y_inter) < 0.04 * (df['cos_intra'].max() - min(df['cos_inter'].min(), 0.0)) else 0.0
    end_label(ax1, layers[-1], y_intra, f"intra {y_intra:.3f}", C_TEXT, dy=nudge)
    end_label(ax1, layers[-1], y_inter, f"inter {y_inter:.3f}", C_TEXT, dy=-nudge)
    r = df.set_index('layer')
    ax1.text(0.98, 0.97, f"layer {peak}: cos_intra / cos_inter = {r.loc[peak, 'ratio']:.2f}\n"
             f"intra excess = {100 * r.loc[peak, 'frac_intra_excess']:.0f}% of cos_all",
             transform=ax1.transAxes, fontsize=8.5, color=C_TEXT2, ha='right', va='top')
    style_axis(ax1, 'Mean pairwise cosine of the normalised diffs', 'mean cos(u_i, u_j), i ≠ j', layers)
    ax1.legend(frameon=False, fontsize=8.5, loc='upper left')

    # (b) R̄ by layer: global n, per-template mean (n/k) against random subsets of n/k, and the counterfactual
    ax2.fill_between(layers, df['rbar_control_ci_low'], df['rbar_control_ci_high'], color=C_CONTROL, alpha=0.45,
                     linewidth=0, label=f'random subsets of {m} pairs (95% interval)')
    ax2.plot(layers, df['rbar_control_mean'], color=C_ALL, linewidth=1.5, label=f'random subsets of {m}: mean')
    ax2.plot(layers, df['rbar_intra_mean'], color=C_INTRA, linewidth=2, marker='o', markersize=5,
             label=f'R̄ within one template (mean of {k} groups of {m})')
    ax2.plot(layers, df['rbar_global'], color=C_GLOBAL, linewidth=2.4, marker='o', markersize=5, label=f'R̄ global (n={n})')
    ax2.plot(layers, df['rbar_counterfactual'], color=C_GLOBAL, linewidth=1.6, linestyle=':',
             label='R̄ counterfactual (intra pairs as coherent as inter)')
    ax2.axvline(peak, color=C_GRID, linewidth=1.2, zorder=0)
    y_g, y_t = df['rbar_global'].iloc[-1], df['rbar_intra_mean'].iloc[-1]
    nudge = 5.0 if abs(y_g - y_t) < 0.04 else 0.0
    end_label(ax2, layers[-1], y_g, f"global {y_g:.3f}", C_TEXT, dy=-nudge if y_g < y_t else nudge)
    end_label(ax2, layers[-1], y_t, f"template {y_t:.3f}", C_TEXT, dy=nudge if y_g < y_t else -nudge)
    ax2.text(0.98, 0.03, f"layer {peak}: R̄ {r.loc[peak, 'rbar_global']:.3f} -> counterfactual {r.loc[peak, 'rbar_counterfactual']:.3f}",
             transform=ax2.transAxes, fontsize=8.5, color=C_TEXT2, ha='right', va='bottom')
    ax2.set_ylim(0, 1)
    style_axis(ax2, 'R̄ = ‖mean d‖ / mean ‖d‖', 'R̄', layers)
    ax2.legend(frameon=False, fontsize=8.5, loc='upper left')

    fig.suptitle(f"{dataset} — template effect on R̄ ({Path(model).name}, n={n}, {k} templates × {m})", x=0.01,
                 ha='left', fontsize=12, color=C_TEXT)
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diffs", default="sweep_1b/diffs/tofu-forget10.pt", help="sweep_epsilon.py diffs cache")
    parser.add_argument("--meta", default="../pares/tofu_forget10_ab_meta.jsonl", help="to_caa_format.py metadata jsonl")
    parser.add_argument("--dataset-json", default="datasets/generate/tofu-forget10/generate_dataset.json",
                        help="A/B json the cache was extracted from (measure_rbar.resolve_dataset_path)")
    parser.add_argument("--rbar-csv", default="resultados_rbar_1b/rbar_results.csv", help="measure_rbar.py output (mandatory cross-check)")
    parser.add_argument("--layer", type=int, default=None, help="layer for the verdict; default = argmax of R̄ global")
    parser.add_argument("--expected-templates", type=int, default=7, help="number of templates the request assumed (only for a note)")
    parser.add_argument("--n-control", type=int, default=200, help="random subsets per group size")
    parser.add_argument("--control-seed", type=int, default=0, help="numpy seed for the random subsets")
    parser.add_argument("--out-dir", default="template_effect")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- step 0 -------------------------------------------------------------------------
    log("=" * 100)
    log("STEP 0: inputs and index alignment")
    cache = load_cache(Path(args.diffs))
    n, total, seed, layers = cache['n'], cache['total'], cache['seed'], list(cache['layers'])
    log(f"  cache: {args.diffs} | model {cache['model']} | dataset {cache['dataset']} | n={n} of total={total} | "
        f"shuffle seed {seed} | read_token_index {cache['read_token_index']} | hidden {cache['hidden']} | "
        f"layers {layers[0]}..{layers[-1]} ({len(layers)})")
    if n != total:
        log(f"  NOTE: the cache keeps the first {n} of {total} shuffled pairs; template counts below are over those {n}.")
    with open(args.dataset_json, encoding='utf-8') as f:
        ab = json.load(f)
    with open(args.meta, encoding='utf-8') as f:
        meta = [json.loads(line) for line in f if line.strip()]
    perm = shuffle_permutation(total, n, seed)
    verify_alignment(cache, ab, meta, perm)
    csv = crosscheck_rbar(cache, Path(args.rbar_csv))

    meta_rows = [meta[j] for j in perm]                      # meta in cached-row order
    templates = np.array([m['template_id'] for m in meta_rows])
    letters = np.array([m['pos_letter'] for m in meta_rows])
    len_pos = np.array([m['len_pos_chars'] for m in meta_rows], dtype=float)
    len_neg = np.array([m['len_neg_chars'] for m in meta_rows], dtype=float)
    ids = check_template_counts(templates, args.expected_templates)
    k = len(ids)
    sizes = sorted(set(np.bincount(templates).tolist()) - {0})
    a_share = float((letters == '(A)').mean())
    log(f"  pos_letter balance in the cached rows: (A) {a_share:.1%} / (B) {1 - a_share:.1%}")

    # ---- analyses per layer ----------------------------------------------------------------
    log("=" * 100)
    log(f"ANALYSES on {len(layers)} layers (float64) | control: {args.n_control} random subsets per group size "
        f"{sizes}, seed {args.control_seed}")
    subsets = control_subsets(n, sizes, args.n_control, args.control_seed)
    has_branches = 'pos' in cache and 'neg' in cache
    main_rows, template_rows, letter_rows, confound_rows = [], [], [], []
    for layer in layers:
        d = cache['diffs'][layer].numpy().astype(np.float64)
        dec = cosine_decomposition(d, templates)
        G, intra, inter = dec['_G'], dec['_intra'], dec['_inter']
        t_rows, ctrl_by_size = rbar_by_group(d, templates, ids, subsets, G, intra)
        for r in t_rows:
            template_rows.append({'layer': layer, **r})
        rbar_intra = np.array([r['rbar'] for r in t_rows])
        # the control interval reported per layer is the one of the (first) group size; with balanced
        # templates there is exactly one size
        ctrl = ctrl_by_size[sizes[0]]
        lo, hi = np.percentile(ctrl, CI_PERCENTILES)
        row = {
            'layer': layer,
            'cos_intra': dec['cos_intra'], 'cos_inter': dec['cos_inter'], 'cos_all': dec['cos_all'], 'ratio': dec['ratio'],
            'rbar_global': dec['rbar_global'], 'rbar_intra_mean': float(rbar_intra.mean()),
            'rbar_control_mean': float(ctrl.mean()), 'rbar_control_ci_low': float(lo), 'rbar_control_ci_high': float(hi),
            # extra columns
            'rbar_intra_min': float(rbar_intra.min()), 'rbar_intra_max': float(rbar_intra.max()),
            'n_templates_above_ci': int(sum(r['above_ci'] for r in t_rows)),
            'n_templates_below_ci': int(sum(r['below_ci'] for r in t_rows)),
            'rbar_control_std': float(ctrl.std(ddof=1)),
            'w_intra': dec['w_intra'], 'frac_intra_excess': dec['frac_intra_excess'],
            'rbar_counterfactual': dec['rbar_counterfactual'],
            'rbar_u': dec['rbar_u'], 'rbar_u_counterfactual': dec['rbar_u_counterfactual'],
            'cos_intra_weighted': dec['cos_intra_weighted'], 'cos_inter_weighted': dec['cos_inter_weighted'],
            'identity_err': dec['identity_err'],
        }
        if has_branches:
            row.update(rbar_branches(cache['pos'][layer].numpy().astype(np.float64),
                                     cache['neg'][layer].numpy().astype(np.float64), templates, ids))
        main_rows.append(row)
        letter_rows.append({'layer': layer, **letter_table(G, intra, inter, letters)})
        confound_rows.append({'layer': layer, **length_confound(d, len_pos, len_neg)})
        log(f"  layer {layer:2d}: cos_intra {dec['cos_intra']:.4f} | cos_inter {dec['cos_inter']:.4f} | ratio {dec['ratio']:.2f} | "
            f"intra excess {100 * dec['frac_intra_excess']:5.1f}% of cos_all | R̄ {dec['rbar_global']:.4f} -> counterfactual "
            f"{dec['rbar_counterfactual']:.4f} | R̄ template mean {rbar_intra.mean():.4f} vs control {ctrl.mean():.4f} "
            f"[{lo:.4f}, {hi:.4f}] ({row['n_templates_above_ci']}/{k} above)")

    df = pd.DataFrame(main_rows)
    df_t = pd.DataFrame(template_rows)
    df_letter = pd.DataFrame(letter_rows)
    df_conf = pd.DataFrame(confound_rows)
    df.to_csv(out_dir / 'template_effect.csv', index=False)
    df_t.to_csv(out_dir / 'template_effect_by_template.csv', index=False)
    df_letter.to_csv(out_dir / 'template_effect_letter.csv', index=False)
    df_conf.to_csv(out_dir / 'template_effect_confounds.csv', index=False)

    if not has_branches:
        log("  ANALYSIS 3 SKIPPED: the cache holds (pos - neg) only (sweep_epsilon.extract_diffs, sweep_epsilon.py:207); "
            "h_pos and h_neg are not recoverable from the difference. Re-extract saving both branches to run it.")

    # ---- verdict layer --------------------------------------------------------------------
    peak = int(df.loc[df['rbar_global'].idxmax(), 'layer']) if args.layer is None else args.layer
    assert peak in layers, f"--layer {peak} not in the cache layers {layers}"
    r = df.set_index('layer').loc[peak]
    tl = df_t[df_t['layer'] == peak].sort_values('template_id')
    lt = df_letter.set_index('layer').loc[peak]
    cf = df_conf.set_index('layer').loc[peak]
    m = sizes[0]

    log("=" * 100)
    log(f"LAYER {peak} detail ({'argmax of R̄ global' if args.layer is None else 'from --layer'})")
    log(f"  R̄ per template (n={m} each) vs {args.n_control} random subsets of {m}: mean {r['rbar_control_mean']:.4f}, "
        f"95% interval [{r['rbar_control_ci_low']:.4f}, {r['rbar_control_ci_high']:.4f}]")
    for _, t in tl.iterrows():
        flag = 'ABOVE' if t['above_ci'] else ('below' if t['below_ci'] else 'inside')
        log(f"    template {int(t['template_id']):2d}: R̄ {t['rbar']:.4f} (percentile {t['control_percentile']:5.1f} of the control, {flag}) | "
            f"cos within template {t['cos_within']:.4f}")
    log(f"  letter (pos_letter) 2x2 at layer {peak}:")
    log(f"    intra-template: same letter {lt['cos_intra_same_letter']:.4f} ({int(lt['n_pairs_intra_same_letter'])} pairs) | "
        f"different letter {lt['cos_intra_diff_letter']:.4f} ({int(lt['n_pairs_intra_diff_letter'])} pairs)")
    log(f"    inter-template: same letter {lt['cos_inter_same_letter']:.4f} ({int(lt['n_pairs_inter_same_letter'])} pairs) | "
        f"different letter {lt['cos_inter_diff_letter']:.4f} ({int(lt['n_pairs_inter_diff_letter'])} pairs)")
    log(f"  length at layer {peak}: corr(cos(d_i, LOO mean), len_pos - len_neg): pearson {cf['pearson_dlen']:+.3f} (p={cf['pearson_dlen_p']:.2g}), "
        f"spearman {cf['spearman_dlen']:+.3f} (p={cf['spearman_dlen_p']:.2g}); "
        f"len_pos alone: pearson {cf['pearson_len_pos']:+.3f}; len_neg alone: pearson {cf['pearson_len_neg']:+.3f}")

    # other datasets at the same layer, for the ranking question
    others = csv[(csv['model'] == cache['model']) & (csv['dataset'] != cache['dataset']) & (csv['layer'] == peak)]
    best_other = None
    if len(others):
        b = others.loc[others['rbar'].idxmax()]
        best_other = (str(b['dataset']), float(b['rbar']))

    # ---- verdict ----------------------------------------------------------------------------
    drop = 1.0 - r['rbar_counterfactual'] / r['rbar_global']
    outside = (r['rbar_intra_mean'] > r['rbar_control_ci_high']) or (r['rbar_intra_mean'] < r['rbar_control_ci_low'])
    above = r['rbar_intra_mean'] > r['rbar_control_ci_high']
    rank_changes = best_other is not None and r['rbar_counterfactual'] < best_other[1] <= r['rbar_global']
    if drop >= SUBSTANTIAL_DROP or rank_changes:
        conclusion = "las plantillas inflan R̄ de forma sustancial"
    elif above:
        conclusion = "hay un efecto de plantilla detectable pero pequeño: no explica el R̄ alto"
    else:
        conclusion = "las plantillas no explican el R̄ alto"
    lines = [
        "VEREDICTO",
        f"  cos_intra / cos_inter en la capa {peak}: {r['ratio']:.2f}   (cos_intra {r['cos_intra']:.4f}, cos_inter {r['cos_inter']:.4f}, "
        f"cos_all {r['cos_all']:.4f}; {100 * r['w_intra']:.1f}% de los pares son intra-plantilla)",
        f"  fracción del coseno global atribuible al exceso intra-plantilla: {100 * r['frac_intra_excess']:.1f}%",
        f"  R̄ intra-plantilla vs control del mismo n: {'fuera' if outside else 'dentro'} del IC 95%   "
        f"(media de {k} plantillas de {m}: {r['rbar_intra_mean']:.4f}; control de {m} al azar: {r['rbar_control_mean']:.4f} "
        f"[{r['rbar_control_ci_low']:.4f}, {r['rbar_control_ci_high']:.4f}]; {int(r['n_templates_above_ci'])}/{k} plantillas por encima)",
        f"  R̄ global {r['rbar_global']:.4f} -> R̄ contrafactual (pares intra tan coherentes como los inter) {r['rbar_counterfactual']:.4f}: "
        f"caída de {100 * drop:.1f}%",
    ]
    if best_other is not None:
        was_top = r['rbar_global'] > best_other[1]
        is_top = r['rbar_counterfactual'] > best_other[1]
        lines.append(f"  ranking en la capa {peak}: mejor otro dataset = {best_other[0]} (R̄ {best_other[1]:.4f}); "
                     f"{cache['dataset']} global {r['rbar_global']:.4f} {'por encima' if was_top else 'por debajo'} -> "
                     f"contrafactual {r['rbar_counterfactual']:.4f} {'por encima' if is_top else 'por debajo'}"
                     f"{' (el ranking cambia)' if rank_changes else ''}")
    lines.append(f"  criterio: 'sustancial' si la caída contrafactual >= {100 * SUBSTANTIAL_DROP:.0f}% o si el ranking cambia; "
                 f"'detectable' si la media intra-plantilla supera el percentil {CI_PERCENTILES[1]} del control")
    if not has_branches:
        lines.append("  (análisis 3, rama positiva sola: no ejecutado, el cache solo tiene pos - neg)")
    lines.append(f"  => {conclusion}")
    verdict = "\n".join(lines)
    log("=" * 100)
    log(verdict)
    (out_dir / 'template_effect_verdict.txt').write_text(verdict + "\n", encoding='utf-8')

    plot(df, cache['dataset'], cache['model'], n, k, m, peak, out_dir / 'template_effect.png')
    log("=" * 100)
    log(f"wrote {out_dir / 'template_effect.csv'}, {out_dir / 'template_effect_by_template.csv'}, "
        f"{out_dir / 'template_effect_letter.csv'}, {out_dir / 'template_effect_confounds.csv'}, "
        f"{out_dir / 'template_effect.png'}, {out_dir / 'template_effect_verdict.txt'}")


if __name__ == '__main__':
    main()
