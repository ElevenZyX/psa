"""
Is the answer letter ("(A)" vs "(B)") the dominant direction of the difference vectors, and is that
specific to tofu-forget10 or a property of the CAA A/B protocol shared by the 7 PSA datasets?

Every pair is read at read_token_index=-2, i.e. at the answer letter token, and d_i = h(prompt, pos
letter) - h(prompt, neg letter). If the residual stream at "A" differs from the one at "B" by a large
fixed vector, every d_i carries ±(e_A - e_B) on top of the behaviour, with the sign set by which letter
is answer_matching_behavior. Then two d_i with the same letter have a large cosine, two with different
letters a small or negative one, and the mean over a letter-balanced dataset cancels the letter part
while every ‖d_i‖ keeps it: R̄ = ‖mean d‖ / mean ‖d‖ is deflated, uniformly for every dataset built
with the protocol. Note the sign: projecting the letter out shrinks the denominator and (for balanced
letters) barely touches the numerator, so R̄ is expected to RISE, not fall. The script reports the
signed relative change.

Inputs: the sweep_epsilon.py caches <diffs-dir>/<dataset>.pt for the 8 datasets, the A/B jsons
(datasets/generate/<dataset>/generate_dataset.json) for the letters, and measure_rbar.py's csv.
Step 0 (row alignment, independent verification, R̄ cross-check) is cache_alignment.py, shared with
test_template_effect.py; any dataset that cannot be verified stops the run.

Letter of a pair = answer_matching_behavior, treated as categorical. answer_not_matching_behavior is
checked to be a function of it (in the 8 datasets it is: "(A)" -> "(B)", anything else -> "(A)"), so
the category identifies the (pos, neg) letter combination and therefore the letter part of d_i.

Analysis 1  cos_same / cos_diff: mean cosine of the normalised d_i over pairs with the same / different
            letter (i ≠ j, float64), gap = cos_same - cos_diff, per dataset and layer.
Analysis 2  R̄ restricted to the pairs of one letter (every letter with >= --min-letter-n pairs), against
            R̄ over random subsets of the same size drawn from all pairs (letters mixed), --n-control
            draws, percentile 95% interval. rbar_letterA / rbar_letterB in the main csv, every letter
            in letter_effect_by_letter.csv.
Analysis 3  Letter subspace: with M_X = mean(d_i | letter X) for the letters with >= --min-letter-n
            pairs and X0 the most frequent letter, the subspace is span{M_X - M_X0}, orthonormalised
            by SVD (rank = #letters - 1; for two letters it is exactly v̂ = (M_A - M_B)/‖M_A - M_B‖,
            asserted). d_i^clean = d_i - Q Qᵀ d_i, rbar_projected = R̄ of the d_i^clean.
            var_explained_by_letter = variance of the centred d_i inside the subspace / total variance
            (trace of the covariance). mean_energy_in_letter = ‖Q Qᵀ mean d‖² / ‖mean d‖², the share of
            the numerator the projection removes (~0 when the letters are balanced).
Analysis 4  Ranking of the 8 datasets by rbar_projected at each dataset's own peak layer (argmax of
            rbar_global over layers), next to rbar_global.

Outputs (in --out-dir): letter_effect.csv, letter_effect_by_letter.csv, letter_effect_ranking.csv,
letter_effect_cos_by_layer.png, letter_effect_rbar_by_layer.png, letter_effect_scatter.png,
letter_effect_verdict.txt

    python test_letter_effect.py
    python test_letter_effect.py --diffs-dir sweep_1b/diffs --rbar-csv resultados_rbar_1b/rbar_results.csv --out-dir letter_effect
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cache_alignment import (DEFAULT_DATASETS, OWN_DATASET, PSA_DATASETS, align, load_cache, load_dataset_json,
                             load_rbar_csv, log)

CI_PERCENTILES = (2.5, 97.5)
# Verdict criteria (printed with the verdict): the letter effect is "present" in a dataset at its peak layer
# when R̄ restricted to letter A and to letter B both exceed the 97.5th percentile of same-size random subsets
# and the cosine gap is at least GAP_MIN.
GAP_MIN = 0.05

# Same colour slots as measure_rbar.SERIES_COLORS (dataviz categorical palette), tofu-forget10 in black.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
C_SAME, C_DIFF, C_A, C_B, C_GLOBAL, C_CONTROL, C_GRID = '#2a78d6', '#eb6834', '#2a78d6', '#eb6834', '#0b0b0b', '#b5b4ae', '#e5e4e0'
C_TEXT, C_TEXT2, C_SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


# --------------------------------------------------------------------------------------
# Letters
# --------------------------------------------------------------------------------------
def letters_of(ab: list[dict], perm: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([ab[j]['answer_matching_behavior'] for j in perm])
    neg = np.array([ab[j]['answer_not_matching_behavior'] for j in perm])
    return pos, neg


def describe_letters(name: str, pos: np.ndarray, neg: np.ndarray, min_n: int) -> list[str]:
    n = len(pos)
    counts = Counter(pos.tolist())
    log(f"  letters (answer_matching_behavior) in the {n} cached rows: "
        + ", ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in sorted(counts.items())))
    combos = Counter(zip(pos.tolist(), neg.tolist()))
    neg_of = {}
    for (p, q), c in sorted(combos.items()):
        neg_of.setdefault(p, set()).add(q)
    ambiguous = {p: sorted(q) for p, q in neg_of.items() if len(q) > 1}
    if ambiguous:
        log(f"  WARNING: answer_not_matching_behavior is not a function of the letter: {ambiguous}; the letter category "
            f"does not identify the (pos, neg) combination and the letter subspace below is only approximate.")
    elif len(combos) > 2:
        log(f"  (pos, neg) combinations: " + ", ".join(f"{p}->{q} {c}" for (p, q), c in sorted(combos.items())))
    used = [k for k, v in sorted(counts.items()) if v >= min_n]
    dropped = {k: v for k, v in sorted(counts.items()) if v < min_n}
    if dropped:
        log(f"  letters with < {min_n} pairs are kept in the data but get no per-letter R̄ and no subspace direction: {dropped}")
    if '(A)' not in used or '(B)' not in used:
        raise SystemExit(f"[{name}] letters (A) and (B) must both have >= {min_n} pairs; have {dict(counts)}")
    return used


# --------------------------------------------------------------------------------------
# Numerics (float64)
# --------------------------------------------------------------------------------------
def rbar_of(d: np.ndarray) -> float:
    return float(np.linalg.norm(d.mean(axis=0)) / np.linalg.norm(d, axis=1).mean())


def cosine_by_letter(d: np.ndarray, letters: np.ndarray) -> tuple[float, float]:
    U = d / np.linalg.norm(d, axis=1, keepdims=True)
    G = U @ U.T
    n = len(d)
    off = ~np.eye(n, dtype=bool)
    same = letters[:, None] == letters[None, :]
    return float(G[same & off].mean()), float(G[~same].mean())


def letter_subspace(d: np.ndarray, letters: np.ndarray, used: list[str]) -> tuple[np.ndarray, str]:
    """Orthonormal rows Q (r, h) spanning {M_X - M_X0 : X in used, X != X0}, X0 = most frequent used letter."""
    counts = {x: int((letters == x).sum()) for x in used}
    x0 = max(used, key=lambda x: (counts[x], x == '(A)'))
    M = {x: d[letters == x].mean(axis=0) for x in used}
    D = np.stack([M[x] - M[x0] for x in used if x != x0])          # (k-1, h)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    keep = sv > 1e-10 * sv.max()
    Q = Vt[keep]
    if len(used) == 2:  # the subspace is the single direction v̂ = (M_A - M_B)/‖·‖, up to sign
        v = M['(A)'] - M['(B)']
        v /= np.linalg.norm(v)
        assert Q.shape[0] == 1 and abs(abs(float(Q[0] @ v)) - 1.0) < 1e-10, "2-letter subspace is not (M_A - M_B)/‖·‖"
    return Q, x0


def project_out(d: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return d - (d @ Q.T) @ Q


def variance_shares(d: np.ndarray, Q: np.ndarray) -> tuple[float, float, float]:
    """(centred variance share inside the subspace, share of mean ‖d‖² inside it, share of ‖mean d‖² inside it)."""
    n = d.shape[0]
    m = d.mean(axis=0)
    dc = d - m
    total = float((dc ** 2).sum() / n)
    inside = float(((dc @ Q.T) ** 2).sum() / n)
    energy = float(((d @ Q.T) ** 2).sum() / (d ** 2).sum())
    mean_inside = float(((m @ Q.T) ** 2).sum() / (m ** 2).sum())
    return inside / total, energy, mean_inside


def control_subsets(n: int, sizes: list[int], n_control: int, seed: int) -> dict[int, np.ndarray]:
    """Fixed random subsets per size (drawn once per dataset, shared by every layer)."""
    rng = np.random.default_rng(seed)
    return {m: np.stack([rng.choice(n, size=m, replace=False) for _ in range(n_control)]) for m in sorted(set(sizes))}


def control_rbar(d: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(d, axis=1)
    return np.array([np.linalg.norm(d[idx].mean(axis=0)) / norms[idx].mean() for idx in subsets])


# --------------------------------------------------------------------------------------
# One dataset
# --------------------------------------------------------------------------------------
def analyse_dataset(name: str, cache: dict, pos: np.ndarray, used: list[str], n_control: int, seed: int
                    ) -> tuple[list[dict], list[dict]]:
    n = cache['n']
    n_by_letter = {x: int((pos == x).sum()) for x in used}
    subsets = control_subsets(n, list(n_by_letter.values()), n_control, seed)
    rows, letter_rows = [], []
    for layer in cache['layers']:
        d = cache['diffs'][layer].numpy().astype(np.float64)
        cos_same, cos_diff = cosine_by_letter(d, pos)
        rbar_global = rbar_of(d)
        Q, x0 = letter_subspace(d, pos, used)
        d_clean = project_out(d, Q)
        rbar_projected = rbar_of(d_clean)
        var_share, energy_share, mean_share = variance_shares(d, Q)
        ctrl = {m: control_rbar(d, S) for m, S in subsets.items()}
        per_letter = {}
        for x in used:
            r_x = rbar_of(d[pos == x])
            c = ctrl[n_by_letter[x]]
            lo, hi = np.percentile(c, CI_PERCENTILES)
            per_letter[x] = (r_x, float(c.mean()), float(lo), float(hi))
            letter_rows.append({'dataset': name, 'layer': layer, 'letter': x, 'n': n_by_letter[x], 'rbar': r_x,
                                'control_mean': float(c.mean()), 'control_ci_low': float(lo), 'control_ci_high': float(hi),
                                'control_percentile': float((c <= r_x).mean() * 100.0), 'above_ci': bool(r_x > hi)})
        rA, cA, loA, hiA = per_letter['(A)']
        rB, cB, loB, hiB = per_letter['(B)']
        rows.append({
            'dataset': name, 'layer': layer,
            'cos_same': cos_same, 'cos_diff': cos_diff, 'gap': cos_same - cos_diff,
            'rbar_global': rbar_global, 'rbar_letterA': rA, 'rbar_letterB': rB,
            'rbar_control_mean': cA, 'rbar_control_ci_low': loA, 'rbar_control_ci_high': hiA,   # subsets of n_A pairs
            'rbar_projected': rbar_projected, 'var_explained_by_letter': var_share,
            # extra columns
            'rbar_rel_change': rbar_projected / rbar_global - 1.0,
            'energy_in_letter': energy_share, 'mean_energy_in_letter': mean_share,
            'rbar_controlB_mean': cB, 'rbar_controlB_ci_low': loB, 'rbar_controlB_ci_high': hiB,  # subsets of n_B pairs
            'n': n, 'n_letterA': n_by_letter['(A)'], 'n_letterB': n_by_letter['(B)'],
            'letters_used': ' '.join(used), 'subspace_rank': int(Q.shape[0]), 'subspace_reference': x0,
        })
    return rows, letter_rows


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def style_axis(ax, title: str, layers: list[int] | None, ylabel: str | None = None, note: str | None = None) -> None:
    ax.set_title(title, loc='left', fontsize=10, color=C_TEXT)
    if note:  # numbers of the peak layer go in the title row, outside the plot area, so they never collide with the data
        ax.set_title(note, loc='right', fontsize=7.5, color=C_TEXT2)
    if layers is not None:
        ax.set_xticks(layers[::2] if len(layers) > 10 else layers)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(C_SURFACE)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_TEXT2, fontsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(C_GRID)


def grid_figure(datasets: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    ncol = 4
    nrow = -(-len(datasets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow), sharex=True, sharey=True)
    fig.patch.set_facecolor(C_SURFACE)
    axes = list(np.atleast_1d(axes).ravel())
    for ax in axes[len(datasets):]:
        ax.axis('off')
    return fig, axes[:len(datasets)]


def plot_cos(df: pd.DataFrame, peaks: dict[str, int], datasets: list[str], header: str, path: Path) -> None:
    layers = sorted(df['layer'].unique())
    fig, axes = grid_figure(datasets)
    for ax, name in zip(axes, datasets):
        sub = df[df['dataset'] == name].sort_values('layer')
        ax.axhline(0, color=C_GRID, linewidth=1)
        ax.axvline(peaks[name], color=C_GRID, linewidth=1.2, zorder=0)
        ax.plot(sub['layer'], sub['cos_same'], color=C_SAME, linewidth=2, marker='o', markersize=4, label='same letter')
        ax.plot(sub['layer'], sub['cos_diff'], color=C_DIFF, linewidth=2, marker='s', markersize=4, label='different letter')
        p = sub.set_index('layer').loc[peaks[name]]
        style_axis(ax, name if name != OWN_DATASET else f"{name} (own)", layers,
                   note=f"L{peaks[name]}: same {p['cos_same']:.2f}, diff {p['cos_diff']:.2f}, gap {p['gap']:.2f}")
    for ax in axes[-4:]:
        ax.set_xlabel('layer index', color=C_TEXT2, fontsize=9)
    axes[0].set_ylabel('mean cos(u_i, u_j), i ≠ j', color=C_TEXT2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='lower center', ncol=len(labels))
    fig.suptitle(f"Mean pairwise cosine of the normalised diffs by answer letter — {header}", x=0.01, ha='left',
                 fontsize=11, color=C_TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rbar(df: pd.DataFrame, peaks: dict[str, int], datasets: list[str], header: str, path: Path) -> None:
    layers = sorted(df['layer'].unique())
    fig, axes = grid_figure(datasets)
    for ax, name in zip(axes, datasets):
        sub = df[df['dataset'] == name].sort_values('layer')
        nA, nB = int(sub['n_letterA'].iloc[0]), int(sub['n_letterB'].iloc[0])
        ax.axvline(peaks[name], color=C_GRID, linewidth=1.2, zorder=0)
        ax.fill_between(sub['layer'], sub['rbar_control_ci_low'], sub['rbar_control_ci_high'], color=C_CONTROL, alpha=0.45,
                        linewidth=0, label=f'random subsets of n_A (95% interval)')
        if nA != nB:
            ax.fill_between(sub['layer'], sub['rbar_controlB_ci_low'], sub['rbar_controlB_ci_high'], color=C_CONTROL, alpha=0.3,
                            linewidth=0, label=f'random subsets of n_B (95% interval)')
        ax.plot(sub['layer'], sub['rbar_letterA'], color=C_A, linewidth=1.8, marker='o', markersize=4, label=f'R̄ letter (A) only')
        ax.plot(sub['layer'], sub['rbar_letterB'], color=C_B, linewidth=1.8, marker='s', markersize=4, label=f'R̄ letter (B) only')
        ax.plot(sub['layer'], sub['rbar_global'], color=C_GLOBAL, linewidth=2.2, marker='o', markersize=4, label='R̄ global')
        ax.plot(sub['layer'], sub['rbar_projected'], color=C_GLOBAL, linewidth=1.6, linestyle=':', label='R̄ with the letter projected out')
        p = sub.set_index('layer').loc[peaks[name]]
        style_axis(ax, name if name != OWN_DATASET else f"{name} (own)", layers,
                   note=f"n_A={nA}, n_B={nB}\nL{peaks[name]}: R̄ {p['rbar_global']:.2f} · A {p['rbar_letterA']:.2f} · "
                        f"B {p['rbar_letterB']:.2f} · proj {p['rbar_projected']:.2f}")
        ax.set_ylim(0, 1)
    for ax in axes[-4:]:
        ax.set_xlabel('layer index', color=C_TEXT2, fontsize=9)
    axes[0].set_ylabel('R̄', color=C_TEXT2, fontsize=9)
    handles, labels = {}, []
    for ax in axes:  # union of the labels (the n_B band only exists where n_A != n_B)
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in handles:
                handles[l_] = h_
                labels.append(l_)
    fig.legend([handles[l_] for l_ in labels], labels, frameon=False, fontsize=9, loc='lower center', ncol=len(labels))
    fig.suptitle(f"R̄ global, R̄ restricted to one letter, same-size control and R̄ after projecting the letter out — {header}",
                 x=0.01, ha='left', fontsize=11, color=C_TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scatter(rank: pd.DataFrame, header: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    fig.patch.set_facecolor(C_SURFACE)
    lim = float(max(rank['rbar_global'].max(), rank['rbar_projected'].max()) * 1.15)
    ax.plot([0, lim], [0, lim], color=C_CONTROL, linewidth=1.2, linestyle='--', zorder=0)
    ax.text(0.62 * lim, 0.62 * lim, 'no change', color=C_TEXT2, fontsize=8, ha='center', va='bottom', rotation=45)
    # label positions: greedy push-apart in y so that neighbouring labels never overlap
    order = rank.sort_values('rbar_projected').reset_index(drop=True)
    min_sep = 0.045 * lim
    ys = order['rbar_projected'].to_numpy(dtype=float).copy()
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    for (_, r), y_label in zip(order.iterrows(), ys):
        own = r['dataset'] == OWN_DATASET
        # colour follows the dataset (same slot as in measure_rbar's plots), not its rank
        slot = PSA_DATASETS.index(r['dataset']) if r['dataset'] in PSA_DATASETS else 0
        color = C_GLOBAL if own else SERIES_COLORS[slot % len(SERIES_COLORS)]
        ax.scatter(r['rbar_global'], r['rbar_projected'], s=70 if own else 50, color=color, zorder=3,
                   edgecolor=C_SURFACE, linewidth=1.5)
        ax.annotate(f"{r['dataset']} (L{int(r['peak_layer'])})  #{int(r['rank_global'])}→#{int(r['rank_projected'])}",
                    (r['rbar_global'], r['rbar_projected']), xytext=(r['rbar_global'] + 0.02 * lim, y_label),
                    textcoords='data', fontsize=8.5, color=C_TEXT, va='center', fontweight='bold' if own else 'normal',
                    arrowprops=dict(arrowstyle='-', color=C_CONTROL, linewidth=0.8, shrinkA=0, shrinkB=3)
                    if abs(y_label - r['rbar_projected']) > 1e-9 else None)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('R̄ global at the peak layer', color=C_TEXT2)
    ax.set_ylabel('R̄ with the letter subspace projected out', color=C_TEXT2)
    style_axis(ax, 'R̄ before and after removing the answer-letter direction (#rank before→after)', None)
    fig.suptitle(header, x=0.01, ha='left', fontsize=9, color=C_TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    parser.add_argument("--diffs-dir", default="sweep_1b/diffs", help="directory with the sweep_epsilon.py caches <dataset>.pt")
    parser.add_argument("--datasets", nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument("--rbar-csv", default="resultados_rbar_1b/rbar_results.csv", help="measure_rbar.py output (mandatory cross-check)")
    parser.add_argument("--n-control", type=int, default=200, help="random subsets per letter size")
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--min-letter-n", type=int, default=5,
                        help="a letter needs at least this many pairs to get a per-letter R̄ and a subspace direction")
    parser.add_argument("--out-dir", default="letter_effect")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = load_rbar_csv(Path(args.rbar_csv))

    all_rows, all_letter_rows, model = [], [], None
    log("=" * 100)
    for name in args.datasets:
        log(f"[{name}] STEP 0")
        cache = load_cache(Path(args.diffs_dir) / f"{name}.pt")
        if model is None:
            model = cache['model']
        elif cache['model'] != model:
            raise SystemExit(f"[{name}] cache model {cache['model']!r} != {model!r}: mixed models in {args.diffs_dir}")
        log(f"  n={cache['n']} of total={cache['total']} | shuffle seed {cache['seed']} | read_token_index {cache['read_token_index']} | "
            f"hidden {cache['hidden']} | {len(cache['layers'])} layers")
        ab = load_dataset_json(name)
        perm = align(cache, ab, csv, args.rbar_csv)
        pos, neg = letters_of(ab, perm)
        used = describe_letters(name, pos, neg, args.min_letter_n)
        rows, letter_rows = analyse_dataset(name, cache, pos, used, args.n_control, args.control_seed)
        all_rows.extend(rows)
        all_letter_rows.extend(letter_rows)
        peak = max(rows, key=lambda r: r['rbar_global'])
        log(f"  peak layer {peak['layer']}: cos_same {peak['cos_same']:.3f} | cos_diff {peak['cos_diff']:.3f} | gap {peak['gap']:.3f} | "
            f"R̄ {peak['rbar_global']:.3f} | R̄_A {peak['rbar_letterA']:.3f} R̄_B {peak['rbar_letterB']:.3f} vs control "
            f"[{peak['rbar_control_ci_low']:.3f}, {peak['rbar_control_ci_high']:.3f}] | R̄ projected {peak['rbar_projected']:.3f} "
            f"({100 * peak['rbar_rel_change']:+.1f}%) | letter var share {100 * peak['var_explained_by_letter']:.1f}% | "
            f"subspace rank {peak['subspace_rank']}")
        log("=" * 100)

    df = pd.DataFrame(all_rows)
    df_letter = pd.DataFrame(all_letter_rows)
    df.to_csv(out_dir / 'letter_effect.csv', index=False)
    df_letter.to_csv(out_dir / 'letter_effect_by_letter.csv', index=False)

    # ---- ranking at each dataset's own peak layer -------------------------------------------
    peaks = {name: int(df[df['dataset'] == name].sort_values('rbar_global').iloc[-1]['layer']) for name in args.datasets}
    rank = pd.DataFrame([df[(df['dataset'] == name) & (df['layer'] == peaks[name])].iloc[0] for name in args.datasets])
    rank = rank.rename(columns={'layer': 'peak_layer'})
    rank['letter_effect_present'] = ((rank['rbar_letterA'] > rank['rbar_control_ci_high'])
                                     & (rank['rbar_letterB'] > rank['rbar_controlB_ci_high'])
                                     & (rank['gap'] >= GAP_MIN))
    rank['rank_global'] = rank['rbar_global'].rank(ascending=False).astype(int)
    rank['rank_projected'] = rank['rbar_projected'].rank(ascending=False).astype(int)
    rank = rank.sort_values('rbar_projected', ascending=False)
    cols = ['dataset', 'peak_layer', 'rbar_global', 'rank_global', 'rbar_projected', 'rank_projected', 'rbar_rel_change',
            'cos_same', 'cos_diff', 'gap', 'rbar_letterA', 'rbar_letterB', 'rbar_control_ci_high', 'rbar_controlB_ci_high',
            'var_explained_by_letter', 'mean_energy_in_letter', 'n', 'n_letterA', 'n_letterB', 'letters_used', 'subspace_rank',
            'letter_effect_present']
    rank = rank[cols]
    rank.to_csv(out_dir / 'letter_effect_ranking.csv', index=False)
    log("RANKING at each dataset's peak layer (sorted by R̄ with the letter projected out)")
    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.float_format', '{:.4f}'.format):
        log(rank[['dataset', 'peak_layer', 'rbar_global', 'rank_global', 'rbar_projected', 'rank_projected', 'rbar_rel_change',
                  'cos_same', 'cos_diff', 'gap', 'rbar_letterA', 'rbar_letterB', 'rbar_control_ci_high',
                  'var_explained_by_letter', 'mean_energy_in_letter', 'letter_effect_present']].to_string(index=False))

    # ---- verdict --------------------------------------------------------------------------------
    rk = rank.set_index('dataset')
    psa = [d for d in args.datasets if d in PSA_DATASETS and d in rk.index]
    own = OWN_DATASET if OWN_DATASET in rk.index else None
    present = [d for d in psa if rk.loc[d, 'letter_effect_present']]
    absent = [d for d in psa if not rk.loc[d, 'letter_effect_present']]
    own_present = bool(rk.loc[own, 'letter_effect_present']) if own else None
    if not absent:
        answer = "sí, en los 7"
    elif not present:
        answer = "no, en ninguno"
    else:
        answer = f"solo en algunos: {', '.join(present)} (ausente en {', '.join(absent)})"
    if own is not None and not own_present:
        conclusion = (f"el efecto de letra NO está presente en {own} según el criterio"
                      + (f" (sí en {', '.join(present)})" if present else " ni en los datasets de PSA"))
    elif not absent:
        conclusion = "es una propiedad del protocolo A/B de CAA (leer en el token de la letra)"
    elif not present:
        conclusion = "es específico de mi dataset"
    else:
        conclusion = "aparece en parte de los datasets de PSA: ni exclusivo del dataset propio ni universal en el protocolo"
    gap_psa = float(rk.loc[psa, 'gap'].mean()) if psa else float('nan')
    chg_psa = float(rk.loc[psa, 'rbar_rel_change'].mean()) if psa else float('nan')
    lines = ["VEREDICTO",
             f"  ¿el efecto de letra aparece en los 7 datasets de PSA?   {answer}",
             f"  brecha media (cos_same - cos_diff) en la capa de pico:  PSA {gap_psa:.2f} | "
             + (f"{own} {rk.loc[own, 'gap']:.2f}" if own else "(dataset propio no incluido)"),
             f"  cambio relativo de R̄ al proyectar fuera la letra:       PSA {100 * chg_psa:+.1f}% | "
             + (f"{own} {100 * rk.loc[own, 'rbar_rel_change']:+.1f}%" if own else "")
             + "   (negativo = R̄ cae, positivo = R̄ sube al quitar la letra)"]
    if own:
        lines.append(f"  ranking tras proyectar: {own} queda en posición {int(rk.loc[own, 'rank_projected'])} de {len(rk)} "
                     f"(antes: {int(rk.loc[own, 'rank_global'])} de {len(rk)})")
        lines.append(f"  R̄ por letra en {own} (capa {int(rk.loc[own, 'peak_layer'])}): A {rk.loc[own, 'rbar_letterA']:.3f}, "
                     f"B {rk.loc[own, 'rbar_letterB']:.3f} vs global {rk.loc[own, 'rbar_global']:.3f} y control "
                     f"[{rk.loc[own, 'rbar_control_ci_high']:.3f} p97.5]; varianza en el subespacio de letra "
                     f"{100 * rk.loc[own, 'var_explained_by_letter']:.1f}%")
    lines.append(f"  criterio de presencia: R̄_A y R̄_B por encima del p{CI_PERCENTILES[1]} de subconjuntos aleatorios del mismo n "
                 f"y brecha >= {GAP_MIN}; 'los 7' = presente en todos los datasets de PSA analizados ({len(psa)})")
    lines.append(f"  => {conclusion}")
    verdict = "\n".join(lines)
    log("=" * 100)
    log(verdict)
    (out_dir / 'letter_effect_verdict.txt').write_text(verdict + "\n", encoding='utf-8')

    # ---- plots ----------------------------------------------------------------------------------
    header = f"{Path(model).name} | {args.n_control} control subsets, seed {args.control_seed}"
    plot_cos(df, peaks, args.datasets, header, out_dir / 'letter_effect_cos_by_layer.png')
    plot_rbar(df, peaks, args.datasets, header, out_dir / 'letter_effect_rbar_by_layer.png')
    plot_scatter(rank, header, out_dir / 'letter_effect_scatter.png')
    log("=" * 100)
    log(f"wrote {out_dir / 'letter_effect.csv'}, {out_dir / 'letter_effect_by_letter.csv'}, {out_dir / 'letter_effect_ranking.csv'}, "
        f"{out_dir / 'letter_effect_cos_by_layer.png'}, {out_dir / 'letter_effect_rbar_by_layer.png'}, "
        f"{out_dir / 'letter_effect_scatter.png'}, {out_dir / 'letter_effect_verdict.txt'}")


if __name__ == '__main__':
    main()
