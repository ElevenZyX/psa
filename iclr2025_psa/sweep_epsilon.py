"""
Epsilon sweep: PSA (one Gaussian mechanism per layer, basic composition) vs PSA-prima (one
Gaussian mechanism on the concatenation of the selected layers), both measured against the clean
clipped mean, on cached difference vectors.

Pipeline
  1. Extract ONCE per dataset the residual-stream difference vectors d_i^l = h_pos_i^l - h_neg_i^l
     for ALL decoder blocks, with the same call chain run.py and measure_rbar.py use
     (steering_vectors.extract_activations, which train_steering_vector calls internally,
     read_token_index=-2, batch_size=1), and cache them in <out-dir>/diffs/<dataset>.pt.
     A re-run with the cache present never loads the model.
  2. On the cached tensors: for the sweep layers S, every clip config, both mechanisms, every
     epsilon and every seed, cosine fidelity and norm ratio against the clean reference.

Scale convention (replicates run.py:44-53, priv_mean)
  run.py clips each d_i to L2 norm <= C, divides by C (every row then has norm <= 1), averages
  and adds noise_multiplier * N(0, I). The averaged vector has L2 sensitivity 2/n under
  replace-one neighbouring and noise_multiplier is the ABSOLUTE std of the noise on that
  normalised vector. Both mechanisms here use exactly that convention:
      PSA        v_l = mean_i clip_{C_l}(d_i^l) / C_l                + sigma_psa   * z_l   (z_l in R^h, one per layer)
      PSA-prima  v   = mean_i clip_{Cg}([d_i^{l_1} .. d_i^{l_S}]) / Cg + sigma_prima * z     (z in R^{S*h}, one draw)
  and are multiplied back by C_l / Cg so that norm ratios are in the units of the clean
  reference (cosines are scale invariant). check_priv_mean_replication() asserts numerically that
  C * priv_mean_as_in_run_py(d, z) == the PSA vector computed here.

Privacy accounting
  PSA:        each layer gets (eps/S, delta/S)  -> (eps, delta) by basic composition.
  PSA-prima:  one mechanism at (eps, delta).
  sigma is calibrated with the analytic Gaussian mechanism (Balle & Wang 2018) by default, or with
  the classical formula sigma = (2/n) sqrt(2 ln(1.25/delta)) / eps (--calibration classical, the
  paper's convention, only a valid guarantee for eps < 1).

Clip configs (c_global_mode x c_level)
  theoretical: C_l = C for every layer, Cg = C * sqrt(S).  C in --c-levels: pNN = percentile NN of
               the per-layer norms ||d_i^l|| pooled over the sweep layers, or a fixed number.
  percentile:  C_l = pNN of ||d_i^l|| per layer, Cg = pNN of the concatenated norms ||D_i||.
               (fixed levels are skipped in this mode.)
  The clean reference of a config is the per-layer clipped mean at that config's C_l (no noise).

Seeds are paired: seed k draws one standard normal z_k in R^{S*h} (torch.Generator, CPU, float64)
that is shared by both mechanisms, all epsilons and all clip configs, so the PSA-prima minus PSA
difference can be tested as a paired difference.

    python sweep_epsilon.py --out-dir sweep_1b
    python sweep_epsilon.py --out-dir sweep_1b --layers 11 12 13 14 15 --c-levels p50 20
    python sweep_epsilon.py --out-dir sweep_1b --calibration classical --n-seeds 50

Outputs (in --out-dir): sweep_results.csv (per seed), sweep_summary.csv (over seeds),
sweep_separation.csv, sweep_gain.csv, sweep_correlation.csv, clip_config.csv, norm_stats.csv,
sigma_table.csv, sweep_answers.txt, PNG plots, diffs/<dataset>.pt.
"""
import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.stats import norm as normal_dist, pearsonr, spearmanr, t as t_dist

from measure_rbar import DEFAULT_DATASETS, OWN_DATASET, READ_TOKEN_INDEX, SERIES_COLORS, load_pairs
from utils import make_dataset, make_pos_neg_pair, diagnose_read_token_index, device

MECHANISMS = ['psa', 'prima']
MECH_COLOR = {'psa': '#2a78d6', 'prima': '#eb6834'}        # categorical slots 1 and 2
MECH_LABEL = {'psa': 'PSA (per layer, basic composition)', 'prima': "PSA-prima (concatenated)"}
PSEUDO_LAYERS = ['mean', 'concat']  # rows aggregated over the sweep layers
RESULT_COLUMNS = ['model', 'dataset', 'mechanism', 'c_global_mode', 'c_level', 'epsilon', 'seed',
                  'layer', 'cosine', 'norm_ratio']
KEY = ['dataset', 'mechanism', 'c_global_mode', 'c_level', 'epsilon', 'layer']


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def gaussian_delta(eps: float, sigma: float, sens: float) -> float:
    """Exact delta of the Gaussian mechanism N(0, sigma^2 I) with L2 sensitivity `sens` at
    privacy level eps (Balle & Wang 2018, Theorem 8)."""
    a = sens / (2.0 * sigma)
    b = eps * sigma / sens
    return float(normal_dist.cdf(a - b) - math.exp(eps + normal_dist.logcdf(-a - b)))


def sigma_classical(eps: float, delta: float, sens: float) -> float:
    """Classical Gaussian mechanism (Dwork & Roth Thm A.1). With sens = 2/n this is the formula
    2 sqrt(2 ln(1.25/delta)) / (n eps). Only a valid (eps, delta) guarantee for eps < 1."""
    return sens * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def sigma_analytic(eps: float, delta: float, sens: float) -> float:
    """Smallest sigma such that the Gaussian mechanism is (eps, delta)-DP (Balle & Wang 2018)."""
    f = lambda log_s: gaussian_delta(eps, math.exp(log_s), sens) - delta
    lo = math.log(1e-4 * sens)
    hi = math.log(1e4 * sigma_classical(eps, delta, sens))
    assert f(lo) > 0.0 > f(hi), f"analytic calibration bracket failed for eps={eps}: {f(lo)}, {f(hi)}"
    return math.exp(brentq(f, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=500))


CALIBRATIONS = {'analytic': sigma_analytic, 'classical': sigma_classical}


@dataclass
class Accountant:
    """sigma (on the normalised scale, i.e. the run.py noise_multiplier) for each mechanism."""
    calibration: str
    n: int
    delta: float
    S: int

    @property
    def sens(self) -> float:
        return 2.0 / self.n  # replace-one sensitivity of the mean of n vectors of norm <= 1

    def sigma_psa(self, eps_total: float) -> float:
        return CALIBRATIONS[self.calibration](eps_total / self.S, self.delta / self.S, self.sens)

    def sigma_prima(self, eps_total: float) -> float:
        return CALIBRATIONS[self.calibration](eps_total, self.delta, self.sens)

    def eps_equivalent_theory(self, eps_total: float) -> float:
        """eps' such that PSA at eps' has the same per-coordinate noise std as PSA-prima at eps in
        the `theoretical` config (C_l = C, Cg = C sqrt(S)): sigma_psa(eps') = sqrt(S) sigma_prima(eps).
        The theoretical eps-equivalent gain is eps'/eps (constant 2.38 for S=5 under the classical
        formula, eps-dependent under the analytic one)."""
        target = math.sqrt(self.S) * self.sigma_prima(eps_total)
        g = lambda e: self.sigma_psa(e) - target
        lo, hi = eps_total * 1.0001, eps_total * 1000.0
        if not (g(lo) > 0.0 > g(hi)):
            return float('nan')
        return brentq(g, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=500)


def sigma_table(acc: Accountant, epsilons: list[float]) -> pd.DataFrame:
    rows = []
    for eps in epsilons:
        s_psa, s_prima = acc.sigma_psa(eps), acc.sigma_prima(eps)
        rows.append({
            'n': acc.n, 'calibration': acc.calibration, 'epsilon': eps,
            'eps_layer': eps / acc.S, 'delta_layer': acc.delta / acc.S,
            'sigma_psa': s_psa, 'sigma_prima': s_prima,
            'sigma_psa_over_sqrtS_sigma_prima': s_psa / (math.sqrt(acc.S) * s_prima),
            'eps_equiv_gain_theory': acc.eps_equivalent_theory(eps) / eps,
            'delta_real_psa_layer_classical': gaussian_delta(eps / acc.S, sigma_classical(eps / acc.S, acc.delta / acc.S, acc.sens), acc.sens),
            'delta_real_prima_classical': gaussian_delta(eps, sigma_classical(eps, acc.delta, acc.sens), acc.sens),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Extraction / cache
# --------------------------------------------------------------------------------------
def cache_path(out_dir: Path, name: str) -> Path:
    return out_dir / 'diffs' / f"{Path(name).stem if name.endswith('.json') else name}.pt"


def extract_diffs(name: str, model, tokenizer, model_name: str, n_samples: int, seed: int) -> dict:
    """d_i^l for every decoder block, exactly as run.py's aggregator would receive them."""
    from steering_vectors import extract_activations  # imported here so the sweep runs without it
    import steering_vectors
    import transformers

    mwe_data, total = load_pairs(name, n_samples, seed)
    pairs = make_dataset(mwe_data, tokenizer)
    num_layers = model.config.num_hidden_layers
    layers_all = list(range(num_layers))
    print(f"[{name}] extracting {len(pairs)} pairs (dataset size {total}) x {num_layers} layers")

    # Same prompt diagnostic measure_rbar.py runs before a long extraction.
    pos0, neg0 = make_pos_neg_pair(mwe_data[0], tokenizer)
    print("POS:", repr(pos0))
    print("NEG:", repr(neg0))
    ok = diagnose_read_token_index(pos0, tokenizer, READ_TOKEN_INDEX,
                                   expected_letter=mwe_data[0]['answer_matching_behavior'][1], label='[pos] ')
    ok = diagnose_read_token_index(neg0, tokenizer, READ_TOKEN_INDEX,
                                   expected_letter=mwe_data[0]['answer_not_matching_behavior'][1], label='[neg] ') and ok
    if not ok:
        raise SystemExit(f"read_token_index={READ_TOKEN_INDEX} does not land on the answer letter for {name!r}")

    # train_steering_vector.py:195-207 is exactly this call; the aggregator (run.py priv_mean)
    # then receives torch.concat(pos_acts[l]), torch.concat(neg_acts[l]) (train_steering_vector.py:152-154).
    pos_acts, neg_acts = extract_activations(
        model, tokenizer, pairs,
        layers=layers_all,
        read_token_index=READ_TOKEN_INDEX,
        move_to_cpu=True,
        show_progress=True,
        batch_size=1,
        tqdm_desc=f"Extracting {name}",
    )
    assert sorted(pos_acts.keys()) == layers_all == sorted(neg_acts.keys()), \
        f"library returned layers {sorted(pos_acts.keys())}, expected {layers_all}"
    hidden = model.config.hidden_size
    diffs = {}
    for layer in layers_all:
        pos = torch.concat(pos_acts[layer])
        neg = torch.concat(neg_acts[layer])
        assert pos.shape == neg.shape == (len(pairs), hidden), f"layer {layer}: {tuple(pos.shape)} vs {tuple(neg.shape)}"
        assert pos.dtype == neg.dtype == torch.float32, f"layer {layer}: activations are {pos.dtype}/{neg.dtype}, expected float32"
        diffs[layer] = (pos - neg).contiguous()
    return {
        'model': model_name, 'dataset': name, 'n': len(pairs), 'total': total, 'seed': seed,
        'read_token_index': READ_TOKEN_INDEX, 'hidden': hidden, 'layers': layers_all,
        'diffs': diffs, 'example_pos': pairs[0][0], 'example_neg': pairs[0][1],
        'steering_vectors_version': steering_vectors.__version__,
        'transformers_version': str(transformers.__version__), 'torch_version': str(torch.__version__),
    }


def load_cache(path: Path, model_name: str, n_samples: int, seed: int, sweep_layers: list[int]) -> dict:
    # weights_only=True (the default since torch 2.6) is enough: the cache holds tensors, dicts, lists, ints and strs only.
    cache = torch.load(path, map_location='cpu', weights_only=True)
    problems = []
    if cache['model'] != model_name:
        problems.append(f"model {cache['model']!r} != {model_name!r}")
    if cache['seed'] != seed:
        problems.append(f"shuffle seed {cache['seed']} != {seed}")
    if cache['read_token_index'] != READ_TOKEN_INDEX:
        problems.append(f"read_token_index {cache['read_token_index']} != {READ_TOKEN_INDEX}")
    if cache['n'] != min(n_samples, cache['total']):
        problems.append(f"n {cache['n']} != min(n_samples={n_samples}, total={cache['total']})")
    missing = [l for l in sweep_layers if l not in cache['diffs']]
    if missing:
        problems.append(f"layers {missing} not in cache (has {cache['layers']})")
    if problems:
        raise SystemExit(f"cache {path} does not match this run: {'; '.join(problems)}. "
                         f"Re-extract with --force-extract or use another --out-dir.")
    return cache


def rbar_from_cache(cache: dict) -> pd.DataFrame:
    """Same arithmetic as measure_rbar.RbarAggregator (float32), for every cached layer."""
    rows = []
    for layer in cache['layers']:
        d = cache['diffs'][layer]
        mean_vec = torch.mean(d, dim=0)
        norm_mean = torch.norm(mean_vec).item()
        mean_norm = torch.norm(d, dim=1).mean().item()
        rows.append({'dataset': cache['dataset'], 'layer': layer, 'rbar': norm_mean / mean_norm,
                     'norm_mean': norm_mean, 'mean_norm': mean_norm, 'n': cache['n']})
    return pd.DataFrame(rows)


def crosscheck_rbar(rbar_here: pd.DataFrame, rbar_csv: pd.DataFrame | None, model_name: str) -> None:
    if rbar_csv is None:
        return
    name = rbar_here['dataset'].iloc[0]
    ref = rbar_csv[(rbar_csv['model'] == model_name) & (rbar_csv['dataset'] == name)]
    if ref.empty:
        print(f"  NOTE: no rows for ({model_name}, {name}) in the rbar csv; nothing to cross-check.")
        return
    merged = rbar_here.merge(ref[['layer', 'rbar', 'norm_mean', 'mean_norm', 'n']], on='layer', suffixes=('', '_csv'))
    worst = (merged['rbar'] - merged['rbar_csv']).abs().max()
    n_ok = bool((merged['n'] == merged['n_csv']).all())
    if worst > 1e-4 or not n_ok:
        print(f"  WARNING: rbar from the cached diffs differs from the rbar csv (max |diff| = {worst:.2e}, "
              f"same n: {n_ok}). The extraction pipelines are NOT equivalent; investigate before trusting the sweep.")
    else:
        print(f"  rbar cross-check vs csv OK on {len(merged)} layers (max |diff| = {worst:.1e})")


# --------------------------------------------------------------------------------------
# Clipping and mechanisms (original units)
# --------------------------------------------------------------------------------------
def clip_rows(x: torch.Tensor, C: float) -> tuple[torch.Tensor, float]:
    """run.py:47-49: scale each row to norm <= C. Returns (clipped, fraction of rows clipped)."""
    norms = torch.norm(x, dim=1)
    scale_factors = torch.clamp(C / norms, max=1.0).view(-1, 1)
    return x * scale_factors, (norms > C).double().mean().item()


def priv_mean_as_in_run_py(pos, neg, C, noise_multiplier, noise):
    """Verbatim run.py:44-53 with the N(0,1) draw injected instead of sampled."""
    diff = pos - neg
    norms = torch.norm(diff, dim=1)
    scale_factors = torch.clamp(C / norms, max=1.0).view(-1, 1)
    diff = diff * scale_factors
    diff /= C
    mu = torch.mean(diff, dim=0)
    return mu + (noise_multiplier * noise)


def check_priv_mean_replication(d_layer: torch.Tensor, C: float, sigma: float) -> None:
    """C * priv_mean(run.py) must equal mean(clip_C(d)) + C*sigma*z, i.e. the PSA vector of this
    script in original units. Run on real float32 diffs of one layer."""
    g = torch.Generator(device='cpu').manual_seed(12345)
    z = torch.randn(d_layer.shape[1], generator=g, dtype=torch.float32)
    ref = C * priv_mean_as_in_run_py(d_layer.clone(), torch.zeros_like(d_layer), C, sigma, z)
    clipped, _ = clip_rows(d_layer.clone(), C)
    mine = torch.mean(clipped, dim=0) + C * sigma * z
    err = (ref - mine).abs().max().item()
    tol = 1e-5 * max(1.0, ref.abs().max().item())
    assert err <= tol, f"scale convention mismatch vs run.py priv_mean: max |diff| = {err:.3e} > {tol:.3e}"
    print(f"  priv_mean replication check OK (C={C:.4g}, sigma={sigma:.4g}, max |diff| = {err:.1e})")


@dataclass
class ClipConfig:
    mode: str            # 'theoretical' | 'percentile'
    level: str           # 'p10' .. 'p90' | 'C20'
    C_layer: np.ndarray  # (S,) per-layer clip bound used by PSA and by the clean reference
    Cg: float            # clip bound of the concatenated vector (PSA-prima)


def parse_c_levels(tokens: list[str]) -> list[tuple[str, float]]:
    levels = []
    for tok in tokens:
        if tok.lower().startswith('p'):
            q = float(tok[1:])
            assert 0 < q < 100, f"bad percentile level {tok!r}"
            levels.append((f"p{q:g}", q))
        else:
            levels.append((f"C{float(tok):g}", float(tok)))
    return levels


def build_clip_configs(norms_layer: np.ndarray, norms_cat: np.ndarray, modes: list[str],
                       levels: list[tuple[str, float]]) -> list[ClipConfig]:
    S = norms_layer.shape[1]
    configs = []
    for mode in modes:
        for label, value in levels:
            fixed = label.startswith('C')
            if mode == 'theoretical':
                C = value if fixed else float(np.percentile(norms_layer.reshape(-1), value))
                configs.append(ClipConfig(mode, label, np.full(S, C), C * math.sqrt(S)))
            elif mode == 'percentile':
                if fixed:
                    continue  # a fixed C has no percentile counterpart for the concatenation
                C_layer = np.percentile(norms_layer, value, axis=0)
                configs.append(ClipConfig(mode, label, C_layer.astype(float), float(np.percentile(norms_cat, value))))
            else:
                raise ValueError(mode)
    return configs


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cos along the last dim with broadcasting."""
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def ci95(x: np.ndarray) -> tuple[float, float, float]:
    """mean, std (ddof=1) and t-based 95% half-width."""
    k = len(x)
    m, s = float(x.mean()), float(x.std(ddof=1))
    return m, s, float(t_dist.ppf(0.975, k - 1) * s / math.sqrt(k))


# --------------------------------------------------------------------------------------
# Sweep on one dataset
# --------------------------------------------------------------------------------------
def sweep_dataset(cache: dict, layers: list[int], modes: list[str], levels: list[tuple[str, float]],
                  epsilons: list[float], Z: torch.Tensor, seeds: list[int], calibration: str, delta: float,
                  model_name: str) -> tuple[list[dict], list[dict], list[dict], list[dict], Accountant]:
    name, n, h = cache['dataset'], cache['n'], cache['hidden']
    S, K = len(layers), Z.shape[0]
    d = torch.stack([cache['diffs'][l] for l in layers], dim=1).double()  # (n, S, h)
    D = d.reshape(n, S * h)
    acc = Accountant(calibration, n, delta, S)

    norms_layer = d.norm(dim=2).numpy()   # (n, S)
    norms_cat = D.norm(dim=1).numpy()     # (n,)
    m_raw = d.mean(dim=0)                 # (S, h) unclipped mean
    configs = build_clip_configs(norms_layer, norms_cat, modes, levels)

    norm_rows = []
    for j, l in enumerate(layers):
        x = norms_layer[:, j]
        norm_rows.append({'dataset': name, 'layer': l, 'n': n, 'mean_norm': x.mean(), 'max_norm': x.max(),
                          **{f'p{q}': np.percentile(x, q) for q in (10, 25, 50, 90)}})
    norm_rows.append({'dataset': name, 'layer': 'concat', 'n': n, 'mean_norm': norms_cat.mean(), 'max_norm': norms_cat.max(),
                      **{f'p{q}': np.percentile(norms_cat, q) for q in (10, 25, 50, 90)}})
    print(f"  norms: per-layer mean {np.round(norms_layer.mean(0), 2).tolist()} | concat mean {norms_cat.mean():.2f} "
          f"p90 {np.percentile(norms_cat, 90):.2f} max {norms_cat.max():.2f}")

    result_rows, pred_rows, clip_rows_out = [], [], []
    for cfg in configs:
        # ---- noiseless vectors (original units) -----------------------------------
        m_psa = torch.empty(S, h, dtype=torch.float64)
        frac_psa = np.empty(S)
        for j in range(S):
            clipped, frac_psa[j] = clip_rows(d[:, j, :], float(cfg.C_layer[j]))
            m_psa[j] = clipped.mean(dim=0)
        clipped_cat, frac_prima = clip_rows(D, cfg.Cg)
        m_prima = clipped_cat.mean(dim=0).reshape(S, h)
        ref = m_psa                                    # clean reference of this config
        ref_norm = ref.norm(dim=1)                     # (S,)
        m = {'psa': m_psa, 'prima': m_prima}
        frac = {'psa': frac_psa, 'prima': np.full(S, frac_prima)}
        bound = {'psa': cfg.C_layer, 'prima': np.full(S, cfg.Cg)}
        cos_noiseless = {mech: cosine(m[mech], ref).numpy() for mech in MECHANISMS}
        cos_noiseless_cat = {mech: cosine(m[mech].reshape(-1), ref.reshape(-1)).item() for mech in MECHANISMS}
        cos_ref_raw = cosine(ref, m_raw).numpy()
        for mech in MECHANISMS:
            for j, l in enumerate(layers):
                clip_rows_out.append({
                    'dataset': name, 'mechanism': mech, 'c_global_mode': cfg.mode, 'c_level': cfg.level,
                    'layer': l, 'clip_bound': bound[mech][j], 'frac_clipped': frac[mech][j],
                    'cos_noiseless_vs_ref': cos_noiseless[mech][j],
                    'norm_noiseless_over_ref': (m[mech][j].norm() / ref_norm[j]).item(),
                    'ref_norm': ref_norm[j].item(), 'cos_ref_vs_unclipped_mean': cos_ref_raw[j],
                })
            clip_rows_out.append({
                'dataset': name, 'mechanism': mech, 'c_global_mode': cfg.mode, 'c_level': cfg.level,
                'layer': 'concat', 'clip_bound': cfg.Cg if mech == 'prima' else float('nan'),
                'frac_clipped': frac_prima if mech == 'prima' else float('nan'),
                'cos_noiseless_vs_ref': cos_noiseless_cat[mech],
                'norm_noiseless_over_ref': (m[mech].norm() / ref.norm()).item(),
                'ref_norm': ref.norm().item(), 'cos_ref_vs_unclipped_mean': cosine(ref.reshape(-1), m_raw.reshape(-1)).item(),
            })

        # ---- noise ---------------------------------------------------------------
        for eps in epsilons:
            sig = {'psa': acc.sigma_psa(eps), 'prima': acc.sigma_prima(eps)}
            for mech in MECHANISMS:
                s = torch.tensor(bound[mech] * sig[mech], dtype=torch.float64)       # (S,) per-coordinate noise std, original units
                V = m[mech][None] + s[None, :, None] * Z                              # (K, S, h)
                cos_l = cosine(V, ref[None]).numpy()                                  # (K, S)
                nr_l = (V.norm(dim=2) / ref_norm[None]).numpy()                       # (K, S)
                cos_cat = cosine(V.reshape(K, -1), ref.reshape(1, -1)).numpy()        # (K,)
                nr_cat = (V.reshape(K, -1).norm(dim=1) / ref.norm()).numpy()
                # analytic prediction: E[cos] ~ cos(m, ref) * ||m|| / sqrt(||m||^2 + h s^2), h large
                mn = m[mech].norm(dim=1)
                pred_cos_l = cos_noiseless[mech] * (mn / torch.sqrt(mn ** 2 + h * s ** 2)).numpy()
                pred_nr_l = (torch.sqrt(mn ** 2 + h * s ** 2) / ref_norm).numpy()
                tot = torch.sqrt(m[mech].norm() ** 2 + h * (s ** 2).sum())
                pred_cos_cat = cos_noiseless_cat[mech] * (m[mech].norm() / tot).item()
                pred_nr_cat = (tot / ref.norm()).item()
                base = {'model': model_name, 'dataset': name, 'mechanism': mech, 'c_global_mode': cfg.mode,
                        'c_level': cfg.level, 'epsilon': eps}
                for k, seed in enumerate(seeds):
                    for j, l in enumerate(layers):
                        result_rows.append({**base, 'seed': seed, 'layer': l, 'cosine': cos_l[k, j], 'norm_ratio': nr_l[k, j]})
                    result_rows.append({**base, 'seed': seed, 'layer': 'mean', 'cosine': cos_l[k].mean(), 'norm_ratio': nr_l[k].mean()})
                    result_rows.append({**base, 'seed': seed, 'layer': 'concat', 'cosine': cos_cat[k], 'norm_ratio': nr_cat[k]})
                for j, l in enumerate(layers):
                    pred_rows.append({**base, 'layer': l, 'sigma': sig[mech], 'clip_bound': bound[mech][j],
                                      'noise_std_orig': s[j].item(), 'frac_clipped': frac[mech][j],
                                      'cos_pred': pred_cos_l[j], 'norm_ratio_pred': pred_nr_l[j], 'n': n})
                pred_rows.append({**base, 'layer': 'mean', 'sigma': sig[mech], 'clip_bound': float('nan'),
                                  'noise_std_orig': float(np.mean(s.numpy())), 'frac_clipped': float(np.mean(frac[mech])),
                                  'cos_pred': float(np.mean(pred_cos_l)), 'norm_ratio_pred': float(np.mean(pred_nr_l)), 'n': n})
                pred_rows.append({**base, 'layer': 'concat', 'sigma': sig[mech], 'clip_bound': cfg.Cg if mech == 'prima' else float('nan'),
                                  'noise_std_orig': float(np.sqrt(np.mean(s.numpy() ** 2))), 'frac_clipped': frac_prima if mech == 'prima' else float('nan'),
                                  'cos_pred': pred_cos_cat, 'norm_ratio_pred': pred_nr_cat, 'n': n})
        print(f"  [{cfg.mode:11s} {cfg.level:5s}] C_layer={np.round(cfg.C_layer, 2).tolist()} Cg={cfg.Cg:.2f} | "
              f"frac clipped psa={np.round(frac_psa, 2).tolist()} prima={frac_prima:.2f} | "
              f"prima noiseless cos vs ref (mean over layers)={cos_noiseless['prima'].mean():.4f} | "
              f"ref vs unclipped mean cos={cos_ref_raw.mean():.4f}")
    return result_rows, pred_rows, clip_rows_out, norm_rows, acc


# --------------------------------------------------------------------------------------
# Aggregation and statistics
# --------------------------------------------------------------------------------------
def summarize(results: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(['model'] + KEY, sort=False)
    agg = g.agg(n_seeds=('seed', 'size'), cos_mean=('cosine', 'mean'), cos_std=('cosine', lambda x: x.std(ddof=1)),
                norm_ratio_mean=('norm_ratio', 'mean'), norm_ratio_std=('norm_ratio', lambda x: x.std(ddof=1))).reset_index()
    tcrit = t_dist.ppf(0.975, agg['n_seeds'] - 1)
    half = tcrit * agg['cos_std'] / np.sqrt(agg['n_seeds'])
    agg['cos_ci95_lo'], agg['cos_ci95_hi'] = agg['cos_mean'] - half, agg['cos_mean'] + half
    half = tcrit * agg['norm_ratio_std'] / np.sqrt(agg['n_seeds'])
    agg['norm_ratio_ci95_lo'], agg['norm_ratio_ci95_hi'] = agg['norm_ratio_mean'] - half, agg['norm_ratio_mean'] + half
    out = agg.merge(preds, on=['model'] + KEY, how='left', validate='one_to_one')
    out['cos_pred_minus_emp'] = out['cos_pred'] - out['cos_mean']
    return out


def separation(results: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Paired (same seed => same z) PSA-prima minus PSA difference of the layer-mean cosine, plus
    whether the two marginal 95% bands overlap."""
    r = results[results['layer'] == 'mean']
    keys = ['dataset', 'c_global_mode', 'c_level', 'epsilon']
    wide = r.pivot_table(index=keys + ['seed'], columns='mechanism', values='cosine').reset_index()
    wide['diff'] = wide['prima'] - wide['psa']
    rows = []
    for key, sub in wide.groupby(keys, sort=False):
        m, s, hw = ci95(sub['diff'].to_numpy())
        rows.append(dict(zip(keys, key), n_seeds=len(sub), diff_mean=m, diff_std=s, diff_ci95_lo=m - hw, diff_ci95_hi=m + hw,
                         paired_separated=bool((m - hw > 0) or (m + hw < 0)), winner=('prima' if m > 0 else 'psa')))
    sep = pd.DataFrame(rows)
    sm = summary[summary['layer'] == 'mean'].pivot_table(index=keys, columns='mechanism',
                                                         values=['cos_ci95_lo', 'cos_ci95_hi', 'cos_mean']).reset_index()
    sm.columns = ['_'.join(c).rstrip('_') if c[1] else c[0] for c in sm.columns]
    sep = sep.merge(sm, on=keys)
    sep['marginal_bands_disjoint'] = (sep['cos_ci95_lo_prima'] > sep['cos_ci95_hi_psa']) | (sep['cos_ci95_lo_psa'] > sep['cos_ci95_hi_prima'])
    return sep


def eps_equivalent_empirical(eps_grid: np.ndarray, cos_psa: np.ndarray, target: float) -> tuple[float, str]:
    """eps' at which the PSA curve reaches `target` (linear interpolation in log eps). Returns
    (eps', censor) with censor '>' if PSA never reaches it inside the grid, '<' if PSA is already
    above it at the smallest eps."""
    order = np.argsort(eps_grid)
    x, y = np.log(eps_grid[order]), np.maximum.accumulate(cos_psa[order])
    if target > y[-1]:
        return float(eps_grid[order][-1]), '>'
    if target < y[0]:
        return float(eps_grid[order][0]), '<'
    return float(np.exp(np.interp(target, y, x))), ''


def gains(summary: pd.DataFrame, accountants: dict[str, Accountant]) -> pd.DataFrame:
    sm = summary[summary['layer'] == 'mean']
    per_layer = summary[~summary['layer'].isin(PSEUDO_LAYERS)]
    rows = []
    for (name, mode, level), sub in sm.groupby(['dataset', 'c_global_mode', 'c_level'], sort=False):
        psa = sub[sub['mechanism'] == 'psa'].sort_values('epsilon')
        prima = sub[sub['mechanism'] == 'prima'].sort_values('epsilon')
        eps_grid, cos_psa = psa['epsilon'].to_numpy(), psa['cos_mean'].to_numpy()
        non_monotone = int((np.diff(cos_psa) < 0).sum())
        acc = accountants[name]
        pl_cfg = per_layer[(per_layer['dataset'] == name) & (per_layer['c_global_mode'] == mode) & (per_layer['c_level'] == level)]
        for _, row in prima.iterrows():
            eps = row['epsilon']
            eps_eq, censor = eps_equivalent_empirical(eps_grid, cos_psa, row['cos_mean'])
            theory = acc.eps_equivalent_theory(eps) / eps if mode == 'theoretical' else float('nan')
            # per-layer noise-std ratio PSA/prima at this eps (the noise-only reference in both modes)
            pl = pl_cfg[pl_cfg['epsilon'] == eps]
            r_l = pl[pl['mechanism'] == 'psa'].sort_values('layer')['noise_std_orig'].to_numpy() / \
                pl[pl['mechanism'] == 'prima'].sort_values('layer')['noise_std_orig'].to_numpy()
            rows.append({'dataset': name, 'c_global_mode': mode, 'c_level': level, 'epsilon': eps,
                         'cos_prima': row['cos_mean'], 'eps_equiv_psa': eps_eq, 'censor': censor,
                         'gain_empirical': eps_eq / eps, 'gain_theory_eps_equiv': theory,
                         'noise_std_ratio_min': r_l.min(), 'noise_std_ratio_max': r_l.max(),
                         'psa_curve_non_monotone_steps': non_monotone})
    return pd.DataFrame(rows)


def correlation(summary: pd.DataFrame, rbar_all: pd.DataFrame, eps_target: float, mechanism: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R̄ / norm_mean at the peak-R̄ layer vs cosine drop 1 - cos at eps_target, one point per dataset."""
    sm = summary[(summary['layer'] == 'mean') & (summary['mechanism'] == mechanism)]
    eps_avail = np.sort(sm['epsilon'].unique())
    eps_used = float(eps_avail[np.argmin(np.abs(np.log(eps_avail) - np.log(eps_target)))])
    if not np.isclose(eps_used, eps_target):
        print(f"WARNING: eps={eps_target} is not in the sweep; the correlation uses the closest one, eps={eps_used}")
    sm = sm[np.isclose(sm['epsilon'], eps_used)]
    peak = rbar_all.loc[rbar_all.groupby('dataset')['rbar'].idxmax()][['dataset', 'layer', 'rbar', 'norm_mean', 'mean_norm', 'n']]
    peak = peak.rename(columns={'layer': 'peak_layer', 'rbar': 'rbar_peak', 'norm_mean': 'norm_mean_peak', 'mean_norm': 'mean_norm_peak'})
    pts = sm.merge(peak, on='dataset', suffixes=('', '_rbar'))
    pts['degradation'] = 1.0 - pts['cos_mean']
    pts['degradation_ci95_half'] = pts['cos_mean'] - pts['cos_ci95_lo']
    stat_rows = []
    for (mode, level), sub in pts.groupby(['c_global_mode', 'c_level'], sort=False):
        row = {'c_global_mode': mode, 'c_level': level, 'epsilon': eps_used, 'mechanism': mechanism, 'n_points': len(sub)}
        for xcol in ('rbar_peak', 'norm_mean_peak'):
            if len(sub) >= 3 and sub[xcol].nunique() > 1 and sub['degradation'].nunique() > 1:
                pr, pp = pearsonr(sub[xcol], sub['degradation'])
                sr, sp = spearmanr(sub[xcol], sub['degradation'])
                slope, intercept = np.polyfit(sub[xcol], sub['degradation'], 1)
            else:
                pr = pp = sr = sp = slope = intercept = float('nan')
            row.update({f'pearson_r_{xcol}': pr, f'pearson_p_{xcol}': pp, f'spearman_rho_{xcol}': sr, f'spearman_p_{xcol}': sp,
                        f'slope_{xcol}': slope, f'intercept_{xcol}': intercept})
        stat_rows.append(row)
    return pts, pd.DataFrame(stat_rows)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color='#e5e4e0', linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _panel_cos_vs_eps(ax, summary: pd.DataFrame, name: str, mode: str, level: str, epsilons: list[float], title: str):
    sub = summary[(summary['dataset'] == name) & (summary['c_global_mode'] == mode) & (summary['c_level'] == level)
                  & (summary['layer'] == 'mean')]
    for mech in MECHANISMS:
        s = sub[sub['mechanism'] == mech].sort_values('epsilon')
        if s.empty:
            continue
        ax.fill_between(s['epsilon'], s['cos_ci95_lo'], s['cos_ci95_hi'], color=MECH_COLOR[mech], alpha=0.18, linewidth=0)
        ax.plot(s['epsilon'], s['cos_mean'], color=MECH_COLOR[mech], marker='o', markersize=4, linewidth=2, label=MECH_LABEL[mech])
        ax.plot(s['epsilon'], s['cos_pred'], color=MECH_COLOR[mech], linestyle='--', linewidth=1, alpha=0.8,
                label=f"{'PSA' if mech == 'psa' else 'PSA-prima'} analytic prediction")
    ax.set_xscale('log')
    ax.set_xticks(epsilons)
    ax.set_xticklabels([f"{e:g}" for e in epsilons], fontsize=8)
    ax.set_ylim(min(-0.02, float(sub['cos_ci95_lo'].min()) - 0.02) if not sub.empty else -0.02, 1.02)
    ax.set_title(title, fontsize=10)
    _style(ax)


def plot_cos_vs_eps(summary: pd.DataFrame, datasets: list[str], n_by_dataset: dict[str, int], mode: str, level: str,
                    epsilons: list[float], header: str, path: Path) -> None:
    ncols = min(4, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, datasets):
        _panel_cos_vs_eps(ax, summary, name, mode, level, epsilons, f"{name} (n={n_by_dataset.get(name)})")
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('ε total (log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('cos(v_priv, v_clean), mean over layers')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{header} — clip config {mode}/{level}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dataset_big(summary: pd.DataFrame, name: str, n: int, modes: list[str], levels_by_mode: dict[str, list[str]],
                     epsilons: list[float], header: str, path: Path) -> None:
    ncols = max(len(v) for v in levels_by_mode.values())
    fig, axes = plt.subplots(len(modes), ncols, figsize=(5.2 * ncols, 4.4 * len(modes)), squeeze=False)
    for i, mode in enumerate(modes):
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(levels_by_mode[mode]):
                ax.set_visible(False)
                continue
            level = levels_by_mode[mode][j]
            _panel_cos_vs_eps(ax, summary, name, mode, level, epsilons, f"{mode} / {level}")
            ax.set_xlabel('ε total (log)')
            if j == 0:
                ax.set_ylabel('cos(v_priv, v_clean), mean over layers')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{header} — {name} (n={n})", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(points: pd.DataFrame, stats: pd.DataFrame, datasets: list[str], mode: str, levels: list[str],
                 eps_used: float, mechanism: str, n_full: int, header: str, path: Path) -> None:
    xcols = [('rbar_peak', 'R̄ at the peak layer'), ('norm_mean_peak', '‖mean diff‖ at the peak layer')]
    fig, axes = plt.subplots(len(levels), 2, figsize=(11, 3.9 * len(levels)), squeeze=False)
    alignment = [d for d in datasets if d != OWN_DATASET]
    color_of = {d: SERIES_COLORS[i % len(SERIES_COLORS)] for i, d in enumerate(alignment)}
    color_of[OWN_DATASET] = 'black'
    for i, level in enumerate(levels):
        sub = points[(points['c_global_mode'] == mode) & (points['c_level'] == level)]
        st = stats[(stats['c_global_mode'] == mode) & (stats['c_level'] == level)]
        for (xcol, xlabel), ax in zip(xcols, axes[i]):
            for _, p in sub.iterrows():
                full = p['n'] >= n_full
                col = color_of.get(p['dataset'], '#6b6b6b')
                ax.errorbar(p[xcol], p['degradation'], yerr=p['degradation_ci95_half'], fmt='none', ecolor='#9a9a94', elinewidth=1, capsize=2)
                ax.scatter(p[xcol], p['degradation'], s=55, color=col if full else 'white', edgecolors=col, linewidths=1.6, zorder=5)
                ax.annotate(p['dataset'] + ('' if full else f" (n={int(p['n'])})"), (p[xcol], p['degradation']),
                            textcoords='offset points', xytext=(5, 4), fontsize=7.5, color='#3a3a38')
            if not st.empty and np.isfinite(st.iloc[0][f'slope_{xcol}']):
                xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 20)
                ax.plot(xs, st.iloc[0][f'slope_{xcol}'] * xs + st.iloc[0][f'intercept_{xcol}'], color='#6b6b6b', linewidth=1.2, linestyle='--')
                ax.text(0.02, 0.96, f"Pearson r={st.iloc[0][f'pearson_r_{xcol}']:.2f} (p={st.iloc[0][f'pearson_p_{xcol}']:.2g})\n"
                                    f"Spearman ρ={st.iloc[0][f'spearman_rho_{xcol}']:.2f} (p={st.iloc[0][f'spearman_p_{xcol}']:.2g})\n"
                                    f"n={len(sub)} datasets",
                        transform=ax.transAxes, va='top', fontsize=8, color='#3a3a38',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.85))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"1 − cos at ε={eps_used:g} ({mechanism})")
            ax.set_title(f"{mode} / {level}", fontsize=10)
            _style(ax)
    fig.suptitle(f"{header}\nR̄ / ‖mean diff‖ at the peak-R̄ layer vs cosine drop at ε={eps_used:g} ({mechanism}); hollow = n < {n_full}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------------------
def write_answers(sep: pd.DataFrame, gain: pd.DataFrame, corr_stats: pd.DataFrame, summary: pd.DataFrame,
                  sigmas: pd.DataFrame, args, S: int, h: int, config_order: list[tuple[str, str]], path: Path) -> None:
    lines = []
    P = lines.append

    def by_config(df: pd.DataFrame):
        for mode, level in config_order:
            sub = df[(df['c_global_mode'] == mode) & (df['c_level'] == level)]
            if len(sub):
                yield (mode, level), sub
    P("=" * 100)
    P("RESPUESTAS")
    P("=" * 100)
    P(f"modelo {args.model} | capas {sorted(args.layers)} (S={S}) | calibración {args.calibration} | δ_total={args.delta:g} | "
      f"{args.n_seeds} semillas apareadas | ε ∈ {sorted(args.epsilons)}")
    P("")
    # ---- chequeo de escala ---------------------------------------------------------
    sm_ = summary[summary['layer'] == 'mean']
    dev = sm_['cos_pred_minus_emp'].abs()
    outside = float(((sm_['cos_pred'] < sm_['cos_ci95_lo']) | (sm_['cos_pred'] > sm_['cos_ci95_hi'])).mean())
    tol = 0.02 + 4.0 / h  # the E[cos] approximation has a bias of order 1/h; the rest is seed noise
    P(f"Chequeo de escala: |cos predicho analíticamente − cos empírico| (media sobre capas): "
      f"máx {dev.max():.4f}, mediana {dev.median():.4f} sobre {len(dev)} combinaciones; la predicción cae fuera del IC 95% empírico "
      f"en {100 * outside:.1f}% de ellas (esperado ≈5% si no hay sesgo). "
      + (f"OK (máx ≤ {tol:.3f} = 0.02 + 4/h)." if dev.max() <= tol and outside <= 0.25
         else "ADVERTENCIA: desviación sistemática entre predicción y empírico; revisar la convención de escala / el clipping."))
    P("")
    # ---- 1. separación ------------------------------------------------------------------
    P("1. ¿En qué rango de ε se separan PSA y PSA-prima más allá de la barra de error?")
    P("   Criterio A (apareado): IC 95% t de la diferencia prima−PSA del coseno medio por capas excluye 0 (misma semilla ⇒ mismo z).")
    P("   Criterio B (marginal): las bandas IC 95% de los dos mecanismos no se solapan.")
    for (mode, level), sub in by_config(sep):
        P(f"   [{mode}/{level}]")
        any_sep = False
        for eps, se in sub.groupby('epsilon', sort=True):
            a_prima = se[se['paired_separated'] & (se['winner'] == 'prima')]['dataset'].tolist()
            a_psa = se[se['paired_separated'] & (se['winner'] == 'psa')]['dataset'].tolist()
            b = se[se['marginal_bands_disjoint']]['dataset'].tolist()
            if a_prima or a_psa or b:
                any_sep = True
            P(f"      ε={eps:<6g} A: prima>PSA en {len(a_prima)}/{len(se)} datasets, PSA>prima en {len(a_psa)}/{len(se)} | "
              f"B: bandas disjuntas en {len(b)}/{len(se)} | Δcos medio = {se['diff_mean'].mean():+.4f}"
              + (f" | PSA gana en: {a_psa}" if a_psa else ""))
        if not any_sep:
            P("      => NO HAY DIFERENCIA DETECTABLE en ningún ε con ninguno de los dos criterios.")
        else:
            eps_a = sorted(sub[sub['paired_separated']]['epsilon'].unique())
            eps_b = sorted(sub[sub['marginal_bands_disjoint']]['epsilon'].unique())
            P(f"      => criterio A (apareado) en ε ∈ {[float(e) for e in eps_a]}; criterio B (bandas) en ε ∈ {[float(e) for e in eps_b]}"
              + ("" if eps_b else " — SIN separación visible en las bandas marginales, solo en la prueba apareada."))
    P("")
    # ---- 2. ganancia --------------------------------------------------------------------
    P(f"2. ¿La ganancia observada se acerca al sqrt(S) teórico (= {math.sqrt(S):.3f})?")
    P("   ganancia empírica = ε'/ε, donde ε' es el ε al que PSA alcanza el coseno de PSA-prima en ε (interpolación en log ε).")
    P("   '>' = PSA no alcanza ese coseno dentro de la grilla (cota inferior); '<' = prima es peor que PSA en ε mínimo (cota superior).")
    r_std = sigmas['sigma_psa_over_sqrtS_sigma_prima']
    P(f"   referencia teórica (modo theoretical, solo ruido, n={int(sigmas['n'].iloc[0])}, calibración {args.calibration}): "
      f"ratio de std de ruido PSA/prima en unidades originales = σ_psa/(√S·σ_prima) = {r_std.min():.3f}..{r_std.max():.3f} sobre la grilla "
      f"(constante {math.sqrt(S) * math.sqrt(math.log(1.25 * S / args.delta) / math.log(1.25 / args.delta)):.3f} = √S·√(ln(1.25S/δ)/ln(1.25/δ)) bajo la fórmula clásica); "
      f"ganancia teórica ε-equivalente: {sigmas['eps_equiv_gain_theory'].min():.3f}..{sigmas['eps_equiv_gain_theory'].max():.3f}. "
      f"Compara la empírica con esta última.")
    for (mode, level), sub in by_config(gain):
        ok = sub[sub['censor'] == '']
        cen = sub[sub['censor'] != '']
        line = f"   [{mode}/{level}] "
        if len(ok):
            line += (f"ganancia empírica: mediana {ok['gain_empirical'].median():.2f}, rango [{ok['gain_empirical'].min():.2f}, "
                     f"{ok['gain_empirical'].max():.2f}] ({len(ok)} puntos dataset×ε no censurados)")
        else:
            line += "sin puntos no censurados"
        if len(cen):
            line += f"; {len(cen)} censurados ({(cen['censor'] == '>').sum()} con '>', {(cen['censor'] == '<').sum()} con '<')"
        if mode == 'theoretical':
            line += f" | teórica ε-equiv: {sub['gain_theory_eps_equiv'].min():.2f}..{sub['gain_theory_eps_equiv'].max():.2f}"
        else:
            line += (f" | ratio de std de ruido por capa PSA/prima: {sub['noise_std_ratio_min'].min():.2f}..{sub['noise_std_ratio_max'].max():.2f} "
                     f"(desbalance de normas: <sqrt(S) en capas de norma chica, >sqrt(S) en las grandes)")
        nm = sub['psa_curve_non_monotone_steps'].max()
        if nm:
            line += f" | NOTA: la curva PSA no es monótona en hasta {int(nm)} pasos (ruido entre ε); se usó el máximo acumulado."
        P(line)
    P("")
    # ---- 3. correlación -----------------------------------------------------------------
    P(f"3. ¿Correlaciona R̄ con la degradación (1 − cos a ε={args.corr_eps:g}, {args.corr_mechanism}, media por capas)?")
    for _, sub in by_config(corr_stats):
        r = sub.iloc[0]
        P(f"   [{r['c_global_mode']}/{r['c_level']}] n={int(r['n_points'])} datasets | "
          f"R̄:  Pearson r={r['pearson_r_rbar_peak']:+.2f} (p={r['pearson_p_rbar_peak']:.2g}), Spearman ρ={r['spearman_rho_rbar_peak']:+.2f} (p={r['spearman_p_rbar_peak']:.2g}) | "
          f"‖mean diff‖: Pearson r={r['pearson_r_norm_mean_peak']:+.2f} (p={r['pearson_p_norm_mean_peak']:.2g}), "
          f"Spearman ρ={r['spearman_rho_norm_mean_peak']:+.2f} (p={r['spearman_p_norm_mean_peak']:.2g})")
    npts = int(corr_stats['n_points'].max()) if len(corr_stats) else 0
    df_ = max(npts - 2, 1)
    tc = t_dist.ppf(0.975, df_)
    P(f"   Honestidad estadística: con n={npts} puntos, |r| tiene que superar ≈{tc / math.sqrt(tc ** 2 + df_):.2f} "
      f"para p<0.05; un p-valor aquí no distingue una relación real de dos datasets extremos. Además (a) la predicción analítica dice que lo que manda es "
      f"‖mean diff‖ = R̄·mean‖diff‖ (no R̄ solo), y (b) los datasets con n<{args.n_samples} tienen más ruido al mismo ε (σ ∝ 1/n), lo que confunde la relación. "
      f"Tómalo como exploratorio, no como evidencia.")
    text = "\n".join(lines)
    print(text)
    path.write_text(text, encoding='utf-8')


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, 'reconfigure'):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    parser.add_argument("--datasets", nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument("--layers", type=int, nargs='+', default=[7, 8, 9, 10, 11], help="sweep layers S")
    parser.add_argument("--epsilons", type=float, nargs='+', default=[0.25, 0.5, 1, 2, 4, 8, 16, 32], help="total epsilon grid")
    parser.add_argument("--delta", type=float, default=1e-5, help="total delta")
    parser.add_argument("--c-modes", nargs='+', default=['theoretical', 'percentile'], choices=['theoretical', 'percentile'])
    parser.add_argument("--c-levels", nargs='+', default=['p10', 'p25', 'p50', 'p90', '20'],
                        help="pNN = percentile of the observed norms, a number = fixed C")
    parser.add_argument("--calibration", default='analytic', choices=list(CALIBRATIONS))
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="dataset shuffle seed (same as run.py / measure_rbar.py)")
    parser.add_argument("--n-seeds", type=int, default=20, help="noise seeds 0..n-1 (torch.Generator, CPU)")
    parser.add_argument("--out-dir", default="sweep_results")
    parser.add_argument("--rbar-csv", default="resultados_rbar_1b/rbar_results.csv",
                        help="measure_rbar.py output, used to cross-check the extraction; the scatter uses R̄ recomputed from the cache")
    parser.add_argument("--corr-eps", type=float, default=1.0, help="epsilon at which the cosine drop of the scatter is taken")
    parser.add_argument("--corr-mechanism", default='psa', choices=MECHANISMS)
    parser.add_argument("--force-extract", action='store_true')
    parser.add_argument("--skip-plots", action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / 'diffs').mkdir(parents=True, exist_ok=True)
    layers = sorted(args.layers)
    assert len(set(layers)) == len(layers), f"duplicate layers in {args.layers}"
    S = len(layers)
    epsilons = sorted(args.epsilons)
    levels = parse_c_levels(args.c_levels)
    seeds = list(range(args.n_seeds))
    assert args.n_seeds >= 2, "need at least 2 seeds for a confidence interval"

    print("=" * 100)
    print(f"model        : {args.model}")
    print(f"sweep layers : {layers} (S={S})")
    print(f"epsilons     : {epsilons} | delta_total={args.delta:g} | calibration={args.calibration}")
    print(f"clip configs : modes {args.c_modes} x levels {[l for l, _ in levels]}")
    print(f"n_samples    : {args.n_samples} | shuffle seed {args.seed} | noise seeds {seeds[0]}..{seeds[-1]} (paired)")
    print(f"datasets     : {args.datasets}")
    print(f"out dir      : {out_dir.resolve()}")
    print("=" * 100)

    # ---- calibration sanity print ------------------------------------------------------
    acc_ref = Accountant(args.calibration, args.n_samples, args.delta, S)
    k = 2.0 * math.sqrt(2.0 * math.log(1.25 / args.delta))
    eps_c = k / (args.n_samples * 0.02)
    eps_a = brentq(lambda e: gaussian_delta(e, 0.02, 2.0 / args.n_samples) - args.delta, 1e-3, 1e3)
    print(f"paper noise_multiplier=0.02 at n={args.n_samples}, delta={args.delta:g}: eps per layer = {eps_c:.4f} (classical) / "
          f"{eps_a:.4f} (analytic); basic composition over {S} layers: eps_total = {S * eps_c:.3f} (classical), delta_total = {S * args.delta:g}")
    st = sigma_table(acc_ref, epsilons)
    print(f"sigma table for n={args.n_samples} (sigma = run.py noise_multiplier scale; datasets with fewer pairs get larger sigma):")
    with pd.option_context('display.width', 250, 'display.float_format', '{:.6g}'.format):
        print(st.to_string(index=False))
    if args.calibration == 'classical':
        bad = st[(st['delta_real_psa_layer_classical'] > args.delta / S) | (st['delta_real_prima_classical'] > args.delta)]
        if len(bad):
            print(f"WARNING: with the classical formula, eps in {bad['epsilon'].tolist()} are NOT (eps, delta)-DP "
                  f"(real delta above target); the relative comparison is still consistent.")
    st.to_csv(out_dir / 'sigma_table.csv', index=False)

    rbar_csv = None
    if Path(args.rbar_csv).is_file():
        rbar_csv = pd.read_csv(args.rbar_csv)
        print(f"rbar csv for cross-check: {args.rbar_csv} ({len(rbar_csv)} rows)")
    else:
        print(f"NOTE: rbar csv {args.rbar_csv!r} not found; R̄ is recomputed from the cached diffs (no cross-check).")

    # ---- caches / model ------------------------------------------------------------------
    need = [d for d in args.datasets if args.force_extract or not cache_path(out_dir, d).is_file()]
    model = tokenizer = None
    if need:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from steering_vectors import get_num_matching_layers, guess_and_enhance_layer_config
        print(f"extracting {len(need)} dataset(s): {need}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model = model.to(device).eval()
        param_dtype = next(model.parameters()).dtype
        assert param_dtype == torch.float32, f"model loaded as {param_dtype}, expected float32"
        num_layers = model.config.num_hidden_layers
        layer_config = guess_and_enhance_layer_config(model, None, 'decoder_block')
        lib_num_layers = get_num_matching_layers(model, layer_config['decoder_block'])
        assert lib_num_layers == num_layers, f"steering_vectors matched {lib_num_layers} decoder blocks but config says {num_layers}"
        assert all(0 <= l < num_layers for l in layers), f"layers {layers} out of range [0, {num_layers})"
        print(f"model loaded: {num_layers} layers, hidden {model.config.hidden_size}, dtype {param_dtype}, device {next(model.parameters()).device}")
    else:
        print("all diffs cached; the model is not loaded")

    Z = None  # noise draws shared by everything (paired design); built once the hidden size is known
    all_results, all_preds, all_clip, all_norms, all_rbar = [], [], [], [], []
    accountants: dict[str, Accountant] = {}
    n_by_dataset: dict[str, int] = {}
    errors: dict[str, str] = {}
    results_path = out_dir / 'sweep_results.csv'
    for name in args.datasets:
        t0 = time.time()
        try:
            path = cache_path(out_dir, name)
            if name in need:
                cache = extract_diffs(name, model, tokenizer, args.model, args.n_samples, args.seed)
                torch.save(cache, path)
                print(f"[{name}] diffs cached -> {path} ({path.stat().st_size / 1e6:.0f} MB)")
            cache = load_cache(path, args.model, args.n_samples, args.seed, layers)
            n_by_dataset[name] = cache['n']
            if cache['n'] < args.n_samples:
                print(f"[{name}] WARNING: only {cache['n']} pairs (< {args.n_samples}); sigma is calibrated with n={cache['n']}, "
                      f"i.e. {args.n_samples / cache['n']:.2f}x more noise than an n={args.n_samples} dataset at the same epsilon.")
            if Z is None:
                h = cache['hidden']
                Z = torch.stack([torch.randn(S, h, generator=torch.Generator(device='cpu').manual_seed(s), dtype=torch.float64)
                                 for s in seeds])
                print(f"noise draws Z: {tuple(Z.shape)} float64, seeds {seeds[0]}..{seeds[-1]}")
            assert cache['hidden'] == Z.shape[2], f"hidden size {cache['hidden']} != {Z.shape[2]} (mixed models in one out-dir?)"
            rb = rbar_from_cache(cache)
            crosscheck_rbar(rb, rbar_csv, args.model)
            check_priv_mean_replication(cache['diffs'][layers[0]], C=20.0, sigma=0.02)
            res, preds, clip, norms, acc = sweep_dataset(cache, layers, args.c_modes, levels, epsilons, Z, seeds,
                                                         args.calibration, args.delta, args.model)
        except Exception:
            errors[name] = traceback.format_exc()
            print(f"ERROR on dataset {name!r} (continuing with the next one):\n{errors[name]}")
            continue
        accountants[name] = acc
        all_rbar.append(rb)
        all_results.extend(res)
        all_preds.extend(preds)
        all_clip.extend(clip)
        all_norms.extend(norms)
        pd.DataFrame(all_results, columns=RESULT_COLUMNS).to_csv(results_path, index=False)  # partial results
        print(f"[{name}] done in {time.time() - t0:.0f}s | {len(res)} rows | partial results -> {results_path}")

    print("=" * 100)
    if not all_results:
        raise SystemExit("no dataset succeeded; nothing to report")
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- aggregation --------------------------------------------------------------------
    results = pd.DataFrame(all_results, columns=RESULT_COLUMNS)
    preds = pd.DataFrame(all_preds)
    summary = summarize(results, preds)
    summary.to_csv(out_dir / 'sweep_summary.csv', index=False)
    pd.DataFrame(all_clip).to_csv(out_dir / 'clip_config.csv', index=False)
    pd.DataFrame(all_norms).to_csv(out_dir / 'norm_stats.csv', index=False)
    rbar_all = pd.concat(all_rbar, ignore_index=True)
    rbar_all.to_csv(out_dir / 'rbar_from_cache.csv', index=False)
    sep = separation(results, summary)
    sep.to_csv(out_dir / 'sweep_separation.csv', index=False)
    gain = gains(summary, accountants)
    gain.to_csv(out_dir / 'sweep_gain.csv', index=False)
    points, corr_stats = correlation(summary, rbar_all, args.corr_eps, args.corr_mechanism)
    points.to_csv(out_dir / 'sweep_correlation_points.csv', index=False)
    corr_stats.to_csv(out_dir / 'sweep_correlation.csv', index=False)

    # ---- plots -----------------------------------------------------------------------------
    datasets_done = [d for d in args.datasets if d in n_by_dataset and d not in errors]
    levels_by_mode = {m: list(summary[summary['c_global_mode'] == m]['c_level'].unique()) for m in args.c_modes}
    header = f"{Path(args.model).name} | layers {layers} | {args.calibration} calibration | δ={args.delta:g} | {args.n_seeds} seeds"
    if not args.skip_plots:
        for mode in args.c_modes:
            for level in levels_by_mode[mode]:
                plot_cos_vs_eps(summary, datasets_done, n_by_dataset, mode, level, epsilons, header,
                                out_dir / f'cos_vs_eps_{mode}_{level}.png')
        if OWN_DATASET in datasets_done:
            plot_dataset_big(summary, OWN_DATASET, n_by_dataset[OWN_DATASET], args.c_modes, levels_by_mode, epsilons, header,
                             out_dir / f'cos_vs_eps_{OWN_DATASET}_big.png')
        eps_used = float(points['epsilon'].iloc[0]) if len(points) else args.corr_eps
        for mode in args.c_modes:
            plot_scatter(points, corr_stats, datasets_done, mode, levels_by_mode[mode], eps_used, args.corr_mechanism,
                         args.n_samples, header, out_dir / f'scatter_rbar_vs_degradation_{mode}.png')
        print(f"plots written to {out_dir}")

    # ---- summary table ------------------------------------------------------------------------
    sm = summary[summary['layer'] == 'mean'].pivot_table(index=['dataset', 'c_global_mode', 'c_level', 'epsilon'],
                                                          columns='mechanism', values=['cos_mean', 'cos_pred'], sort=False).reset_index()
    sm.columns = ['_'.join(c).rstrip('_') if c[1] else c[0] for c in sm.columns]
    print("cosine (mean over layers, mean over seeds) — empirical vs analytic prediction:")
    with pd.option_context('display.max_rows', None, 'display.width', 250, 'display.float_format', '{:.4f}'.format):
        print(sm.to_string(index=False))
    print("=" * 100)
    config_order = [(m, l) for m in args.c_modes for l in levels_by_mode[m]]
    write_answers(sep, gain, corr_stats, summary, st, args, S, int(Z.shape[2]), config_order, out_dir / 'sweep_answers.txt')
    print(f"\nwrote {results_path}, sweep_summary.csv, sweep_separation.csv, sweep_gain.csv, sweep_correlation.csv, "
          f"clip_config.csv, norm_stats.csv, sigma_table.csv, sweep_answers.txt in {out_dir}")
    if errors:
        print("\nDATASETS THAT FAILED:")
        for name, tb in errors.items():
            print(f"  - {name}: {tb.strip().splitlines()[-1]}")
        (out_dir / 'sweep_errors.log').write_text("\n\n".join(f"### {n}\n{tb}" for n, tb in errors.items()))


if __name__ == '__main__':
    main()
