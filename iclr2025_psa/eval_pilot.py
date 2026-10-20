"""
Pilot steering evaluation, 1B model (open-unlearning/tofu_Llama-3.2-1B-Instruct_full): PSA vs PSA-1G plus references,
on hallucination, refusal and tofu-forget10.

Naming. The mechanism called "PSA-prima" / 'prima' in sweep_epsilon.py, sweep_c.py and their CSVs (clip_stats.csv,
mechanism column; c_selection.csv's Cg / Cg_eval are its clip bound) is called PSA-1G here: ONE Gaussian on the
concatenated layers. Only this script uses the new name; where it calls those modules or reads those CSVs it maps
prima -> PSA-1G and says so in a comment. The repo-wide rename is left for a separate cleanup.

Vectors, built from the sweep_epsilon.py caches <diffs-dir>/<dataset>.pt (n = 400 pairs, d_i^l = h_pos - h_neg at
read_token_index -2, every layer, prompts with utils.CHAT_DATE), on the window W = 7..11 (S = 5 layers), with
C = C_eval and Cg = Cg_eval of sweep_c_1b/c_selection.csv:
  Mean Steering (paper)  v_l = mean_i d_i^l / max_i ||d_i^l||                        run.py:37-41 scaled_mean, no noise
  PSA sin ruido          v_l = mean_i d_i^l / max(C, ||d_i^l||)
  PSA                    v_l = mean_i d_i^l / max(C, ||d_i^l||) + sigma_PSA(eps) z_l,   sigma_PSA = sigma(eps/S, delta/S, 2/n)
                         PSA Algorithm 1 = run.py:44-53 priv_mean with noise_multiplier = sigma_PSA; NOT multiplied by C.
  PSA-1G sin ruido       v   = sqrt(S) * mean_i D_i / max(Cg, ||D_i||),                 D_i = [d_i^7, ..., d_i^11]
  PSA-1G                 v   = sqrt(S) * (mean_i D_i / max(Cg, ||D_i||) + sigma_1G(eps) z),   sigma_1G = sigma(eps, delta, 2/n)
                         x sqrt(S) is post-processing (no privacy cost): with every row clipped, the mean over the S layers
                         of ||row_l||^2 is 1, as for a PSA row. The S x h result is split back into layers 7..11 in order.
  sigma: Balle-Wang analytic calibration through sweep_epsilon.Accountant (the accountant sweep_c.py uses), delta = 1e-5,
  sensitivity 2/n with the real n. z: sweep_c.paired_draws, one N(0, I_{S x h}) per seed (float64), shared by PSA (z_l = row l)
  and PSA-1G (z = the same draw flattened), every eps and every dataset, as in sweep_c.py.

Conditions. A/B metric = utils.evaluate_model (the function run.py uses through `from utils import *`), unchanged: mean over
the test set of P(answer_matching_behavior) normalised over the two answers. Steering exactly as run.py:93/100:
SteeringVector.apply(model, multiplier=lambda, min_token_index=0), i.e. added on every token.
  sin steering (once), and with lambda = +2 and -2: Mean Steering (paper), PSA sin ruido, PSA-1G sin ruido,
  PSA and PSA-1G at eps 1, 4, 8 with seeds 0, 1.
Test sets: datasets/test/<d>/test_dataset_ab.json for hallucination and refusal. tofu-forget10 has no separate test: its 400
generate pairs are used as they are (letters already balanced 200/200) -> IN-SAMPLE, they are the pairs the vectors come from.

Checks (all abort): cache metadata (sweep_epsilon.load_cache), shuffle reproduction (cache_alignment.verify_permutation),
R̄ from the cache == R̄ csv (cache_alignment.crosscheck_rbar; the csv is mandatory), c_selection window / C_eval / Cg_eval,
sweep_epsilon.check_priv_mean_replication, PSA == run.py priv_mean with the same z, the first training prompt rebuilt with
utils.make_pos_neg_pair == the prompt stored in the cache, CHAT_DATE in the test prompts (utils.diagnose_read_token_index),
Mean Steering from the cache == train_steering_vector + run.py scaled_mean on the same 400 hallucination pairs
(max |diff| <= 1e-5 per layer), and no steering hook left on the model after each condition.

    python eval_pilot.py --selftest        # CPU, synthetic data + a tiny random Llama, nothing downloaded, nothing written
    python eval_pilot.py                   # sweep_1b/diffs + sweep_c_1b/c_selection.csv + resultados_rbar_1b/rbar_results.csv
                                           # -> pilot_1b/ (aborts if it already exists)

Outputs (--out-dir): resultados.csv (dataset, condicion, epsilon, semilla, lambda, ab), generaciones.txt, resumen.txt
(A/B tables with +2 / -2, sigma per mechanism and eps, ||v_l|| and cos(v_l, Mean Steering) per layer and condition,
clip fractions, results of the checks).
"""
import argparse
import io
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cache_alignment import (OWN_DATASET, crosscheck_rbar, load_dataset_json, load_rbar_csv, shuffle_permutation,
                             verify_permutation)
from measure_rbar import READ_TOKEN_INDEX
# sweep_epsilon / sweep_c call PSA-1G "prima": Accountant.sigma_prima, MECHS 'prima' (see the module docstring).
from sweep_c import MECHS, MODELS, SEL_COLUMNS, paired_draws, sweep_one
from sweep_epsilon import (Accountant, check_priv_mean_replication, clip_rows, cosine, load_cache, noisy_vectors,
                           priv_mean_as_in_run_py, sigma_analytic)

MODEL_TAG = '1b'
MODEL = MODELS[MODEL_TAG]
DATASETS = ['hallucination', 'refusal', OWN_DATASET]
EXPECTED_WINDOW = '7-11'      # sweep_c_1b/c_selection.csv, every dataset of the 1B
EXPECTED_C_EVAL = 2.0         # idem (C_eval; Cg_eval = C_eval * sqrt(S))
S = 5
EPSILONS = [1.0, 4.0, 8.0]
SEEDS = [0, 1]
LAMBDAS = [2.0, -2.0]
DELTA = 1e-5
CALIBRATION = 'analytic'
N_SAMPLES = 400
SHUFFLE_SEED = 42             # shuffle the caches were extracted with (measure_rbar.load_pairs)
CHECK4_DATASET = 'hallucination'
CHECK4_TOL = 1e-5             # max |diff| per layer, Mean Steering from the cache vs train_steering_vector
PRIV_MEAN_TOL = 1e-5          # relative, same as sweep_epsilon.check_priv_mean_replication
GEN_DATASET = OWN_DATASET
GEN_N, GEN_SEED, GEN_MAX_NEW_TOKENS = 5, 0, 64
GEN_EPS, GEN_NOISE_SEED, GEN_LAMBDA = [1.0, 8.0], 0, 2.0
CHOICES_SEP = "\n\nChoices:\n"   # separator of the A/B questions (pares/to_caa_format.py:41, same in the PSA datasets)

NONE = 'sin steering'
MEAN = 'Mean Steering (paper)'
PSA0 = 'PSA sin ruido'
G1_0 = 'PSA-1G sin ruido'
PSA = 'PSA'
G1 = 'PSA-1G'
CONDITION_ORDER = [NONE, MEAN, PSA0, G1_0, PSA, G1]
RESULT_COLUMNS = ['dataset', 'condicion', 'epsilon', 'semilla', 'lambda', 'ab']


def log(msg: str = '') -> None:
    print(msg, flush=True)


class Report:
    """Lines printed now and written to resumen.txt at the end."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = '') -> None:
        log(msg)
        self.lines.append(msg)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def window_layers(w: str) -> list[int]:
    a, b = (int(x) for x in w.split('-'))
    return list(range(a, b + 1))


def cond_label(v: dict) -> str:
    if v['condicion'] in (PSA, G1):
        return f"{v['condicion']} ε={v['epsilon']:g} s={v['semilla']}"
    return v['condicion']


# --------------------------------------------------------------------------------------
# run.py replica (run.py cannot be imported: at module level it reads `args`, defined only under __main__, and loads a
# model, run.py:144-148; scaled_mean is local to get_results)
# --------------------------------------------------------------------------------------
def scaled_mean_as_in_run_py(pos, neg):
    """Verbatim run.py:37-41."""
    diff = pos - neg
    scale = torch.max(torch.norm(diff, dim=1))
    diff /= scale
    return torch.mean(diff, dim=0)


# --------------------------------------------------------------------------------------
# c_selection.csv
# --------------------------------------------------------------------------------------
def read_c_selection(src, datasets: list[str]) -> dict[str, dict]:
    """src: path or file-like. C_eval / Cg_eval per dataset (C itself is empty when sweep_c reported a tie).
    Cg / Cg_eval are the clip bound of the concatenated vector, i.e. of PSA-1G ('prima' in sweep_c.py)."""
    sel = pd.read_csv(src, keep_default_na=False, na_values=[''])
    missing = [c for c in SEL_COLUMNS if c not in sel.columns]
    if missing:
        raise SystemExit(f"c_selection.csv has no columns {missing}; run sweep_c.py --recompute-selection on it first")
    sub = sel[sel['model'] == MODEL]
    out = {}
    for name in datasets:
        r = sub[sub['dataset'] == name]
        if len(r) != 1:
            raise SystemExit(f"c_selection.csv: {len(r)} rows for ({MODEL}, {name}), expected 1")
        r = r.iloc[0]
        if str(r['ventana']) != EXPECTED_WINDOW:
            raise SystemExit(f"c_selection.csv: {name} has window {r['ventana']!r}, expected {EXPECTED_WINDOW!r}")
        C, Cg = float(r['C_eval']), float(r['Cg_eval'])
        if C != EXPECTED_C_EVAL:
            raise SystemExit(f"c_selection.csv: {name} has C_eval {C:g}, expected {EXPECTED_C_EVAL:g}")
        if abs(Cg - C * math.sqrt(S)) > 1e-9 * C * math.sqrt(S):
            raise SystemExit(f"c_selection.csv: {name} has Cg_eval {Cg!r} != C_eval * sqrt({S}) = {C * math.sqrt(S)!r}")
        out[name] = {'C': C, 'Cg': Cg, 'ventana': str(r['ventana']), 'C_sel': r['C'], 'empate': str(r['empate']) == 'True',
                     'C_empatados': str(r['C_empatados']), 'recorte_total': str(r['recorte_total']) == 'True'}
    return out


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------
def psa_rows(d: torch.Tensor, C: float) -> tuple[torch.Tensor, np.ndarray]:
    """d (n, S, h) -> d_i^l / max(C, ||d_i^l||) per layer (= clip_rows / C), and the fraction of rows clipped per layer."""
    rows = torch.empty_like(d)
    frac = np.empty(d.shape[1])
    for j in range(d.shape[1]):
        clipped, frac[j] = clip_rows(d[:, j, :], C)
        rows[:, j, :] = clipped / C
    return rows, frac


def g1_rows(d: torch.Tensor, Cg: float) -> tuple[torch.Tensor, float]:
    """d (n, S, h) -> D_i / max(Cg, ||D_i||) on the concatenation, reshaped back to (n, S, h) (before x sqrt(S)),
    and the fraction of concatenated rows clipped."""
    n, s, h = d.shape
    clipped, frac = clip_rows(d.reshape(n, s * h), Cg)
    return (clipped / Cg).reshape(n, s, h), frac


def build_vectors(d32: torch.Tensor, C: float, Cg: float, acc: Accountant, Z: torch.Tensor, epsilons: list[float],
                  seeds: list[int]) -> tuple[list[dict], dict]:
    """d32: (n, S, h) float32 cached diffs on the window. Z: (len(seeds), S, h) float64 from paired_draws.
    Returns the vectors [{condicion, epsilon, semilla, sigma, sigma_coord, vec (S, h) float64}] and clip fractions.
    sigma = calibrated std on the normalised mechanism; sigma_coord = std per coordinate of the vector actually applied
    (sigma for PSA, sqrt(S) * sigma for PSA-1G)."""
    n, s, h = d32.shape
    assert s == S and Z.shape == (len(seeds), S, h), (tuple(d32.shape), tuple(Z.shape))
    d = d32.double()
    # Mean Steering on the float32 diffs, as run.py's aggregator receives them (pos - neg with neg = 0 gives d)
    mean_paper = torch.stack([scaled_mean_as_in_run_py(d32[:, j, :], torch.zeros_like(d32[:, j, :]))
                              for j in range(S)]).double()
    rp, frac_psa = psa_rows(d, C)
    m_psa = rp.mean(dim=0)
    rg, frac_g1 = g1_rows(d, Cg)
    m_g1 = rg.mean(dim=0)
    sq = math.sqrt(S)
    nan = float('nan')
    vecs = [{'condicion': MEAN, 'epsilon': nan, 'semilla': nan, 'sigma': 0.0, 'sigma_coord': 0.0, 'vec': mean_paper},
            {'condicion': PSA0, 'epsilon': nan, 'semilla': nan, 'sigma': 0.0, 'sigma_coord': 0.0, 'vec': m_psa},
            {'condicion': G1_0, 'epsilon': nan, 'semilla': nan, 'sigma': 0.0, 'sigma_coord': 0.0, 'vec': sq * m_g1}]
    ones = np.ones(S)                                   # normalised scale: the bound is 1 (no multiplication by C / Cg)
    for eps in epsilons:
        s_psa = acc.sigma_psa(eps)
        s_g1 = acc.sigma_prima(eps)                     # 'prima' in sweep_epsilon = PSA-1G
        V_psa, _ = noisy_vectors(m_psa, ones, s_psa, Z)
        V_g1, _ = noisy_vectors(m_g1, ones, s_g1, Z)    # same z, laid out (S, h) = the flattened draw split per layer
        for k, seed in enumerate(seeds):
            vecs.append({'condicion': PSA, 'epsilon': eps, 'semilla': seed, 'sigma': s_psa, 'sigma_coord': s_psa,
                         'vec': V_psa[k]})
            vecs.append({'condicion': G1, 'epsilon': eps, 'semilla': seed, 'sigma': s_g1, 'sigma_coord': sq * s_g1,
                         'vec': sq * V_g1[k]})
    return vecs, {'frac_psa': frac_psa, 'frac_g1': frac_g1}


def find_vec(vecs: list[dict], cond: str, eps: float | None = None, seed: int | None = None) -> dict:
    hits = [v for v in vecs if v['condicion'] == cond and (eps is None or v['epsilon'] == eps)
            and (seed is None or v['semilla'] == seed)]
    assert len(hits) == 1, (cond, eps, seed, len(hits))
    return hits[0]


def to_steering_vector(vec: torch.Tensor, layers: list[int], dtype: torch.dtype, dev):
    from steering_vectors import SteeringVector
    assert vec.shape[0] == len(layers)
    return SteeringVector({l: vec[j].to(dtype).to(dev) for j, l in enumerate(layers)}, 'decoder_block')


def check_psa_equals_priv_mean(d32: torch.Tensor, C: float, sigma: float, z: torch.Tensor, ours: torch.Tensor,
                               label: str) -> str:
    """Our PSA vector (S, h) == run.py priv_mean (verbatim copy, float32) with noise_multiplier = sigma and the same z."""
    worst = 0.0
    for j in range(d32.shape[1]):
        x = d32[:, j, :].clone()
        ref = priv_mean_as_in_run_py(x, torch.zeros_like(x), C, sigma, z[j].float()).double()
        err = (ref - ours[j]).abs().max().item()
        tol = PRIV_MEAN_TOL * max(1.0, ref.abs().max().item())
        assert err <= tol, f"{label} layer pos {j}: PSA != run.py priv_mean, max |diff| {err:.3e} > {tol:.3e}"
        worst = max(worst, err)
    msg = f"  {label}: PSA == run.py priv_mean (C={C:g}, noise_multiplier={sigma:.5g}, same z) OK, max |diff| = {worst:.1e}"
    log(msg)
    return msg


def compare_mean_vectors(lib: torch.Tensor, ours: torch.Tensor, layers: list[int], tol: float) -> list[str]:
    """Check 4: max |diff| per layer between train_steering_vector + scaled_mean and our cache-built vector."""
    lines = []
    diffs = (lib.double() - ours.double()).abs().amax(dim=1)
    cs = cosine(lib.double(), ours.double())
    for j, l in enumerate(layers):
        lines.append(f"  capa {l}: max |diff| = {diffs[j].item():.2e} | max |v| = {ours[j].abs().max().item():.3e} | "
                     f"‖v‖ = {ours[j].norm().item():.4f} | cos = {cs[j].item():.8f}")
    for x in lines:
        log(x)
    bad = [layers[j] for j in range(len(layers)) if not diffs[j].item() <= tol]
    if bad:
        raise AssertionError(f"check 4: Mean Steering from the cache differs from train_steering_vector + run.py "
                             f"scaled_mean by more than {tol:g} on layers {bad} (max {diffs.max().item():.2e})")
    return lines


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_window_cache(diffs_dir: Path, name: str, layers: list[int], rbar_csv: pd.DataFrame, csv_label: str) -> dict:
    path = diffs_dir / f"{name}.pt"
    if not path.is_file():
        raise SystemExit(f"no cache {path} (extract it with sweep_epsilon.py --extract-only)")
    cache = load_cache(path, MODEL, N_SAMPLES, SHUFFLE_SEED, layers)       # model / seed / read_token_index / n / layers
    if cache['dataset'] != name:
        raise SystemExit(f"{path}: cache dataset {cache['dataset']!r} != {name!r}")
    n, h = cache['n'], cache['hidden']
    for l in cache['layers']:
        d = cache['diffs'][l]
        if tuple(d.shape) != (n, h) or d.dtype != torch.float32:
            raise SystemExit(f"{path} layer {l}: {tuple(d.shape)} {d.dtype}, expected ({n}, {h}) float32")
    data = load_dataset_json(name)
    perm = shuffle_permutation(cache['total'], n, cache['seed'])
    verify_permutation(cache, data, perm)
    crosscheck_rbar(cache, rbar_csv, csv_label)
    log(f"[{name}] cache {path} | n={n} of {cache['total']} | hidden {h} | versions: steering_vectors "
        f"{cache.get('steering_vectors_version')}, transformers {cache.get('transformers_version')}, "
        f"torch {cache.get('torch_version')}")
    return {'n': n, 'hidden': h, 'd32': torch.stack([cache['diffs'][l] for l in layers], dim=1).contiguous(),
            'train_rows': [data[i] for i in perm], 'example_pos': cache['example_pos'],
            'example_neg': cache['example_neg'],
            'versions': {k: cache.get(k) for k in ('steering_vectors_version', 'transformers_version', 'torch_version')}}


def load_test(name: str) -> tuple[list[dict], str]:
    if name == OWN_DATASET:
        data = load_dataset_json(name)            # its own 400 generate pairs, json order, letters as they are
        cnt = Counter(r['answer_matching_behavior'] for r in data)
        if cnt != Counter({'(A)': 200, '(B)': 200}):
            raise SystemExit(f"{name}: answer_matching_behavior letters {dict(cnt)}, expected 200/200")
        return data, f"datasets/generate/{name}/generate_dataset.json (IN-SAMPLE)"
    path = Path('datasets') / 'test' / name / 'test_dataset_ab.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f), str(path)


def gen_questions(data: list[dict]) -> list[tuple[int, str]]:
    idx = random.Random(GEN_SEED).sample(range(len(data)), GEN_N)
    out = []
    for i in idx:
        q = data[i]['question']
        assert q.count(CHOICES_SEP) == 1, f"row {i}: {CHOICES_SEP!r} appears {q.count(CHOICES_SEP)} times"
        out.append((i, q.split(CHOICES_SEP)[0]))
    return out


def format_generation_prompt(tokenizer, question: str) -> str:
    """Same chat format as utils._format_llama3 (system SYS_PROMPT, date_string=CHAT_DATE, the template's BOS stripped so
    the tokenizer adds exactly one), ending in the assistant header instead of an answer. utils.generate_text_with_template
    is not used: it has no system prompt, unlike the extraction prompts."""
    from utils import CHAT_DATE, LLAMA3_STRIP_TEMPLATE_BOS, SYS_PROMPT, check_today_date
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': question}],
                                         tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE)
    if LLAMA3_STRIP_TEMPLATE_BOS and tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    check_today_date(text, '[generación] ')
    return text


# --------------------------------------------------------------------------------------
# Model-side helpers
# --------------------------------------------------------------------------------------
def n_forward_hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def ab_eval(model, tokenizer, test_pairs, sv, lam: float, base_hooks: int) -> float:
    from utils import evaluate_model
    with torch.no_grad():                        # evaluate_model itself has no no_grad; numbers are identical
        if sv is None:
            r = evaluate_model(model, tokenizer, test_pairs)
        else:
            with sv.apply(model, multiplier=lam, min_token_index=0):
                r = evaluate_model(model, tokenizer, test_pairs)
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit(f"{n_forward_hooks(model)} forward hooks on the model after a condition, expected {base_hooks}: "
                         f"a steering hook was not removed. Stopping.")
    return float(r)


def generate(model, tokenizer, text: str, sv, lam: float, base_hooks: int) -> str:
    from utils import device
    enc = tokenizer(text, return_tensors='pt').to(device)
    ids = enc['input_ids'][0]
    assert ids[0].item() == tokenizer.bos_token_id and ids[1].item() != tokenizer.bos_token_id, "expected exactly one BOS"
    kw = dict(do_sample=False, max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=None, top_p=None,
              pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        if sv is None:
            out = model.generate(**enc, **kw)
        else:
            with sv.apply(model, multiplier=lam, min_token_index=0):
                out = model.generate(**enc, **kw)
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit("a steering hook was not removed after generation. Stopping.")
    return tokenizer.decode(out[0, ids.shape[0]:], skip_special_tokens=True)


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def results_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df['semilla'] = df['semilla'].astype('Int64')     # empty for the noiseless conditions, 0 / 1 otherwise
    return df


def ab_table(P: Report, res: pd.DataFrame, name: str, extra: list[tuple[str, float, float]] = ()) -> None:
    sub = res[res['dataset'] == name]
    base = sub.loc[sub['condicion'] == NONE, 'ab'].iloc[0]

    def get(cond, lam, eps=None, seed=None):
        r = sub[(sub['condicion'] == cond) & (sub['lambda'] == lam)]
        if eps is not None:
            r = r[r['epsilon'] == eps]
        if seed is not None:
            r = r[r['semilla'] == seed]
        return r['ab'].to_numpy()

    def line(label, plus, minus):
        P(f"  {label:38s} {plus:8.4f} {minus:8.4f} {plus - minus:+9.4f}")

    P(f"  {'condición':38s} {'λ=+2':>8s} {'λ=−2':>8s} {'(+2)−(−2)':>9s}")
    P(f"  {NONE:38s} {base:8.4f} (λ=0)")
    for cond in (MEAN, PSA0, G1_0):
        line(cond, get(cond, 2.0)[0], get(cond, -2.0)[0])
    for label, plus, minus in extra:
        line(label, plus, minus)
    for cond in (PSA, G1):
        for eps in EPSILONS:
            pl, mi = [], []
            for seed in SEEDS:
                a, b = get(cond, 2.0, eps, seed)[0], get(cond, -2.0, eps, seed)[0]
                pl.append(a)
                mi.append(b)
                line(f"{cond} ε={eps:g} semilla {seed}", a, b)
            line(f"{cond} ε={eps:g} media semillas", float(np.mean(pl)), float(np.mean(mi)))


def sigma_lines(P: Report, acc: Accountant, h: int) -> None:
    P(f"  n={acc.n}, δ={acc.delta:g}, S={acc.S}, calibración {acc.calibration} (Balle-Wang), sensibilidad 2/n = {acc.sens:.5f}")
    P(f"  {'ε':>4s} | {'σ PSA (por coord.)':>18s} {'‖ruido‖/capa':>12s} | {'σ PSA-1G':>9s} {'√5·σ PSA-1G (por coord.)':>24s} "
      f"{'‖ruido‖/capa':>12s} | {'σ_PSA/(√5σ_1G)':>14s}")
    for eps in EPSILONS:
        sp, sg = acc.sigma_psa(eps), acc.sigma_prima(eps)      # sigma_prima = PSA-1G
        P(f"  {eps:4g} | {sp:18.6f} {sp * math.sqrt(h):12.3f} | {sg:9.6f} {math.sqrt(S) * sg:24.6f} "
          f"{math.sqrt(S) * sg * math.sqrt(h):12.3f} | {sp / (math.sqrt(S) * sg):14.3f}")
    P("  (‖ruido‖/capa ≈ σ_coord·√h, esperado por unidad de λ; PSA-1G: σ calibrado sobre el concatenado, aplicado ×√5)")


def vector_lines(P: Report, vecs: list[dict], layers: list[int]) -> None:
    ref = find_vec(vecs, MEAN)['vec']
    P(f"  {'condición':26s} " + " ".join(f"{'L' + str(l) + ' ‖v‖':>9s} {'cos':>6s}" for l in layers))
    for v in vecs:
        nrm = v['vec'].norm(dim=1)
        cs = cosine(v['vec'], ref)
        P(f"  {cond_label(v):26s} " + " ".join(f"{nrm[j].item():9.4f} {cs[j].item():6.3f}" for j in range(len(layers))))
    P("  (cos = coseno con Mean Steering (paper) en la misma capa; diagnóstico interno, no es una métrica del piloto)")


# --------------------------------------------------------------------------------------
# Real run
# --------------------------------------------------------------------------------------
def run(args) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"--out-dir {out_dir} already exists; results are never overwritten. Aborting.")
    t_start = time.time()
    log("=" * 100)
    log(f"PILOT — {MODEL} | datasets {DATASETS} | ε {EPSILONS} | semillas {SEEDS} | λ {LAMBDAS} | δ {DELTA:g}")
    checks: list[str] = []

    # ---------------- preflight, no model ----------------
    sel = read_c_selection(args.c_selection, DATASETS)
    layers = window_layers(EXPECTED_WINDOW)
    for name, s in sel.items():
        log(f"[{name}] c_selection: ventana {s['ventana']} | C_eval {s['C']:g} | Cg_eval {s['Cg']:.4f} (PSA-1G) | "
            f"C {s['C_sel']} | empate {s['empate']} {s['C_empatados']} | recorte_total {s['recorte_total']}")
    rbar_csv = load_rbar_csv(Path(args.rbar_csv))                         # SystemExit if missing
    caches = {name: load_window_cache(Path(args.diffs_dir), name, layers, rbar_csv, args.rbar_csv) for name in DATASETS}
    checks.append(f"caches: metadatos, permutación y R̄ vs {args.rbar_csv} OK para {DATASETS}")
    h = caches[DATASETS[0]]['hidden']
    assert all(c['hidden'] == h for c in caches.values())
    Z = paired_draws(SEEDS, S, h)
    log(f"ruido Z: {tuple(Z.shape)} float64, semillas {SEEDS} (apareadas entre PSA y PSA-1G, ε y datasets)")
    accs, vectors, clipfrac = {}, {}, {}
    for name in DATASETS:
        c = caches[name]
        accs[name] = Accountant(CALIBRATION, c['n'], DELTA, S)
        vectors[name], clipfrac[name] = build_vectors(c['d32'], sel[name]['C'], sel[name]['Cg'], accs[name], Z,
                                                      EPSILONS, SEEDS)
        s_chk = accs[name].sigma_psa(EPSILONS[0])
        check_priv_mean_replication(c['d32'][:, 0, :], C=sel[name]['C'], sigma=s_chk)
        checks.append(check_psa_equals_priv_mean(c['d32'], sel[name]['C'], s_chk, Z[0],
                                                 find_vec(vectors[name], PSA, EPSILONS[0], SEEDS[0])['vec'],
                                                 f"[{name}] PSA ε={EPSILONS[0]:g} s={SEEDS[0]}"))
    checks.append("sweep_epsilon.check_priv_mean_replication OK en la capa 7 de cada dataset")
    tests, overlap = {}, {}
    for name in DATASETS:
        data, src = load_test(name)
        train_q = {r['question'].strip() for r in caches[name]['train_rows']}
        overlap[name] = sum(r['question'].strip() in train_q for r in data)
        tests[name] = (data, src)
        msg = f"[{name}] test {src}: {len(data)} ítems, {overlap[name]} con pregunta igual a uno de los 400 pares del caché"
        if overlap[name]:
            msg += "  <-- AVISO: " + ("in-sample por diseño (no hay test separado)" if name == OWN_DATASET
                                      else "solapamiento train/test")
        log(msg)
        checks.append(msg)

    # ---------------- model ----------------
    out_dir.mkdir(parents=True)
    from utils import diagnose_read_token_index, get_model_and_tokenizer, make_dataset, make_pos_neg_pair
    model, tokenizer = get_model_and_tokenizer(MODEL, load_bfloat=False)   # float32, as run.py
    model.eval()
    if model.config.hidden_size != h or model.config.num_hidden_layers <= max(layers):
        raise SystemExit(f"model hidden {model.config.hidden_size} / {model.config.num_hidden_layers} layers incompatible "
                         f"with the caches (hidden {h}, layers {layers})")
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    base_hooks = n_forward_hooks(model)
    log(f"modelo en {dev}, dtype {dtype}, {model.config.num_hidden_layers} capas, {base_hooks} forward hooks de base")
    test_pairs = {}
    for name in DATASETS:
        c = caches[name]
        pos0, neg0 = make_pos_neg_pair(c['train_rows'][0], tokenizer)
        if (pos0, neg0) != (c['example_pos'], c['example_neg']):
            raise SystemExit(f"[{name}] the first training prompt rebuilt now differs from the one stored in the cache: "
                             f"the prompt format changed since the extraction. Stopping.")
        data, _ = tests[name]
        test_pairs[name] = make_dataset(data, tokenizer)
        log(f"[{name}] TEST POS: {test_pairs[name][0][0]!r}")
        log(f"[{name}] TEST NEG: {test_pairs[name][0][1]!r}")
        diagnose_read_token_index(test_pairs[name][0][0], tokenizer, READ_TOKEN_INDEX,
                                  expected_letter=data[0]['answer_matching_behavior'][1], label=f'[{name} test pos] ')
        diagnose_read_token_index(test_pairs[name][0][1], tokenizer, READ_TOKEN_INDEX,
                                  expected_letter=data[0]['answer_not_matching_behavior'][1], label=f'[{name} test neg] ')
    checks.append("primer prompt de entrenamiento reconstruido == example_pos/neg del caché, para los 3 datasets; "
                  "CHAT_DATE verificado en los prompts de test")

    # ---------------- check 4 ----------------
    log("=" * 100)
    log(f"CHEQUEO 4 — Mean Steering: caché vs train_steering_vector + scaled_mean (run.py:37-41), {CHECK4_DATASET}, "
        f"mismos {caches[CHECK4_DATASET]['n']} pares")
    from steering_vectors import train_steering_vector
    c4 = caches[CHECK4_DATASET]
    log(f"  versiones del caché: {c4['versions']}")
    import steering_vectors
    import transformers
    log(f"  versiones actuales: steering_vectors {steering_vectors.__version__}, transformers {transformers.__version__}, "
        f"torch {torch.__version__}")
    sv_lib = train_steering_vector(model, tokenizer, make_dataset(c4['train_rows'], tokenizer),
                                   read_token_index=READ_TOKEN_INDEX, show_progress=True,
                                   aggregator=scaled_mean_as_in_run_py, layers=layers)
    if sorted(sv_lib.layer_activations) != layers:
        raise SystemExit(f"train_steering_vector returned layers {sorted(sv_lib.layer_activations)}, expected {layers}")
    lib = torch.stack([sv_lib.layer_activations[l].detach().cpu().double() for l in layers])
    c4_lines = compare_mean_vectors(lib, find_vec(vectors[CHECK4_DATASET], MEAN)['vec'], layers, CHECK4_TOL)
    log(f"  assert OK (tol {CHECK4_TOL:g})")
    checks.append(f"chequeo 4 OK: max |diff| por capa ≤ {CHECK4_TOL:g}")
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit("train_steering_vector left hooks on the model. Stopping.")

    # ---------------- A/B ----------------
    rows = []
    lib_ab = {}
    for name in DATASETS:
        log("=" * 100)
        log(f"A/B — {name} ({len(test_pairs[name])} ítems)")
        t0 = time.time()
        r = ab_eval(model, tokenizer, test_pairs[name], None, 0.0, base_hooks)
        rows.append({'dataset': name, 'condicion': NONE, 'epsilon': float('nan'), 'semilla': float('nan'),
                     'lambda': 0.0, 'ab': r})
        log(f"  {NONE:30s} λ=0  ab={r:.4f}")
        for v in vectors[name]:
            sv = to_steering_vector(v['vec'], layers, dtype, dev)
            for lam in LAMBDAS:
                r = ab_eval(model, tokenizer, test_pairs[name], sv, lam, base_hooks)
                rows.append({'dataset': name, 'condicion': v['condicion'], 'epsilon': v['epsilon'],
                             'semilla': v['semilla'], 'lambda': lam, 'ab': r})
                log(f"  {cond_label(v):30s} λ={lam:+g}  ab={r:.4f}")
        if name == CHECK4_DATASET:
            for lam in LAMBDAS:
                lib_ab[lam] = ab_eval(model, tokenizer, test_pairs[name], sv_lib, lam, base_hooks)
                log(f"  {'Mean Steering [train_steering_vector]':30s} λ={lam:+g}  ab={lib_ab[lam]:.4f}")
        results_frame(rows).to_csv(out_dir / 'resultados.csv', index=False)   # partial results survive a crash
        log(f"  [{name}] {time.time() - t0:.0f} s")
    res = results_frame(rows)
    res.to_csv(out_dir / 'resultados.csv', index=False)

    # ---------------- generation ----------------
    log("=" * 100)
    log(f"GENERACIÓN — {GEN_DATASET}, {GEN_N} preguntas (seed {GEN_SEED}), greedy, max_new_tokens={GEN_MAX_NEW_TOKENS}, "
        f"λ={GEN_LAMBDA:+g}")
    gv = vectors[GEN_DATASET]
    gen_conds = [(NONE, None), (MEAN, find_vec(gv, MEAN)), (PSA0, find_vec(gv, PSA0))]
    gen_conds += [(cond_label(find_vec(gv, cond, eps, GEN_NOISE_SEED)), find_vec(gv, cond, eps, GEN_NOISE_SEED))
                  for cond in (PSA, G1) for eps in GEN_EPS]
    G = [f"Generaciones — {MODEL} — {GEN_DATASET} (preguntas del set de forget, sin las opciones A/B)",
         f"greedy (do_sample=False), max_new_tokens={GEN_MAX_NEW_TOKENS}, steering en todos los tokens (min_token_index=0), "
         f"λ={GEN_LAMBDA:+g}, capas {layers}, C={sel[GEN_DATASET]['C']:g}, Cg={sel[GEN_DATASET]['Cg']:.4f}, "
         f"semilla de ruido {GEN_NOISE_SEED}", "Formato: system SYS_PROMPT + user, CHAT_DATE, add_generation_prompt=True", ""]
    for qi, (row, q) in enumerate(gen_questions(load_dataset_json(GEN_DATASET))):
        text = format_generation_prompt(tokenizer, q)
        if qi == 0:
            log(f"  PROMPT: {text!r}")
        G += ["#" * 100, f"### Pregunta {qi + 1} (fila {row} del json): {q}", ""]
        for label, v in gen_conds:
            sv = None if v is None else to_steering_vector(v['vec'], layers, dtype, dev)
            out = generate(model, tokenizer, text, sv, GEN_LAMBDA, base_hooks)
            G += [f"--- {label}", out, ""]
            log(f"  [{qi + 1}] {label}: {out[:100]!r}")
    (out_dir / 'generaciones.txt').write_text("\n".join(G) + "\n", encoding='utf-8')

    # ---------------- summary ----------------
    P = Report()
    P("=" * 100)
    P(f"RESUMEN — piloto de steering — {MODEL}")
    P("=" * 100)
    P(f"capas {layers} | ε {EPSILONS} | semillas {SEEDS} | λ {LAMBDAS} | δ {DELTA:g} | steering en todos los tokens")
    P("PSA: d/max(C,‖d‖) por capa, media + gaussiano por capa a (ε/5, δ/5), sin ×C. "
      "PSA-1G: D/max(Cg,‖D‖) sobre las 5 capas concatenadas, media + UN gaussiano a (ε, δ), ×√5.")
    P("Mean Steering (paper) = scaled_mean de run.py (÷ máx‖d‖ de la capa, media). Métrica A/B = utils.evaluate_model.")
    P("")
    P("CHEQUEOS:")
    for x in checks:
        P(f"  - {x.strip()}")
    P(f"  chequeo 4, por capa ({CHECK4_DATASET}):")
    for x in c4_lines:
        P(f"  {x}")
    for name in DATASETS:
        s = sel[name]
        P("")
        P("=" * 100)
        P(f"{name} | test: {tests[name][1]} ({len(test_pairs[name])} ítems) | C={s['C']:g} Cg={s['Cg']:.4f} | "
          f"c_selection: C={s['C_sel']} empate={s['empate']} {s['C_empatados']} recorte_total={s['recorte_total']}")
        if name == OWN_DATASET:
            P("  AVISO: test in-sample (los mismos 400 pares de los que salen los vectores)")
        P("A/B:")
        extra = [('Mean Steering [train_steering_vector]', lib_ab[2.0], lib_ab[-2.0])] if name == CHECK4_DATASET else []
        ab_table(P, res, name, extra)
        P("σ por coordenada:")
        sigma_lines(P, accs[name], h)
        fp, fg = clipfrac[name]['frac_psa'], clipfrac[name]['frac_g1']
        P(f"Recorte: PSA (C={s['C']:g}) fracción de filas recortadas por capa "
          + " ".join(f"L{l} {fp[j]:.3f}" for j, l in enumerate(layers))
          + f" | PSA-1G (Cg={s['Cg']:.3f}) fracción de filas concatenadas recortadas {fg:.3f}")
        P("‖v_l‖ por capa (vector antes de λ) y coseno con Mean Steering:")
        vector_lines(P, vectors[name], layers)
    P("")
    P(f"tiempo total {time.time() - t_start:.0f} s")
    (out_dir / 'resumen.txt').write_text(P.text(), encoding='utf-8')
    log(f"\nescrito resultados.csv, generaciones.txt, resumen.txt en {out_dir}")


# --------------------------------------------------------------------------------------
# Self-test (CPU, synthetic, nothing written, nothing downloaded)
# --------------------------------------------------------------------------------------
ST_N, ST_H = 400, 64
ST_LAYERS = window_layers(EXPECTED_WINDOW)


def st_diffs(seed: int, scale: float, spread: float = 0.3) -> torch.Tensor:
    """(n, S, h) float32: layer j has its own dominant direction u_j (orthonormal across layers) + isotropic noise;
    row norms ~ scale * lognormal(spread)."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((ST_H, S)))[0].T
    d = 0.6 * U[None] + rng.standard_normal((ST_N, S, ST_H)) / math.sqrt(ST_H)
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    d *= scale * np.exp(spread * rng.standard_normal((ST_N, S, 1)))
    return torch.from_numpy(d.astype(np.float32))


def st_csv(rows: list[dict]) -> io.StringIO:
    return io.StringIO(pd.DataFrame(rows, columns=SEL_COLUMNS).to_csv(index=False))


def selftest(args) -> None:
    t0 = time.time()
    C = EXPECTED_C_EVAL
    Cg = C * math.sqrt(S)
    sq = math.sqrt(S)
    acc = Accountant(CALIBRATION, ST_N, DELTA, S)
    Z = paired_draws(SEEDS, S, ST_H)

    log("#" * 100)
    log(f"SELFTEST A — recorte total (C={C:g}, Cg={Cg:.4f}, normas ~100): PSA norma 1 por fila y capa; "
        f"PSA-1G×√5 norma cuadrática media por capa 1")
    d_big = st_diffs(1, 100.0)
    rp, fp = psa_rows(d_big.double(), C)
    rg, fg = g1_rows(d_big.double(), Cg)
    assert (fp == 1.0).all() and fg == 1.0, (fp, fg)
    err_psa = (rp.norm(dim=2) - 1.0).abs().max().item()
    msq = (sq * rg).pow(2).sum(dim=2).mean(dim=1)                 # per row: mean over layers of ||sqrt(S) row_l||^2
    err_g1 = (msq - 1.0).abs().max().item()
    log(f"  max |‖PSA row_l‖ − 1| = {err_psa:.1e} | max |mean_l ‖√5·PSA-1G row_l‖² − 1| = {err_g1:.1e}")
    assert err_psa < 1e-12 and err_g1 < 1e-12
    vecs_big, _ = build_vectors(d_big, C, Cg, acc, Z, EPSILONS, SEEDS)
    assert torch.allclose(find_vec(vecs_big, PSA0)['vec'], rp.mean(dim=0), rtol=0, atol=1e-15)
    assert torch.allclose(find_vec(vecs_big, G1_0)['vec'], sq * rg.mean(dim=0), rtol=0, atol=1e-15)
    log("  OK (and the noiseless vectors of build_vectors are exactly the means of these rows)")

    log("#" * 100)
    log("SELFTEST B — σ de PSA y PSA-1G == los de sweep_c.py para el mismo ε; vectores == los de sweep_c.py")
    d = st_diffs(2, 1.8, spread=0.5)                               # partial clipping in both mechanisms
    vecs, frac = build_vectors(d, C, Cg, acc, Z, EPSILONS, SEEDS)
    log(f"  fracción recortada: PSA por capa {np.round(frac['frac_psa'], 3)} | PSA-1G {frac['frac_g1']:.3f}")
    assert 0.05 < frac['frac_psa'].min() and frac['frac_psa'].max() < 0.95 and 0.05 < frac['frac_g1'] < 0.95
    dd = d.double()
    m_psa = psa_rows(dd, C)[0].mean(dim=0)
    m_g1 = g1_rows(dd, Cg)[0].mean(dim=0)
    # sweep_c.sweep_one, in original units (sweep_c.py:206-225); 'prima' there = PSA-1G here
    sc_m_psa = torch.stack([clip_rows(dd[:, j, :], C)[0].mean(dim=0) for j in range(S)])
    sc_m_g1 = clip_rows(dd.reshape(ST_N, -1), Cg)[0].mean(dim=0).reshape(S, ST_H)
    cos_sc, _ = sweep_one(dd, Z, acc, [C], EPSILONS)
    ref = dd.mean(dim=0)
    worst = {'sigma': 0.0, 'noise': 0.0, 'vec': 0.0, 'cos': 0.0}
    for e, eps in enumerate(EPSILONS):
        s_psa_sc, s_g1_sc = acc.sigma('psa', eps), acc.sigma('prima', eps)     # exactly the calls of sweep_c.py:224
        s_psa_ind = sigma_analytic(eps / S, DELTA / S, 2.0 / ST_N)             # independent: the formula, not the class
        s_g1_ind = sigma_analytic(eps, DELTA, 2.0 / ST_N)
        V_sc_psa, _ = noisy_vectors(sc_m_psa, np.full(S, C), s_psa_sc, Z)
        V_sc_g1, _ = noisy_vectors(sc_m_g1, np.full(S, Cg), s_g1_sc, Z)
        for k, seed in enumerate(SEEDS):
            vp, vg = find_vec(vecs, PSA, eps, seed), find_vec(vecs, G1, eps, seed)
            assert vp['sigma'] == s_psa_sc == s_psa_ind and vg['sigma'] == s_g1_sc == s_g1_ind, (eps, vp['sigma'], vg['sigma'])
            assert vp['sigma_coord'] == s_psa_sc and vg['sigma_coord'] == sq * s_g1_sc
            worst['noise'] = max(worst['noise'], (vp['vec'] - m_psa - s_psa_sc * Z[k]).abs().max().item(),
                                 (vg['vec'] / sq - m_g1 - s_g1_sc * Z[k]).abs().max().item())
            worst['vec'] = max(worst['vec'], ((C * vp['vec'] - V_sc_psa[k]).abs().max() / V_sc_psa[k].abs().max()).item(),
                               ((Cg * vg['vec'] / sq - V_sc_g1[k]).abs().max() / V_sc_g1[k].abs().max()).item())
            worst['cos'] = max(worst['cos'],
                               float(np.abs(cosine(vp['vec'], ref).numpy() - cos_sc[0, e, MECHS.index('psa'), k]).max()),
                               float(np.abs(cosine(vg['vec'], ref).numpy() - cos_sc[0, e, MECHS.index('prima'), k]).max()))
        log(f"  ε={eps:g}: σ_PSA {s_psa_sc:.6f} (= σ(ε/5, δ/5, 2/n) {s_psa_ind:.6f}) | σ_PSA-1G {s_g1_sc:.6f} "
            f"(= σ(ε, δ, 2/n) {s_g1_ind:.6f}), aplicado ×√5 = {sq * s_g1_sc:.6f}")
    log(f"  max |ruido realizado − σ·z| = {worst['noise']:.1e} | max rel |C·PSA − PSA de sweep_c|, |Cg·PSA-1G/√5 − "
        f"PSA-1G de sweep_c| = {worst['vec']:.1e} | max |cos − cos de sweep_c.sweep_one| = {worst['cos']:.1e}")
    assert worst['noise'] < 1e-12 and worst['vec'] < 1e-12 and worst['cos'] < 1e-12
    check_priv_mean_replication(d[:, 0, :], C=C, sigma=acc.sigma_psa(EPSILONS[0]))
    check_psa_equals_priv_mean(d, C, acc.sigma_psa(EPSILONS[0]), Z[0], find_vec(vecs, PSA, EPSILONS[0], SEEDS[0])['vec'],
                               'synthetic')
    mp = find_vec(vecs, MEAN)['vec']
    dn = d.double().numpy()
    indep = np.stack([(dn[:, j] / np.linalg.norm(dn[:, j], axis=1).max()).mean(axis=0) for j in range(S)])
    err_mean = float(np.abs(mp.numpy() - indep).max())
    log(f"  Mean Steering vs recomputación numpy independiente: max |diff| = {err_mean:.1e}")
    assert err_mean < 1e-6
    log("  OK")

    log("#" * 100)
    log("SELFTEST C — con ε muy grande PSA y PSA-1G se acercan a sus versiones sin ruido")
    big = [1e2, 1e3, 1e4]
    vecs_c, _ = build_vectors(d, C, Cg, acc, Z, big, SEEDS)
    for cond, clean in ((PSA, PSA0), (G1, G1_0)):
        c0 = find_vec(vecs_c, clean)['vec']
        rel, cmin = [], []
        for eps in big:
            r = max((find_vec(vecs_c, cond, eps, s)['vec'] - c0).norm().item() / c0.norm().item() for s in SEEDS)
            cm = min(cosine(find_vec(vecs_c, cond, eps, s)['vec'], c0).min().item() for s in SEEDS)
            rel.append(r)
            cmin.append(cm)
        log(f"  {cond}: ‖v − v_sin_ruido‖/‖v_sin_ruido‖ en ε {big}: {['%.1e' % x for x in rel]} | cos mínimo por capa "
            f"{['%.6f' % x for x in cmin]}")
        assert rel[0] > rel[1] > rel[2] and rel[2] < 1e-2 and cmin[2] > 0.9999, (cond, rel, cmin)
    log("  OK")

    log("#" * 100)
    log("SELFTEST D — lectura de c_selection.csv (y PSA-1G = 'prima' de sweep_c: Cg_eval = C_eval·√5)")
    good = {'model': MODEL, 'ventana': EXPECTED_WINDOW, 'C': float('nan'), 'Cg': float('nan'), 'empate': True,
            'C_empatados': '2 3', 'C_eval': 2.0, 'Cg_eval': 2.0 * sq, 'recorte_total': True}
    rows = [{**good, 'dataset': n} for n in DATASETS]
    out = read_c_selection(st_csv(rows), DATASETS)
    assert all(out[n]['C'] == 2.0 and abs(out[n]['Cg'] - 2 * sq) < 1e-12 and out[n]['empate'] and out[n]['recorte_total']
               for n in DATASETS)
    bad_cases = {'ventana distinta': {'ventana': '8-12'}, 'Cg_eval ≠ C_eval·√5': {'Cg_eval': 2.0 * sq + 1e-3},
                 'C_eval ≠ 2': {'C_eval': 3.0, 'Cg_eval': 3.0 * sq}}
    for label, patch in bad_cases.items():
        try:
            read_c_selection(st_csv([{**r, **(patch if r['dataset'] == 'refusal' else {})} for r in rows]), DATASETS)
        except SystemExit as e:
            log(f"  (esperado) {label}: {str(e)[:90]}")
        else:
            raise AssertionError(f"read_c_selection accepted: {label}")
    try:
        read_c_selection(io.StringIO(pd.DataFrame(rows).drop(columns=['C_eval']).to_csv(index=False)), DATASETS)
    except SystemExit as e:
        log(f"  (esperado) sin C_eval: {str(e)[:90]}")
    else:
        raise AssertionError("read_c_selection accepted a csv without C_eval")
    log("  OK")

    log("#" * 100)
    log(f"SELFTEST E — assert del chequeo 4 (tol {CHECK4_TOL:g})")
    compare_mean_vectors(mp.clone(), mp, ST_LAYERS, CHECK4_TOL)
    pert = mp.clone()
    pert[2, 5] += 0.5 * CHECK4_TOL
    compare_mean_vectors(pert, mp, ST_LAYERS, CHECK4_TOL)
    pert[2, 5] += 1.5 * CHECK4_TOL
    try:
        compare_mean_vectors(pert, mp, ST_LAYERS, CHECK4_TOL)
    except AssertionError as e:
        log(f"  (esperado) {str(e)[:110]}")
    else:
        raise AssertionError("check 4 accepted a 2e-5 difference")
    log("  OK")

    log("#" * 100)
    log("SELFTEST F — SteeringVector.apply en un Llama aleatorio diminuto (12 capas, h=64): el bloque j del vector entra "
        "en la capa 7+j, en todos los tokens, y el hook se retira")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=ST_H, intermediate_size=128, num_hidden_layers=12, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    tiny = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 10))
    base_hooks = n_forward_hooks(tiny)

    def layer_inputs(sv, lam):
        cap = {}

        def mk(i):
            def pre(_m, a, kw):
                cap[i] = (a[0] if a else kw['hidden_states']).detach().clone()
            return pre

        hs = [tiny.model.layers[i].register_forward_pre_hook(mk(i), with_kwargs=True) for i in range(12)]
        try:
            with torch.no_grad():
                if sv is None:
                    tiny(ids)
                else:
                    with sv.apply(tiny, multiplier=lam, min_token_index=0):
                        tiny(ids)
        finally:
            for x in hs:
                x.remove()
        return cap

    def output_jumps(sv, lam):
        """Per window layer, its output right after the steering hook minus right before it, in the same forward pass:
        'before' hooks are registered before sv.apply, 'after' hooks inside it, and forward hooks run in registration order."""
        before, after = {}, {}

        def mk(store, i):
            def hook(_m, _a, out):
                store[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return hook

        hs = [tiny.model.layers[i].register_forward_hook(mk(before, i)) for i in ST_LAYERS]
        try:
            with torch.no_grad(), sv.apply(tiny, multiplier=lam, min_token_index=0):
                hs += [tiny.model.layers[i].register_forward_hook(mk(after, i)) for i in ST_LAYERS]
                tiny(ids)
        finally:
            for x in hs:
                x.remove()
        return {i: after[i] - before[i] for i in ST_LAYERS}

    clean = layer_inputs(None, 0.0)
    for cond in (PSA, G1):
        v = find_vec(vecs, cond, EPSILONS[0], SEEDS[0])['vec']
        sv = to_steering_vector(v, ST_LAYERS, torch.float32, 'cpu')
        assert sorted(sv.layer_activations) == ST_LAYERS
        for lam in LAMBDAS:
            st = layer_inputs(sv, lam)
            assert n_forward_hooks(tiny) == base_hooks, "hook not removed"
            for i in range(ST_LAYERS[0] + 1):
                assert torch.equal(st[i], clean[i]), f"layer {i} input changed (steering must start at layer 7's output)"
            delta = st[ST_LAYERS[0] + 1] - clean[ST_LAYERS[0] + 1]      # input of layer 8 = output of layer 7
            err0 = (delta - (lam * v[0].float()).expand_as(delta)).abs().max().item()
            jumps = output_jumps(sv, lam)
            assert n_forward_hooks(tiny) == base_hooks, "hook not removed"
            errs = [(jumps[l] - (lam * v[j].float()).expand_as(jumps[l])).abs().max().item() for j, l in enumerate(ST_LAYERS)]
            assert err0 < 1e-5 and max(errs) < 1e-5, (cond, lam, err0, errs)
            log(f"  {cond} λ={lam:+g}: salto en la salida de la capa 7+j == λ·bloque j en los 10 tokens para j=0..4 "
                f"(max err {max(errs):.1e}); entrada de la capa 8 − sin steering == λ·bloque 0 ({err0:.1e}); "
                f"capas 0-7 sin cambios; hooks retirados")
    log("  OK")

    log("#" * 100)
    log("SELFTEST G — datos reales (solo json, sin modelo): tests, balance de tofu, preguntas de generación")
    for name in DATASETS:
        data, src = load_test(name)
        assert all({r['answer_matching_behavior'], r['answer_not_matching_behavior']} == {'(A)', '(B)'} for r in data)
        log(f"  {name}: {len(data)} ítems de {src}; letras {dict(Counter(r['answer_matching_behavior'] for r in data))}")
    qs = gen_questions(load_dataset_json(GEN_DATASET))
    assert len(qs) == GEN_N and all('Choices' not in q and '(A)' not in q for _, q in qs)
    assert qs == gen_questions(load_dataset_json(GEN_DATASET)), "generation questions must be deterministic"
    for row, q in qs:
        log(f"  generación, fila {row}: {q!r}")
    log("  OK")
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
    parser.add_argument("--diffs-dir", default=f"sweep_{MODEL_TAG}/diffs")
    parser.add_argument("--c-selection", default=f"sweep_c_{MODEL_TAG}/c_selection.csv")
    parser.add_argument("--rbar-csv", default=f"resultados_rbar_{MODEL_TAG}/rbar_results.csv",
                        help="mandatory: R̄ recomputed from every cache must match it")
    parser.add_argument("--out-dir", default=f"pilot_{MODEL_TAG}", help="must not exist")
    parser.add_argument("--selftest", action='store_true', help="CPU, synthetic, no model download, writes nothing")
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
        return
    if not Path(args.c_selection).is_file():
        raise SystemExit(f"{args.c_selection} not found")
    run(args)


if __name__ == '__main__':
    main()
