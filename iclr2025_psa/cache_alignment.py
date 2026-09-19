"""
Step 0 shared by the analyses that run on sweep_epsilon.py diffs caches (test_template_effect.py,
test_letter_effect.py): load a cache, reproduce the shuffle that put its rows in order, verify that
mapping independently, and cross-check R̄ against measure_rbar.py's csv. No model, no GPU.

Row order of a cache. measure_rbar.load_pairs (measure_rbar.py:129-139, also used by
sweep_epsilon.extract_diffs at sweep_epsilon.py:170) does
      data = json.load(...); random.Random(seed).shuffle(data); data[:n_samples]
and steering_vectors.extract_activations keeps that order (batchify yields data[i:i+batch_size]
sequentially, steering_vectors/utils.py:9-23; torch.concat(pos_acts[layer]) concatenates in call order,
sweep_epsilon.py:203-207). random.shuffle only calls randbelow(i+1) for i = len-1 .. 1, so it depends on
len(data) and the seed only: shuffling range(total) with the same seed reproduces the permutation exactly.

Independent verification. The cache stores the formatted prompts of its row 0 (sweep_epsilon.py:211,
pairs[0]); the Llama-3 chat template embeds the question verbatim (trimmed) and utils._format_llama3
asserts the string ends exactly in the answer, e.g. "(A)". So example_pos / example_neg must contain the
question of json[perm[0]] and of no other json row, and must end in that row's two answer letters. On top
of that, R̄ recomputed from the cache with measure_rbar's float32 arithmetic must match the csv to
RBAR_TOL on every layer with the same n. Any failure raises SystemExit: with a wrong mapping every
downstream number is plausible and wrong.
"""
import json
import random
from pathlib import Path

import pandas as pd
import torch

# Mirrors measure_rbar.PSA_DATASETS / OWN_DATASET (not imported: measure_rbar pulls steering_vectors and
# transformers at import time, which the cache-only analyses do not need).
PSA_DATASETS = ['sycophancy', 'hallucination', 'refusal', 'myopic-reward',
                'survival-instinct', 'coordinate-other-ais', 'corrigible-neutral-HHH']
OWN_DATASET = 'tofu-forget10'
DEFAULT_DATASETS = PSA_DATASETS + [OWN_DATASET]

RBAR_TOL = 1e-4   # same tolerance as sweep_epsilon.crosscheck_rbar


def log(msg: str = '') -> None:
    print(msg, flush=True)


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


def dataset_json_path(name: str) -> Path:
    """Same rule as measure_rbar.resolve_dataset_path: a dataset under ./datasets/generate/ or a json path."""
    p = Path(name)
    if p.suffix == '.json' and p.is_file():
        return p
    return Path('datasets') / 'generate' / name / 'generate_dataset.json'


def load_dataset_json(name: str) -> list[dict]:
    with open(dataset_json_path(name), encoding='utf-8') as f:
        return json.load(f)


def shuffle_permutation(total: int, n: int, seed: int) -> list[int]:
    """perm[k] = json row index of cached row k (measure_rbar.py:134 shuffle, :139 slice)."""
    idx = list(range(total))
    random.Random(seed).shuffle(idx)
    return idx[:n]


def verify_permutation(cache: dict, ab: list[dict], perm: list[int]) -> None:
    """SystemExit unless the cache's example prompts identify json[perm[0]] unambiguously."""
    problems = []
    total, n = cache['total'], cache['n']
    if len(ab) != total:
        problems.append(f"dataset json has {len(ab)} pairs but the cache says total={total}")
    if len(perm) != n or len(set(perm)) != n:
        problems.append(f"permutation has {len(perm)} entries ({len(set(perm))} distinct), expected n={n}")
    if problems:
        raise SystemExit(f"[{cache['dataset']}] index alignment FAILED before the content checks: " + "; ".join(problems))

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
        raise SystemExit(f"[{cache['dataset']}] index alignment FAILED: " + "; ".join(problems) +
                         ". Stopping: with a wrong mapping every number below would be plausible and wrong.")
    log(f"  alignment OK: cached row 0 = json row {perm[0]}; example prompts contain that question only and end in "
        f"{first['answer_matching_behavior']} / {first['answer_not_matching_behavior']}; tail {ex_pos[-40:]!r}")


def rbar_float32_as_measure_rbar(d32: torch.Tensor) -> tuple[float, float, float]:
    """Same arithmetic as measure_rbar.RbarAggregator / sweep_epsilon.rbar_from_cache (float32 torch)."""
    mean_vec = torch.mean(d32, dim=0)
    norm_mean = torch.norm(mean_vec).item()
    mean_norm = torch.norm(d32, dim=1).mean().item()
    return norm_mean / mean_norm, norm_mean, mean_norm


def load_rbar_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"rbar csv {csv_path} not found; the R̄ cross-check is mandatory (pass --rbar-csv)")
    return pd.read_csv(csv_path)


def crosscheck_rbar(cache: dict, csv: pd.DataFrame, csv_label: str = 'rbar csv') -> None:
    """SystemExit unless R̄ recomputed from the cache matches the csv on every layer (tolerance RBAR_TOL, same n)."""
    ref = csv[(csv['model'] == cache['model']) & (csv['dataset'] == cache['dataset'])].set_index('layer')
    if ref.empty:
        raise SystemExit(f"no rows for model={cache['model']!r} dataset={cache['dataset']!r} in {csv_label}")
    worst, bad = 0.0, []
    for layer in cache['layers']:
        if layer not in ref.index:
            raise SystemExit(f"[{cache['dataset']}] layer {layer} is in the cache but not in {csv_label}")
        if int(ref.loc[layer, 'n']) != cache['n']:
            raise SystemExit(f"[{cache['dataset']}] layer {layer}: csv n={int(ref.loc[layer, 'n'])} != cache n={cache['n']}")
        rbar, _, _ = rbar_float32_as_measure_rbar(cache['diffs'][layer])
        diff = abs(rbar - float(ref.loc[layer, 'rbar']))
        worst = max(worst, diff)
        if diff > RBAR_TOL:
            bad.append(layer)
    if bad:
        raise SystemExit(f"[{cache['dataset']}] R̄ recomputed from the cache differs from {csv_label} by up to {worst:.2e} "
                         f"(> {RBAR_TOL:g}) on layers {bad}. The cache is not what was measured; stopping.")
    log(f"  R̄ cross-check vs {csv_label} OK on {len(cache['layers'])} layers (max |diff| = {worst:.1e}, n = {cache['n']})")


def align(cache: dict, ab: list[dict], csv: pd.DataFrame, csv_label: str = 'rbar csv') -> list[int]:
    """Full step 0 for one cache: permutation + independent verification + R̄ cross-check. Returns perm."""
    perm = shuffle_permutation(cache['total'], cache['n'], cache['seed'])
    verify_permutation(cache, ab, perm)
    crosscheck_rbar(cache, csv, csv_label)
    return perm
