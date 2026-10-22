"""
Level 3 steering evaluation: PSA vs PSA-1G on the 8 datasets, lambda +1 / +2, 10 paired noise seeds, one model per run
(--model-tag 1b / 3b / 8b).

Naming. The mechanism called "PSA-prima" / 'prima' in sweep_epsilon.py, sweep_c.py and their CSVs (Accountant.sigma_prima,
MECHS 'prima'; c_selection.csv's Cg / Cg_eval are its clip bound) is called PSA-1G here, as in eval_pilot.py: ONE Gaussian
on the concatenated layers. Where this script calls those modules or reads those CSVs it maps prima -> PSA-1G in a comment.

Vectors: exactly those of eval_pilot.py (its build_vectors is reused unchanged), but on the window W_k, C_eval and Cg_eval
of EACH dataset k, read from sweep_c_<tag>/c_selection.csv (no window or C is assumed; 3B and 8B have per-dataset windows):
  Mean Steering (paper)  v_l = mean_i d_i^l / max_i ||d_i^l||                              run.py:37-41, no noise
  PSA sin ruido          v_l = mean_i d_i^l / max(C, ||d_i^l||)
  PSA                    v_l = mean_i d_i^l / max(C, ||d_i^l||) + sigma_PSA(eps) z_l        NOT multiplied by C
  PSA-1G sin ruido       v   = sqrt(5) * mean_i D_i / max(Cg, ||D_i||)                      D_i = the 5 layers of W_k
  PSA-1G                 v   = sqrt(5) * (mean_i D_i / max(Cg, ||D_i||) + sigma_1G(eps) z)   x sqrt(5) = post-processing
  sigma_PSA = sigma(eps/5, delta/5, 2/n), sigma_1G = sigma(eps, delta, 2/n), Balle-Wang (sweep_epsilon.Accountant), real n
  of each dataset, delta = 1e-5. z: sweep_c.paired_draws, one N(0, I_{5 x h}) per seed 0..9, shared by PSA and PSA-1G,
  every eps and every dataset (seeds 0 and 1 are the draws of the pilot).
Steering: SteeringVector.apply(model, multiplier=lambda, min_token_index=0) on the layers of W_k (every token).
A/B metric: utils.evaluate_model through eval_pilot.ab_eval (which also checks that no steering hook is left).

Conditions per dataset (127 A/B evaluations, no generation, no negative lambda):
  sin steering (once, lambda = 0); Mean Steering (paper), PSA sin ruido, PSA-1G sin ruido with lambda +1, +2;
  PSA and PSA-1G at eps 1, 4, 8 x seeds 0..9 with lambda +1, +2.
Test sets: datasets/test/<d>/test_dataset_ab.json, complete, for the 7 public datasets (overlap with the cached pairs must
be 0). tofu-forget10 has no test: its 400 generate pairs are used as they are (200/200 letters) -> IN-SAMPLE, flagged.

Checks (all abort): cache metadata, shuffle reproduction and R̄ vs the R̄ csv (mandatory); c_selection: 5 consecutive layers,
Cg_eval == C_eval * sqrt(5), recorte_total present (False -> visible warning only); check_priv_mean_replication and
PSA == run.py priv_mean with the same z, per dataset; test/cache overlap 0 on the public datasets; first training prompt
rebuilt == the cached one; CHAT_DATE in every test prompt; no steering hook left after each evaluation; check 4 (Mean
Steering from the cache vs train_steering_vector + scaled_mean, tol 1e-5 per layer) once, on hallucination, in its window.

Resumption: if --out-dir exists, config.json (model, datasets, windows, C, eps, seeds, lambda, delta, library versions,
sha256 of c_selection.csv, of every cache and of every test json) must match the current one exactly, or the run aborts.
resultados.csv is appended after EVERY evaluation (atomic replace; the previous text is kept byte for byte, rows are never
rewritten); rows already present (key dataset, condicion, epsilon, semilla, lambda) are skipped. resumen.txt is rebuilt
from resultados.csv at the end of every complete run.

    python eval_level3.py --selftest                 # CPU, synthetic, nothing downloaded, writes only to a system temp dir
    python eval_level3.py --model-tag 1b [--pilot-csv pilot_1b/resultados.csv]
    python eval_level3.py --model-tag 8b             # sweep_8b/diffs, sweep_c_8b/c_selection.csv,
                                                     # resultados_rbar_8b/rbar_results.csv -> nivel3_8b/

Outputs (--out-dir, default nivel3_<tag>): config.json, resultados.csv (dataset, ventana, condicion, epsilon, semilla,
lambda, ab), resumen.txt.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cache_alignment import (DEFAULT_DATASETS, OWN_DATASET, PSA_DATASETS, crosscheck_rbar, dataset_json_path,
                             load_dataset_json, load_rbar_csv, shuffle_permutation, verify_permutation)
import eval_pilot
from eval_pilot import (CHECK4_DATASET, CHECK4_TOL, G1, G1_0, MEAN, NONE, PSA, PSA0, Report, ab_eval, build_vectors,
                        check_psa_equals_priv_mean, compare_mean_vectors, find_vec, g1_rows, load_test, log,
                        n_forward_hooks, psa_rows, scaled_mean_as_in_run_py, to_steering_vector)
from measure_rbar import READ_TOKEN_INDEX
# sweep_epsilon / sweep_c call PSA-1G "prima": Accountant.sigma_prima, SEL_COLUMNS Cg / Cg_eval (see the module docstring).
from sweep_c import MODELS, SEL_COLUMNS, paired_draws
from sweep_epsilon import Accountant, check_priv_mean_replication, ci95, cosine, load_cache

DATASETS = DEFAULT_DATASETS                  # the 7 public PSA datasets, then tofu-forget10
PUBLIC = PSA_DATASETS
S = 5
EPSILONS = [1.0, 4.0, 8.0]
SEEDS = list(range(10))
LAMBDAS = [1.0, 2.0]
DELTA = 1e-5
CALIBRATION = 'analytic'
N_SAMPLES = 400
SHUFFLE_SEED = 42
N_EVALS = 1 + 3 * len(LAMBDAS) + 2 * len(EPSILONS) * len(SEEDS) * len(LAMBDAS)
RESULT_COLUMNS = ['dataset', 'ventana', 'condicion', 'epsilon', 'semilla', 'lambda', 'ab']
PILOT_DATASETS = ['hallucination', 'refusal', OWN_DATASET]
PILOT_LAMBDA = 2.0
PILOT_SEEDS = [0, 1]
PILOT_TOL = 1e-6
assert N_EVALS == 127 and all(lam > 0 for lam in LAMBDAS)
assert eval_pilot.S == S                     # build_vectors / psa_rows / g1_rows use eval_pilot.S


class Interrupted(Exception):
    """Raised by the selftest's fake evaluator to simulate a job killed in the middle of a run."""


# --------------------------------------------------------------------------------------
# Keys and formatting of resultados.csv
# --------------------------------------------------------------------------------------
def fmt_float(x) -> str:
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else repr(float(x))


def fmt_seed(x) -> str:
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else str(int(x))


def task_key(dataset: str, cond: str, eps, seed, lam) -> tuple:
    return dataset, cond, fmt_float(eps), fmt_seed(seed), fmt_float(lam)


def key_from_strings(r: dict) -> tuple:
    """Key of a row read back from the csv (strings), normalised like task_key."""
    e, s = r['epsilon'], r['semilla']
    return (r['dataset'], r['condicion'], '' if e == '' else repr(float(e)), '' if s == '' else str(int(float(s))),
            repr(float(r['lambda'])))


def serialize_row(row: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(
        [row['dataset'], row['ventana'], row['condicion'], fmt_float(row['epsilon']), fmt_seed(row['semilla']),
         fmt_float(row['lambda']), repr(float(row['ab']))])
    return buf.getvalue()


HEADER = ','.join(RESULT_COLUMNS) + '\n'


class ResultsStore:
    """resultados.csv, append-only. Every append re-reads the file, checks that it is exactly what this process last wrote,
    writes old text + the new line to a temporary file and os.replace()s it: a crash leaves either the old or the new file,
    never a partial line, and existing rows are never rewritten."""

    def __init__(self, path: Path, expected: dict[tuple, str]):
        self.path = path
        self.expected = expected                       # key -> ventana of every evaluation of this configuration
        if path.is_file():
            self.text = self._read()
        else:
            self.text = HEADER
            self._write(HEADER)
        self.keys = self._parse(self.text)

    def _read(self) -> str:
        with open(self.path, encoding='utf-8', newline='') as f:
            return f.read()

    def _write(self, text: str) -> None:
        tmp = self.path.with_name(self.path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _parse(self, text: str) -> set:
        if not text.startswith(HEADER):
            raise SystemExit(f"{self.path}: header is not {HEADER.strip()!r}. Stopping.")
        if not text.endswith('\n'):
            raise SystemExit(f"{self.path}: the last line is incomplete. Stopping (fix the file by hand).")
        keys = set()
        for i, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):
            if None in r or any(v is None for v in r.values()):
                raise SystemExit(f"{self.path} line {i}: wrong number of fields {r}. Stopping.")
            k = key_from_strings(r)
            if k in keys:
                raise SystemExit(f"{self.path} line {i}: duplicated key {k}. Stopping.")
            if k not in self.expected:
                raise SystemExit(f"{self.path} line {i}: key {k} is not an evaluation of this configuration. Stopping.")
            if r['ventana'] != self.expected[k]:
                raise SystemExit(f"{self.path} line {i}: ventana {r['ventana']!r} != {self.expected[k]!r}. Stopping.")
            ab = float(r['ab'])
            if not 0.0 <= ab <= 1.0:
                raise SystemExit(f"{self.path} line {i}: ab = {ab} outside [0, 1]. Stopping.")
            keys.add(k)
        return keys

    def __contains__(self, key: tuple) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def append(self, row: dict) -> None:
        k = task_key(row['dataset'], row['condicion'], row['epsilon'], row['semilla'], row['lambda'])
        if k in self.keys:
            raise SystemExit(f"refusing to write {k} twice")
        if k not in self.expected or self.expected[k] != row['ventana']:
            raise SystemExit(f"{k} / {row['ventana']} is not an evaluation of this configuration")
        if self._read() != self.text:
            raise SystemExit(f"{self.path} was modified by someone else during the run. Stopping.")
        new = self.text + serialize_row(row)
        self._write(new)
        self.text = new
        self.keys.add(k)


def read_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


# --------------------------------------------------------------------------------------
# c_selection.csv (rewrite of eval_pilot.read_c_selection without the fixed 1B model / window 7-11 / C_eval 2)
# --------------------------------------------------------------------------------------
def parse_window(w: str, label: str) -> list[int]:
    m = re.fullmatch(r'(\d+)-(\d+)', w.strip())
    if m is None:
        raise SystemExit(f"{label}: ventana {w!r} is not 'a-b'")
    a, b = int(m.group(1)), int(m.group(2))
    if b - a != S - 1:
        raise SystemExit(f"{label}: ventana {w!r} has {b - a + 1} layers, expected {S} consecutive layers")
    return list(range(a, b + 1))


def read_c_selection(src, model: str, datasets: list[str]) -> dict[str, dict]:
    """src: path or file-like. Window, C_eval and Cg_eval of every dataset (Cg / Cg_eval = clip bound of PSA-1G, 'prima'
    in sweep_c.py). recorte_total must be present (True/False); False only sets a warning flag."""
    sel = pd.read_csv(src, keep_default_na=False, na_values=[''])
    missing = [c for c in SEL_COLUMNS if c not in sel.columns]
    if missing:
        raise SystemExit(f"c_selection.csv has no columns {missing}; run sweep_c.py --recompute-selection on it first")
    sub = sel[sel['model'] == model]
    out = {}
    for name in datasets:
        r = sub[sub['dataset'] == name]
        if len(r) != 1:
            raise SystemExit(f"c_selection.csv: {len(r)} rows for ({model}, {name}), expected 1")
        r = r.iloc[0]
        label = f"c_selection.csv {name}"
        layers = parse_window(str(r['ventana']), label)
        C, Cg = float(r['C_eval']), float(r['Cg_eval'])
        if not (math.isfinite(C) and C > 0):
            raise SystemExit(f"{label}: C_eval {r['C_eval']!r} is not a positive number")
        if abs(Cg - C * math.sqrt(S)) > 1e-9 * C * math.sqrt(S):
            raise SystemExit(f"{label}: Cg_eval {Cg!r} != C_eval * sqrt({S}) = {C * math.sqrt(S)!r}")
        rt = str(r['recorte_total'])
        if rt not in ('True', 'False'):
            raise SystemExit(f"{label}: recorte_total is {r['recorte_total']!r}, expected True or False")
        out[name] = {'ventana': f"{layers[0]}-{layers[-1]}", 'layers': layers, 'C': C, 'Cg': Cg, 'C_sel': r['C'],
                     'empate': str(r['empate']) == 'True', 'C_empatados': str(r['C_empatados']),
                     'recorte_total': rt == 'True'}
    return out


def selection_line(name: str, s: dict) -> str:
    return (f"[{name}] c_selection: ventana {s['ventana']} (capas {s['layers']}) | C_eval {s['C']:g} | Cg_eval {s['Cg']:.4f} "
            f"(PSA-1G) | C {s['C_sel']} | empate {s['empate']} {s['C_empatados']} | recorte_total {s['recorte_total']}")


# --------------------------------------------------------------------------------------
# Data (rewrite of eval_pilot.load_window_cache without the fixed 1B model)
# --------------------------------------------------------------------------------------
def load_window_cache(diffs_dir: Path, name: str, layers: list[int], model: str, rbar_csv: pd.DataFrame,
                      csv_label: str) -> dict:
    path = diffs_dir / f"{name}.pt"
    if not path.is_file():
        raise SystemExit(f"no cache {path} (extract it with sweep_epsilon.py --extract-only)")
    cache = load_cache(path, model, N_SAMPLES, SHUFFLE_SEED, layers)       # model / seed / read_token_index / n / layers
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
    log(f"[{name}] cache {path} | n={n} of {cache['total']} | hidden {h} | ventana {layers} | versions: steering_vectors "
        f"{cache.get('steering_vectors_version')}, transformers {cache.get('transformers_version')}, "
        f"torch {cache.get('torch_version')}")
    return {'path': path, 'n': n, 'hidden': h,
            'd32': torch.stack([cache['diffs'][l] for l in layers], dim=1).contiguous(),
            'train_rows': [data[i] for i in perm], 'example_pos': cache['example_pos'],
            'example_neg': cache['example_neg'],
            'versions': {k: cache.get(k) for k in ('steering_vectors_version', 'transformers_version', 'torch_version')}}


def test_json_path(name: str) -> Path:
    """The file load_test reads (eval_pilot.py:301-310)."""
    return dataset_json_path(name) if name == OWN_DATASET else Path('datasets') / 'test' / name / 'test_dataset_ab.json'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Preflight (no model)
# --------------------------------------------------------------------------------------
def preflight(paths: dict, model: str, checks: list[str]) -> dict:
    sel = read_c_selection(paths['c_selection'], model, DATASETS)
    for name in DATASETS:
        log(selection_line(name, sel[name]))
        if not sel[name]['recorte_total']:
            log(f"  <-- AVISO: [{name}] recorte_total = False en c_selection.csv (C_eval no recorta el 100% de las filas "
                f"de 'otros'); se evalúa igual")
    windows = sorted({s['ventana'] for s in sel.values()})
    checks.append(f"c_selection: 5 capas consecutivas, Cg_eval = C_eval·√5 y recorte_total presente en los "
                  f"{len(DATASETS)} datasets; ventanas usadas {windows}")
    rbar_csv = load_rbar_csv(Path(paths['rbar_csv']))                     # SystemExit if missing
    caches = {name: load_window_cache(Path(paths['diffs_dir']), name, sel[name]['layers'], model, rbar_csv,
                                      paths['rbar_csv']) for name in DATASETS}
    checks.append(f"caches: metadatos, permutación y R̄ vs {paths['rbar_csv']} OK en los {len(DATASETS)} datasets")
    h = caches[DATASETS[0]]['hidden']
    if any(c['hidden'] != h for c in caches.values()):
        raise SystemExit(f"caches with different hidden sizes: {[c['hidden'] for c in caches.values()]}")
    Z = paired_draws(SEEDS, S, h)
    log(f"ruido Z: {tuple(Z.shape)} float64, semillas {SEEDS[0]}..{SEEDS[-1]} (apareadas entre PSA y PSA-1G, ε y datasets)")
    accs, vectors, clipfrac = {}, {}, {}
    for name in DATASETS:
        c, s = caches[name], sel[name]
        accs[name] = Accountant(CALIBRATION, c['n'], DELTA, S)
        vectors[name], clipfrac[name] = build_vectors(c['d32'], s['C'], s['Cg'], accs[name], Z, EPSILONS, SEEDS)
        s_chk = accs[name].sigma_psa(EPSILONS[0])
        check_priv_mean_replication(c['d32'][:, 0, :], C=s['C'], sigma=s_chk)
        check_psa_equals_priv_mean(c['d32'], s['C'], s_chk, Z[0], find_vec(vectors[name], PSA, EPSILONS[0], SEEDS[0])['vec'],
                                   f"[{name}] PSA ε={EPSILONS[0]:g} s={SEEDS[0]} ventana {s['ventana']}")
    checks.append(f"sweep_epsilon.check_priv_mean_replication y PSA == priv_mean de run.py (mismo z) OK en la primera capa "
                  f"de la ventana de cada dataset")
    tests = {}
    for name in DATASETS:
        data, src = load_test(name)
        train_q = {r['question'].strip() for r in caches[name]['train_rows']}
        overlap = sum(r['question'].strip() in train_q for r in data)
        pairs = Counter((r['answer_matching_behavior'], r['answer_not_matching_behavior']) for r in data)
        tests[name] = {'data': data, 'src': src, 'path': test_json_path(name), 'overlap': overlap, 'pairs': pairs}
        msg = (f"[{name}] test {src}: {len(data)} ítems, {overlap} con pregunta igual a uno de los {caches[name]['n']} pares "
               f"del caché")
        if name == OWN_DATASET:
            msg += "  <-- AVISO: IN-SAMPLE por diseño (no hay test separado; son los mismos pares de los vectores)"
        elif overlap:
            raise SystemExit(msg + " -- solapamiento train/test en un dataset público. Stopping.")
        log(msg)
        checks.append(msg)
        odd = {k: v for k, v in pairs.items() if not set(k) <= {'(A)', '(B)'}}
        if odd:
            m2 = (f"[{name}] AVISO: {sum(odd.values())} ítems del test con respuestas fuera de (A)/(B): "
                  f"{ {f'{a} vs {b}': v for (a, b), v in odd.items()} } (la métrica normaliza entre las dos respuestas dadas)")
            log(m2)
            checks.append(m2)
    return {'sel': sel, 'caches': caches, 'h': h, 'Z': Z, 'accs': accs, 'vectors': vectors, 'clipfrac': clipfrac,
            'tests': tests, 'layers': {name: sel[name]['layers'] for name in DATASETS}}


# --------------------------------------------------------------------------------------
# config.json
# --------------------------------------------------------------------------------------
def library_versions() -> dict:
    import scipy
    import steering_vectors
    import transformers
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'transformers': str(transformers.__version__), 'steering_vectors': str(steering_vectors.__version__),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}


def build_config(tag: str, model: str, paths: dict, state: dict) -> dict:
    per = {}
    for name in DATASETS:
        s, c, t = state['sel'][name], state['caches'][name], state['tests'][name]
        per[name] = {'ventana': s['ventana'], 'capas': s['layers'], 'C_eval': s['C'], 'Cg_eval': s['Cg'],
                     'recorte_total': s['recorte_total'], 'n': c['n'],
                     'cache': {'path': Path(c['path']).as_posix(), 'sha256': sha256_file(c['path'])},
                     'test': {'path': Path(t['path']).as_posix(), 'sha256': sha256_file(t['path']),
                              'n_items': len(t['data']), 'in_sample': name == OWN_DATASET}}
    cfg = {'script': 'eval_level3.py', 'model_tag': tag, 'model': model, 'dtype': 'float32',
           'datasets': DATASETS, 'publicos': PUBLIC, 'in_sample': [OWN_DATASET],
           'epsilons': EPSILONS, 'semillas': SEEDS, 'lambdas': LAMBDAS, 'delta': DELTA, 'calibracion': CALIBRATION,
           'S': S, 'n_samples': N_SAMPLES, 'shuffle_seed': SHUFFLE_SEED, 'read_token_index': READ_TOKEN_INDEX,
           'min_token_index': 0, 'evaluaciones_por_dataset': N_EVALS,
           'c_selection': {'path': Path(paths['c_selection']).as_posix(), 'sha256': sha256_file(Path(paths['c_selection']))},
           'rbar_csv': Path(paths['rbar_csv']).as_posix(), 'diffs_dir': Path(paths['diffs_dir']).as_posix(),
           'por_dataset': per, 'versiones': library_versions()}
    return json.loads(json.dumps(cfg))                 # canonical (what json.load gives back)


def dict_diff(a, b, prefix: str = '') -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(f"{p}: {'falta en config.json' if k not in a else 'no está en la configuración actual'}")
            else:
                out += dict_diff(a[k], b[k], p)
        return out
    return [] if a == b else [f"{prefix}: guardado {a!r} != actual {b!r}"]


def check_or_write_config(out_dir: Path, cfg: dict) -> bool:
    """True if resuming (out_dir exists and its config.json equals cfg), False if out_dir was created now."""
    p = out_dir / 'config.json'
    if out_dir.exists():
        if not p.is_file():
            raise SystemExit(f"{out_dir} exists but has no config.json; cannot tell what it contains. Stopping.")
        saved = json.loads(p.read_text(encoding='utf-8'))
        diff = dict_diff(saved, cfg)
        if diff:
            raise SystemExit(f"{p} does not match the current configuration; refusing to mix results:\n  "
                             + "\n  ".join(diff[:30]))
        log(f"{p} coincide exactamente con la configuración actual: se reanuda")
        return True
    out_dir.mkdir(parents=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    log(f"escrito {p}")
    return False


# --------------------------------------------------------------------------------------
# Evaluations
# --------------------------------------------------------------------------------------
def eval_plan(vectors: list[dict]) -> list[dict]:
    """The 127 evaluations of one dataset, in order: sin steering, then every vector of build_vectors x every lambda."""
    nan = float('nan')
    tasks = [{'condicion': NONE, 'epsilon': nan, 'semilla': nan, 'lambda': 0.0, 'vec': None, 'v': None}]
    for v in vectors:
        for lam in LAMBDAS:
            tasks.append({'condicion': v['condicion'], 'epsilon': v['epsilon'], 'semilla': v['semilla'], 'lambda': lam,
                          'vec': v['vec'], 'v': v})
    assert len(tasks) == N_EVALS, len(tasks)
    return tasks


def expected_keys(state: dict) -> dict[tuple, str]:
    out = {}
    for name in DATASETS:
        for t in eval_plan(state['vectors'][name]):
            out[task_key(name, t['condicion'], t['epsilon'], t['semilla'], t['lambda'])] = state['sel'][name]['ventana']
    assert len(out) == len(DATASETS) * N_EVALS
    return out


def sv_for(state: dict, name: str, vec, dtype, dev):
    """Steering vector of `vec` (S, h) on the window of `name`, block j on layer W_name[j]."""
    return None if vec is None else to_steering_vector(vec, state['layers'][name], dtype, dev)


def task_label(t: dict) -> str:
    if t['condicion'] in (PSA, G1):
        return f"{t['condicion']} ε={t['epsilon']:g} s={int(t['semilla'])}"
    return t['condicion']


def ab_loop(state: dict, store: ResultsStore, eval_fn, verbose: bool = True) -> int:
    """eval_fn(name, vec or None, lam) -> ab. Skips keys already in the store; appends after every evaluation."""
    n_new = 0
    for name in DATASETS:
        s = state['sel'][name]
        tasks = eval_plan(state['vectors'][name])
        todo = [t for t in tasks if task_key(name, t['condicion'], t['epsilon'], t['semilla'], t['lambda']) not in store]
        log("=" * 100)
        log(f"A/B — {name} | ventana {s['ventana']} | C {s['C']:g} | {len(state['tests'][name]['data'])} ítems | "
            f"{N_EVALS - len(todo)} de {N_EVALS} ya en resultados.csv, {len(todo)} por calcular"
            + ("  [IN-SAMPLE]" if name == OWN_DATASET else ""))
        t0 = time.time()
        for i, t in enumerate(todo):
            t1 = time.time()
            r = eval_fn(name, t['vec'], t['lambda'])
            store.append({'dataset': name, 'ventana': s['ventana'], 'condicion': t['condicion'], 'epsilon': t['epsilon'],
                          'semilla': t['semilla'], 'lambda': t['lambda'], 'ab': r})
            n_new += 1
            if verbose:
                log(f"  [{i + 1:3d}/{len(todo)}] {task_label(t):26s} λ={t['lambda']:+g}  ab={r:.4f}  ({time.time() - t1:.1f} s)")
        if todo:
            log(f"  [{name}] {len(todo)} evaluaciones en {time.time() - t0:.0f} s")
    return n_new


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def by_seed(sub: pd.DataFrame, cond: str, lam: float, eps: float) -> np.ndarray:
    r = sub[(sub['condicion'] == cond) & (sub['lambda'] == lam) & (sub['epsilon'] == eps)].sort_values('semilla')
    if r['semilla'].astype(int).tolist() != SEEDS:
        raise SystemExit(f"{cond} λ={lam:g} ε={eps:g}: seeds {r['semilla'].tolist()}, expected {SEEDS}")
    return r['ab'].to_numpy(dtype=float)


def single(sub: pd.DataFrame, cond: str, lam: float) -> float:
    r = sub[(sub['condicion'] == cond) & (sub['lambda'] == lam)]
    if len(r) != 1:
        raise SystemExit(f"{cond} λ={lam:g}: {len(r)} rows, expected 1")
    return float(r['ab'].iloc[0])


def paired_stats(p: np.ndarray, g: np.ndarray) -> dict:
    """PSA values p and PSA-1G values g of the same seeds: means, t 95% intervals and the paired difference g - p."""
    mp, _, hp = ci95(p)
    mg, _, hg = ci95(g)
    md, _, hd = ci95(g - p)
    return {'psa': mp, 'psa_lo': mp - hp, 'psa_hi': mp + hp, 'g1': mg, 'g1_lo': mg - hg, 'g1_hi': mg + hg,
            'diff': md, 'diff_lo': md - hd, 'diff_hi': md + hd, 'k': int((g > p).sum()), 'K': len(p)}


def dataset_stats(res: pd.DataFrame, name: str, lam: float, eps: float) -> dict:
    sub = res[res['dataset'] == name]
    return paired_stats(by_seed(sub, PSA, lam, eps), by_seed(sub, G1, lam, eps))


def global_stats(res: pd.DataFrame, names: list[str], lam: float, eps: float) -> dict:
    """Means over `names` of the per-dataset seed means; PSA-1G - PSA paired by seed: D_s = mean over datasets of
    (PSA-1G - PSA)_{d,s} (the same z for every dataset), t 95% interval over the seeds, and #seeds with D_s > 0."""
    P = np.stack([by_seed(res[res['dataset'] == d], PSA, lam, eps) for d in names])        # (datasets, seeds)
    G = np.stack([by_seed(res[res['dataset'] == d], G1, lam, eps) for d in names])
    D = (G - P).mean(axis=0)
    md, _, hd = ci95(D)
    return {'floor': float(np.mean([single(res[res['dataset'] == d], NONE, 0.0) for d in names])),
            'psa0': float(np.mean([single(res[res['dataset'] == d], PSA0, lam) for d in names])),
            'psa': float(P.mean(axis=1).mean()), 'g1': float(G.mean(axis=1).mean()),
            'diff': md, 'diff_lo': md - hd, 'diff_hi': md + hd, 'k': int((D > 0).sum()), 'K': len(D)}


# --------------------------------------------------------------------------------------
# Pilot comparison (--pilot-csv, 1B only)
# --------------------------------------------------------------------------------------
def compare_with_pilot(res: pd.DataFrame, pilot: pd.DataFrame, tol: float = PILOT_TOL) -> tuple[list[str], int]:
    """Rows of the pilot with lambda = +2 (and seeds 0, 1 for PSA / PSA-1G) on hallucination, refusal and tofu-forget10,
    plus the 'sin steering' row, against the same keys here. Returns (lines, number of rows that differ by > tol or are
    missing)."""
    ours = {task_key(r['dataset'], r['condicion'], r['epsilon'], r['semilla'], r['lambda']): float(r['ab'])
            for _, r in res.iterrows()}
    pr = pilot[pilot['dataset'].isin(PILOT_DATASETS)
               & ((pilot['lambda'] == PILOT_LAMBDA) | (pilot['condicion'] == NONE))]
    pr = pr[pr['semilla'].isna() | pr['semilla'].isin(PILOT_SEEDS)]
    lines, bad, worst = [], 0, 0.0
    for _, r in pr.iterrows():
        k = task_key(r['dataset'], r['condicion'], r['epsilon'], r['semilla'], r['lambda'])
        label = f"{r['dataset']:14s} {r['condicion']:22s} ε={fmt_float(r['epsilon']) or '-':5s} s={fmt_seed(r['semilla']) or '-':2s} " \
                f"λ={r['lambda']:+g}"
        if k not in ours:
            bad += 1
            lines.append(f"  AVISO: {label}: no está en resultados.csv")
            continue
        d = abs(ours[k] - float(r['ab']))
        worst = max(worst, d)
        if d > tol:
            bad += 1
            lines.append(f"  AVISO: {label}: piloto {float(r['ab']):.8f} vs nivel 3 {ours[k]:.8f} (|diff| {d:.2e} > {tol:g})")
    expected = len(PILOT_DATASETS) * (1 + 3 + 2 * len(EPSILONS) * len(PILOT_SEEDS))
    head = (f"  {len(pr)} filas del piloto comparadas (esperadas {expected}: por dataset sin steering + λ=+2 de Mean "
            f"Steering, PSA y PSA-1G sin ruido y PSA / PSA-1G con ε {EPSILONS} y semillas {PILOT_SEEDS}); "
            f"max |diff| = {worst:.2e}; {bad} con |diff| > {tol:g} o ausentes")
    if len(pr) != expected:
        lines.insert(0, f"  AVISO: el csv del piloto tiene {len(pr)} filas comparables, se esperaban {expected}")
        bad += abs(expected - len(pr))
    return [head] + lines, bad


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def sigma_lines(P: Report, acc: Accountant, h: int, epsilons: list[float]) -> None:
    """eval_pilot.sigma_lines with the epsilons as a parameter."""
    P(f"  n={acc.n}, δ={acc.delta:g}, S={acc.S}, calibración {acc.calibration} (Balle-Wang), sensibilidad 2/n = {acc.sens:.5f}")
    P(f"  {'ε':>4s} | {'σ PSA (por coord.)':>18s} {'‖ruido‖/capa':>12s} | {'σ PSA-1G':>9s} {'√5·σ PSA-1G (por coord.)':>24s} "
      f"{'‖ruido‖/capa':>12s} | {'σ_PSA/(√5σ_1G)':>14s}")
    for eps in epsilons:
        sp, sg = acc.sigma_psa(eps), acc.sigma_prima(eps)      # sigma_prima = PSA-1G
        P(f"  {eps:4g} | {sp:18.6f} {sp * math.sqrt(h):12.3f} | {sg:9.6f} {math.sqrt(S) * sg:24.6f} "
          f"{math.sqrt(S) * sg * math.sqrt(h):12.3f} | {sp / (math.sqrt(S) * sg):14.3f}")
    P("  (‖ruido‖/capa ≈ σ_coord·√h, esperado por unidad de λ; PSA-1G: σ calibrado sobre el concatenado, aplicado ×√5)")


def vector_lines(P: Report, vecs: list[dict], layers: list[int]) -> None:
    """Like eval_pilot.vector_lines, but the noisy vectors are averaged over the seeds (one row per mechanism and eps)."""
    ref = find_vec(vecs, MEAN)['vec']
    P(f"  {'condición':30s} " + " ".join(f"{'L' + str(l) + ' ‖v‖':>9s} {'cos':>6s}" for l in layers))

    def row(label, group):
        nrm = torch.stack([v['vec'].norm(dim=1) for v in group]).mean(dim=0)
        cs = torch.stack([cosine(v['vec'], ref) for v in group]).mean(dim=0)
        P(f"  {label:30s} " + " ".join(f"{nrm[j].item():9.4f} {cs[j].item():6.3f}" for j in range(len(layers))))

    for cond in (MEAN, PSA0, G1_0):
        row(cond, [find_vec(vecs, cond)])
    for cond in (PSA, G1):
        for eps in EPSILONS:
            group = [v for v in vecs if v['condicion'] == cond and v['epsilon'] == eps]
            row(f"{cond} ε={eps:g} (media {len(group)} sem.)", group)
    P("  (cos = coseno con Mean Steering (paper) en la misma capa; ‖v‖ antes de λ; diagnóstico, no es una métrica)")


def fmt_ci(m: float, lo: float, hi: float) -> str:
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_dci(m: float, lo: float, hi: float) -> str:
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def write_resumen(path: Path | None, model: str, state: dict, res: pd.DataFrame, checks: list[str], c4_lines: list[str],
                  pilot_lines: list[str] | None, elapsed: float | None) -> str:
    for name in DATASETS:
        n_rows = int((res['dataset'] == name).sum())
        if n_rows != N_EVALS:
            raise SystemExit(f"resultados.csv has {n_rows} rows for {name}, expected {N_EVALS}; resumen not written")
    P = Report()
    P("=" * 110)
    P(f"RESUMEN — nivel 3 — {model}")
    P("=" * 110)
    P(f"datasets {DATASETS} | ε {EPSILONS} | semillas {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)}) | λ {LAMBDAS} | δ {DELTA:g} | "
      f"steering en todos los tokens | {N_EVALS} evaluaciones A/B por dataset")
    P("PSA: d/max(C,‖d‖) por capa, media + gaussiano por capa a (ε/5, δ/5), sin ×C. "
      "PSA-1G: D/max(Cg,‖D‖) sobre las 5 capas concatenadas, media + UN gaussiano a (ε, δ), ×√5.")
    P("Mean Steering (paper) = scaled_mean de run.py. Métrica A/B = utils.evaluate_model. Ventana, C y Cg propios de cada "
      "dataset (c_selection.csv).")
    P("IC 95%: t de Student con las 10 semillas (9 g.l.). PSA-1G − PSA: diferencia apareada por semilla (mismo z).")
    P(f"AVISO: {OWN_DATASET} es IN-SAMPLE (se evalúa sobre los mismos 400 pares de los que salen los vectores).")
    small = {d: state['caches'][d]['n'] for d in DATASETS if state['caches'][d]['n'] < N_SAMPLES}
    if small:
        P(f"AVISO: datasets con n < {N_SAMPLES}: {small}; su σ usa su n real (más ruido al mismo ε).")
    no_total = [d for d in DATASETS if not state['sel'][d]['recorte_total']]
    if no_total:
        P(f"AVISO: recorte_total = False en c_selection.csv para {no_total}.")
    P("")
    P("CHEQUEOS:")
    for x in checks:
        P(f"  - {x.strip()}")
    P(f"  chequeo 4, por capa ({CHECK4_DATASET}, ventana {state['sel'][CHECK4_DATASET]['ventana']}):")
    for x in c4_lines:
        P(f"  {x}")
    if pilot_lines is not None:
        P("")
        P(f"COMPARACIÓN CON EL PILOTO (tolerancia {PILOT_TOL:g}):")
        for x in pilot_lines:
            P(x)
    for name in DATASETS:
        s, t, c = state['sel'][name], state['tests'][name], state['caches'][name]
        sub = res[res['dataset'] == name]
        P("")
        P("=" * 110)
        P(f"{name}{'  [IN-SAMPLE]' if name == OWN_DATASET else ''} | ventana {s['ventana']} | C={s['C']:g} Cg={s['Cg']:.4f} | "
          f"n={c['n']} | test: {t['src']} ({len(t['data'])} ítems, solapamiento con el caché {t['overlap']})")
        P(f"  c_selection: C={s['C_sel']} empate={s['empate']} {s['C_empatados']} recorte_total={s['recorte_total']}"
          + ("  <-- AVISO: recorte_total = False" if not s['recorte_total'] else ""))
        if name == OWN_DATASET:
            P("  AVISO: test in-sample (los mismos pares de los que salen los vectores)")
        floor = single(sub, NONE, 0.0)
        for lam in LAMBDAS:
            P(f"  λ = {lam:+g}")
            P(f"    {NONE + ' (λ=0, piso)':30s} {floor:.4f}")
            for cond in (MEAN, PSA0, G1_0):
                P(f"    {cond:30s} {single(sub, cond, lam):.4f}")
            P(f"    {'ε':>4s} | {'PSA media [IC 95%]':>26s} | {'PSA-1G media [IC 95%]':>26s} | "
              f"{'PSA-1G − PSA [IC 95%]':>29s} | PSA-1G > PSA")
            for eps in EPSILONS:
                st = dataset_stats(res, name, lam, eps)
                P(f"    {eps:4g} | {fmt_ci(st['psa'], st['psa_lo'], st['psa_hi']):>26s} | "
                  f"{fmt_ci(st['g1'], st['g1_lo'], st['g1_hi']):>26s} | {fmt_dci(st['diff'], st['diff_lo'], st['diff_hi']):>29s} | "
                  f"{st['k']}/{st['K']} semillas")
        P("  σ por coordenada:")
        sigma_lines(P, state['accs'][name], state['h'], EPSILONS)
        fp, fg = state['clipfrac'][name]['frac_psa'], state['clipfrac'][name]['frac_g1']
        P(f"  Recorte: PSA (C={s['C']:g}) fracción de filas recortadas por capa "
          + " ".join(f"L{l} {fp[j]:.3f}" for j, l in enumerate(s['layers']))
          + f" | PSA-1G (Cg={s['Cg']:.3f}) fracción de filas concatenadas recortadas {fg:.3f}")
        P("  ‖v_l‖ por capa y coseno con Mean Steering:")
        vector_lines(P, state['vectors'][name], s['layers'])
    P("")
    P("=" * 110)
    P(f"TABLA GLOBAL — media sobre los {len(PUBLIC)} datasets públicos (sin {OWN_DATASET}) de las medias por dataset")
    P("PSA-1G − PSA: D_s = media sobre datasets de (PSA-1G − PSA) con la semilla s (mismo z en todos los datasets), "
      "IC 95% t sobre las 10 semillas; '>0' = semillas con D_s > 0.")
    if small:
        P(f"  (incluye {sorted(small)} con n < {N_SAMPLES}: más ruido al mismo ε)")
    for lam in LAMBDAS:
        P(f"  λ = {lam:+g}")
        P(f"    {'':26s} {'ε':>4s} | {'piso':>7s} {'PSA s/r':>7s} {'PSA':>7s} {'PSA-1G':>7s} | {'PSA-1G − PSA [IC 95%]':>29s} | >0")
        for label, names in ((f"7 públicos", PUBLIC), (f"{OWN_DATASET} (IN-SAMPLE)", [OWN_DATASET])):
            for eps in EPSILONS:
                g = global_stats(res, names, lam, eps)
                P(f"    {label:26s} {eps:4g} | {g['floor']:7.4f} {g['psa0']:7.4f} {g['psa']:7.4f} {g['g1']:7.4f} | "
                  f"{fmt_dci(g['diff'], g['diff_lo'], g['diff_hi']):>29s} | {g['k']}/{g['K']}")
    P("  (piso = sin steering, λ=0; PSA s/r = PSA sin ruido; la fila de tofu NO entra en la media de los públicos)")
    if elapsed is not None:
        P("")
        P(f"tiempo de esta ejecución {elapsed:.0f} s")
    text = P.text()
    if path is not None:
        path.write_text(text, encoding='utf-8')
    return text


# --------------------------------------------------------------------------------------
# Real run
# --------------------------------------------------------------------------------------
def check_chat_date_all(test_pairs: list[tuple[str, str]], tokenizer, name: str) -> None:
    """utils.check_today_date on every test prompt, silently (diagnose_read_token_index prints it for the first pair)."""
    from utils import CHAT_DATE, TODAY_DATE_PREFIX, is_llama3_tokenizer
    if not is_llama3_tokenizer(tokenizer):
        raise SystemExit(f"tokenizer {tokenizer.name_or_path!r} is not a Llama-3 tokenizer")
    want = [TODAY_DATE_PREFIX + CHAT_DATE]
    for i, pair in enumerate(test_pairs):
        for prompt in pair:
            if [l for l in prompt.split('\n') if l.startswith(TODAY_DATE_PREFIX)] != want:
                raise SystemExit(f"[{name}] test prompt {i} does not carry exactly {want}. Stopping.")


def run(args) -> None:
    t_start = time.time()
    model_name = MODELS[args.model_tag]
    paths = {'diffs_dir': args.diffs_dir, 'c_selection': args.c_selection, 'rbar_csv': args.rbar_csv}
    log("=" * 100)
    log(f"NIVEL 3 — {model_name} | datasets {DATASETS} | ε {EPSILONS} | semillas {SEEDS[0]}..{SEEDS[-1]} | λ {LAMBDAS} | "
        f"δ {DELTA:g} | {N_EVALS} evaluaciones por dataset")
    log(f"diffs {args.diffs_dir} | c_selection {args.c_selection} | rbar {args.rbar_csv} | out {args.out_dir}")
    log(f"AVISO: {OWN_DATASET} se evalúa IN-SAMPLE (sus 400 pares de generate, los mismos de los vectores)")
    checks: list[str] = []

    # ---------------- preflight, no model ----------------
    state = preflight(paths, model_name, checks)
    cfg = build_config(args.model_tag, model_name, paths, state)
    out_dir = Path(args.out_dir)
    resumed = check_or_write_config(out_dir, cfg)
    store = ResultsStore(out_dir / 'resultados.csv', expected_keys(state))
    log(f"resultados.csv: {len(store)} de {len(DATASETS) * N_EVALS} evaluaciones ya calculadas"
        + (" (reanudación)" if resumed else ""))

    # ---------------- model ----------------
    from utils import diagnose_read_token_index, get_model_and_tokenizer, make_dataset, make_pos_neg_pair
    model, tokenizer = get_model_and_tokenizer(model_name, load_bfloat=False)   # float32, as run.py
    model.eval()
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    if dtype != torch.float32:
        raise SystemExit(f"model loaded as {dtype}, expected float32")
    top = max(max(l) for l in state['layers'].values())
    if model.config.hidden_size != state['h'] or model.config.num_hidden_layers <= top:
        raise SystemExit(f"model hidden {model.config.hidden_size} / {model.config.num_hidden_layers} layers incompatible "
                         f"with the caches (hidden {state['h']}, highest window layer {top})")
    base_hooks = n_forward_hooks(model)
    log(f"modelo en {dev}, dtype {dtype}, {model.config.num_hidden_layers} capas, {base_hooks} forward hooks de base")
    test_pairs = {}
    for name in DATASETS:
        c = state['caches'][name]
        pos0, neg0 = make_pos_neg_pair(c['train_rows'][0], tokenizer)
        if (pos0, neg0) != (c['example_pos'], c['example_neg']):
            raise SystemExit(f"[{name}] the first training prompt rebuilt now differs from the one stored in the cache: "
                             f"the prompt format changed since the extraction. Stopping.")
        data = state['tests'][name]['data']
        test_pairs[name] = make_dataset(data, tokenizer)
        log(f"[{name}] TEST POS: {test_pairs[name][0][0]!r}")
        log(f"[{name}] TEST NEG: {test_pairs[name][0][1]!r}")
        diagnose_read_token_index(test_pairs[name][0][0], tokenizer, READ_TOKEN_INDEX,
                                  expected_letter=data[0]['answer_matching_behavior'][1], label=f'[{name} test pos] ')
        diagnose_read_token_index(test_pairs[name][0][1], tokenizer, READ_TOKEN_INDEX,
                                  expected_letter=data[0]['answer_not_matching_behavior'][1], label=f'[{name} test neg] ')
        check_chat_date_all(test_pairs[name], tokenizer, name)
    checks.append(f"primer prompt de entrenamiento reconstruido == example_pos/neg del caché en los {len(DATASETS)} "
                  f"datasets; CHAT_DATE verificado en todos los prompts de test")

    # ---------------- check 4 ----------------
    log("=" * 100)
    c4 = state['caches'][CHECK4_DATASET]
    c4_layers = state['layers'][CHECK4_DATASET]
    log(f"CHEQUEO 4 — Mean Steering: caché vs train_steering_vector + scaled_mean (run.py:37-41), {CHECK4_DATASET}, "
        f"ventana {c4_layers}, mismos {c4['n']} pares")
    import steering_vectors
    import transformers
    from steering_vectors import train_steering_vector
    log(f"  versiones del caché: {c4['versions']}")
    log(f"  versiones actuales: steering_vectors {steering_vectors.__version__}, transformers {transformers.__version__}, "
        f"torch {torch.__version__}")
    sv_lib = train_steering_vector(model, tokenizer, make_dataset(c4['train_rows'], tokenizer),
                                   read_token_index=READ_TOKEN_INDEX, show_progress=True,
                                   aggregator=scaled_mean_as_in_run_py, layers=c4_layers)
    if sorted(sv_lib.layer_activations) != c4_layers:
        raise SystemExit(f"train_steering_vector returned layers {sorted(sv_lib.layer_activations)}, expected {c4_layers}")
    lib = torch.stack([sv_lib.layer_activations[l].detach().cpu().double() for l in c4_layers])
    c4_lines = compare_mean_vectors(lib, find_vec(state['vectors'][CHECK4_DATASET], MEAN)['vec'], c4_layers, CHECK4_TOL)
    log(f"  assert OK (tol {CHECK4_TOL:g})")
    checks.append(f"chequeo 4 OK en {CHECK4_DATASET} (ventana {state['sel'][CHECK4_DATASET]['ventana']}): max |diff| por "
                  f"capa ≤ {CHECK4_TOL:g}")
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit("train_steering_vector left hooks on the model. Stopping.")
    del sv_lib

    # ---------------- A/B ----------------
    def eval_fn(name, vec, lam):
        return ab_eval(model, tokenizer, test_pairs[name], sv_for(state, name, vec, dtype, dev), lam, base_hooks)

    n_new = ab_loop(state, store, eval_fn)
    checks.append(f"ningún hook de steering quedó en el modelo después de cada evaluación ({n_new} en esta ejecución)")
    res = read_results(store.path)

    # ---------------- pilot comparison ----------------
    pilot_lines = None
    if args.pilot_csv:
        log("=" * 100)
        log(f"COMPARACIÓN CON EL PILOTO — {args.pilot_csv}")
        pilot_lines, bad = compare_with_pilot(res, pd.read_csv(args.pilot_csv))
        for x in pilot_lines:
            log(x)
        if bad:
            log(f"  <-- AVISO: {bad} fila(s) del piloto difieren en más de {PILOT_TOL:g} o faltan")

    # ---------------- summary ----------------
    write_resumen(out_dir / 'resumen.txt', model_name, state, res, checks, c4_lines, pilot_lines, time.time() - t_start)
    log(f"\nresultados.csv ({len(res)} filas), resumen.txt y config.json en {out_dir} | {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------------------
# Self-test (CPU, synthetic, nothing downloaded; files only in a system temp dir)
# --------------------------------------------------------------------------------------
ST_MODEL = 'synthetic/level3-selftest'
ST_H, ST_L = 64, 20
# 8B-like windows (12-16, hallucination / refusal 13-17) and a C_eval that is not 2 on refusal: nothing may be assumed
ST_WINDOWS = {name: ('13-17' if name in ('hallucination', 'refusal') else '12-16') for name in DATASETS}
ST_C = {name: (3.0 if name == 'refusal' else 2.0) for name in DATASETS}
ST_RECORTE = {name: name != 'myopic-reward' for name in DATASETS}


def st_sel_rows(windows: dict, C: dict, recorte: dict, model: str = ST_MODEL) -> list[dict]:
    return [{'model': model, 'dataset': n, 'ventana': windows[n], 'C': float('nan'), 'Cg': float('nan'), 'empate': True,
             'C_empatados': f"{C[n]:g} 3", 'C_eval': C[n], 'Cg_eval': C[n] * math.sqrt(S), 'recorte_total': recorte[n]}
            for n in DATASETS]


def st_csv(rows: list[dict]) -> io.StringIO:
    return io.StringIO(pd.DataFrame(rows, columns=SEL_COLUMNS).to_csv(index=False))


def st_write_inputs(root: Path) -> dict:
    """Synthetic sweep_epsilon-style caches for the 8 real dataset jsons (layer l of every row points along e_l, so the
    layer a vector was built from can be read off its argmax), the matching R̄ csv and a c_selection.csv."""
    from sweep_epsilon import rbar_from_cache
    (root / 'diffs').mkdir(parents=True)
    rb = []
    for k, name in enumerate(DATASETS):
        data = load_dataset_json(name)
        total = len(data)
        n = min(N_SAMPLES, total)
        perm = shuffle_permutation(total, n, SHUFFLE_SEED)
        first = data[perm[0]]
        rng = np.random.default_rng([k, 3])
        diffs = {}
        for l in range(ST_L):
            d = np.zeros((n, ST_H))
            d[:, l] = 1.0
            d += 0.3 * rng.standard_normal((n, ST_H)) / math.sqrt(ST_H)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            d *= (4.0 + 0.2 * l) * np.exp(0.3 * rng.standard_normal((n, 1)))
            diffs[l] = torch.from_numpy(d.astype(np.float32))
        cache = {'model': ST_MODEL, 'dataset': name, 'n': n, 'total': total, 'seed': SHUFFLE_SEED,
                 'read_token_index': READ_TOKEN_INDEX, 'hidden': ST_H, 'layers': list(range(ST_L)), 'diffs': diffs,
                 'example_pos': f"<sys>{first['question'].strip()}<asst>{first['answer_matching_behavior']}",
                 'example_neg': f"<sys>{first['question'].strip()}<asst>{first['answer_not_matching_behavior']}"}
        torch.save(cache, root / 'diffs' / f"{name}.pt")
        rb.append(rbar_from_cache(cache).assign(model=ST_MODEL))
    pd.concat(rb).to_csv(root / 'rbar.csv', index=False)
    pd.DataFrame(st_sel_rows(ST_WINDOWS, ST_C, ST_RECORTE), columns=SEL_COLUMNS).to_csv(root / 'c_selection.csv',
                                                                                         index=False)
    return {'diffs_dir': str(root / 'diffs'), 'c_selection': str(root / 'c_selection.csv'), 'rbar_csv': str(root / 'rbar.csv')}


def st_fake_eval(state: dict, limit: int | None = None):
    """Deterministic stand-in for the model: a function of the steering vector actually built for the dataset's window.
    Raises Interrupted after `limit` calls (a job killed mid-run)."""
    W = {name: torch.from_numpy(np.random.default_rng([k, 9]).standard_normal((S, ST_H))) for k, name in enumerate(DATASETS)}
    calls = [0]

    def f(name, vec, lam):
        if limit is not None and calls[0] >= limit:
            raise Interrupted
        calls[0] += 1
        if vec is None:
            return 0.5 + 0.01 * DATASETS.index(name)
        sv = sv_for(state, name, vec, torch.float64, 'cpu')
        assert sorted(sv.layer_activations) == state['layers'][name], (name, sorted(sv.layer_activations))
        x = sum((sv.layer_activations[l] * W[name][j]).sum().item() for j, l in enumerate(state['layers'][name]))
        return 0.5 + 0.45 * math.tanh(lam * x)

    f.calls = calls
    return f


def expect_exit(fn, label: str) -> None:
    try:
        fn()
    except SystemExit as e:
        log(f"  (esperado) {label}: {str(e).splitlines()[0][:110]}")
    else:
        raise AssertionError(f"accepted: {label}")


def selftest(args) -> None:
    t0 = time.time()
    sq = math.sqrt(S)
    log("#" * 100)
    log("SELFTEST 0 — eval_pilot.selftest(): las funciones de vectores y de steering reutilizadas")
    eval_pilot.selftest(None)

    log("#" * 100)
    log("SELFTEST A — lectura de c_selection.csv con ventanas distintas por dataset (como el 3B y el 8B)")
    w3 = {n: ('10-14' if n == 'sycophancy' else '11-15') for n in DATASETS}
    w8 = ST_WINDOWS
    C2 = {n: 2.0 for n in DATASETS}
    for label, w, C in (('3B', w3, C2), ('8B', w8, ST_C)):
        out = read_c_selection(st_csv(st_sel_rows(w, C, ST_RECORTE)), ST_MODEL, DATASETS)
        for n in DATASETS:
            a = int(w[n].split('-')[0])
            assert out[n]['layers'] == list(range(a, a + S)) and out[n]['ventana'] == w[n], (n, out[n])
            assert out[n]['C'] == C[n] and abs(out[n]['Cg'] - C[n] * sq) < 1e-12
            assert out[n]['recorte_total'] == ST_RECORTE[n]
        log(f"  {label}: " + ", ".join(f"{n} {out[n]['ventana']}" for n in DATASETS))
    log(f"  C_eval distinto de 2 aceptado (refusal C={ST_C['refusal']:g}); recorte_total=False aceptado (myopic-reward)")
    rows = st_sel_rows(w8, ST_C, ST_RECORTE)
    bad = {'4 capas': {'ventana': '12-15'}, '6 capas': {'ventana': '12-17'}, 'ventana invertida': {'ventana': '16-12'},
           'ventana no numérica': {'ventana': '12-16b'}, 'Cg_eval ≠ C_eval·√5': {'Cg_eval': 2.0 * sq + 1e-3},
           'recorte_total vacío': {'recorte_total': ''}, 'C_eval vacío': {'C_eval': '', 'Cg_eval': ''}}
    for label, patch in bad.items():
        expect_exit(lambda: read_c_selection(st_csv([{**r, **(patch if r['dataset'] == 'refusal' else {})} for r in rows]),
                                             ST_MODEL, DATASETS), label)
    expect_exit(lambda: read_c_selection(io.StringIO(pd.DataFrame(rows).drop(columns=['recorte_total']).to_csv(index=False)),
                                         ST_MODEL, DATASETS), 'sin columna recorte_total')
    expect_exit(lambda: read_c_selection(st_csv(rows + rows[:1]), ST_MODEL, DATASETS), 'fila duplicada')
    expect_exit(lambda: read_c_selection(st_csv(rows), MODELS['8b'], DATASETS), 'otro modelo')
    log("  OK")

    tmp = Path(tempfile.mkdtemp(prefix='eval_level3_selftest_'))
    try:
        log("#" * 100)
        log(f"SELFTEST B — preflight sobre caches sintéticos ({tmp}): cada dataset usa su propia ventana y su propio C")
        paths = st_write_inputs(tmp / 'inputs')
        checks: list[str] = []
        state = preflight(paths, ST_MODEL, checks)
        for name in DATASETS:
            layers = state['layers'][name]
            assert layers == parse_window(ST_WINDOWS[name], name)
            raw = torch.load(Path(paths['diffs_dir']) / f"{name}.pt", weights_only=True)
            d = torch.stack([raw['diffs'][l] for l in layers], dim=1).double()     # independent slicing of the window
            for cond in (MEAN, PSA0, G1_0):
                v = find_vec(state['vectors'][name], cond)['vec']
                am = v.abs().argmax(dim=1).tolist()
                assert am == layers, (name, cond, am, layers)
            assert torch.equal(find_vec(state['vectors'][name], PSA0)['vec'], psa_rows(d, ST_C[name])[0].mean(dim=0))
            assert torch.equal(find_vec(state['vectors'][name], G1_0)['vec'],
                               sq * g1_rows(d, ST_C[name] * sq)[0].mean(dim=0))
            assert state['accs'][name].n == raw['n']
        log("  OK: argmax de cada fila de Mean Steering / PSA sin ruido / PSA-1G sin ruido = capa de la ventana propia "
            f"({', '.join(f'{n} {ST_WINDOWS[n]}' for n in DATASETS)}); PSA / PSA-1G sin ruido == recomputación "
            f"independiente con el C propio (refusal C=3); σ con el n real de cada dataset")

        log("#" * 100)
        log(f"SELFTEST C — {N_EVALS} evaluaciones por dataset; reanudación; aborto si config.json no coincide")
        assert all(len(eval_plan(state['vectors'][n])) == N_EVALS for n in DATASETS)
        assert N_EVALS == 1 + 3 * 2 + 2 * 3 * 10 * 2 == 127
        cfg = build_config('selftest', ST_MODEL, paths, state)
        keys = expected_keys(state)
        full = tmp / 'full'
        assert not check_or_write_config(full, cfg)
        fe = st_fake_eval(state)
        store = ResultsStore(full / 'resultados.csv', keys)
        assert ab_loop(state, store, fe, verbose=False) == len(DATASETS) * N_EVALS == fe.calls[0]
        text_full = (full / 'resultados.csv').read_bytes()
        res_full = read_results(full / 'resultados.csv')
        counts = res_full.groupby('dataset').size()
        assert all(counts[n] == N_EVALS for n in DATASETS), counts
        assert all((res_full[res_full['dataset'] == n]['ventana'] == ST_WINDOWS[n]).all() for n in DATASETS)
        log(f"  corrida completa: {len(res_full)} filas, {N_EVALS} por dataset, cada una con la ventana de su dataset")
        part = tmp / 'part'
        cuts = [300, 1, 400]          # mid-dataset (3rd), then after one more row, then mid-dataset again
        done = 0
        snapshots = []
        for cut in cuts:
            check_or_write_config(part, cfg)
            st_ = ResultsStore(part / 'resultados.csv', keys)
            assert len(st_) == done
            try:
                ab_loop(state, st_, st_fake_eval(state, limit=cut), verbose=False)
            except Interrupted:
                done += cut
                snapshots.append((part / 'resultados.csv').read_bytes())
                log(f"  interrumpida después de {done} filas")
            else:
                raise AssertionError("the fake evaluator was supposed to interrupt")
        assert check_or_write_config(part, cfg)
        fe2 = st_fake_eval(state)
        assert ab_loop(state, ResultsStore(part / 'resultados.csv', keys), fe2, verbose=False) == len(DATASETS) * N_EVALS - done
        assert fe2.calls[0] == len(DATASETS) * N_EVALS - done
        assert (part / 'resultados.csv').read_bytes() == text_full, "resumed resultados.csv differs from the full run"
        fe3 = st_fake_eval(state)
        assert ab_loop(state, ResultsStore(part / 'resultados.csv', keys), fe3, verbose=False) == 0 and fe3.calls[0] == 0
        log(f"  reanudada: {len(DATASETS) * N_EVALS - done} evaluaciones nuevas; resultados.csv idéntico byte a byte a la "
            f"corrida completa; una tercera corrida no evalúa nada")
        # existing rows are never rewritten: the file at every interruption is a prefix of the final file
        assert all(text_full.startswith(x) for x in snapshots) and len({len(x) for x in snapshots}) == len(cuts)
        changed = json.loads(json.dumps(cfg))
        changed['epsilons'] = [1.0, 4.0, 16.0]
        expect_exit(lambda: check_or_write_config(part, changed), 'config.json con otros ε')
        changed = json.loads(json.dumps(cfg))
        changed['versiones']['torch'] = '0.0.0'
        expect_exit(lambda: check_or_write_config(part, changed), 'config.json con otra versión de torch')
        # a cache that changes on disk changes its sha256 -> abort
        p_cache = Path(paths['diffs_dir']) / 'refusal.pt'
        orig = p_cache.read_bytes()
        raw = torch.load(p_cache, weights_only=True)
        raw['example_pos'] = raw['example_pos'] + ' '
        torch.save(raw, p_cache)
        cfg_mod = dict(cfg, por_dataset={**cfg['por_dataset'], 'refusal': {**cfg['por_dataset']['refusal'], 'cache': {
            'path': cfg['por_dataset']['refusal']['cache']['path'], 'sha256': sha256_file(p_cache)}}})
        expect_exit(lambda: check_or_write_config(part, cfg_mod), 'sha256 de un caché distinto')
        p_cache.write_bytes(orig)
        assert build_config('selftest', ST_MODEL, paths, state) == cfg
        # corrupt / foreign rows abort
        lines = text_full.decode('utf-8').splitlines(keepends=True)
        for label, text in (('última línea incompleta', ''.join(lines[:10])[:-1]),
                            ('fila duplicada', ''.join(lines[:10] + lines[5:6])),
                            ('clave ajena (λ=−2)', ''.join(lines[:3]) + lines[3].replace(',2.0,', ',-2.0,', 1)),
                            ('ventana distinta', ''.join(lines[:3]) + lines[3].replace(',12-16,', ',11-15,', 1))):
            bad_dir = tmp / f"bad_{abs(hash(label))}"
            bad_dir.mkdir()
            (bad_dir / 'resultados.csv').write_bytes(text.encode('utf-8'))
            expect_exit(lambda: ResultsStore(bad_dir / 'resultados.csv', keys), label)
        expect_exit(lambda: check_or_write_config(tmp / 'inputs', cfg), 'carpeta existente sin config.json')
        log("  OK")

        log("#" * 100)
        log("SELFTEST D — IC 95% (t de Student) y diferencia apareada con números conocidos")
        t975_9 = 2.262157163                             # t_{0.975, 9} from a table (9 decimals), not from scipy
        m, sd, hw = ci95(np.arange(1.0, 11.0))
        assert abs(m - 5.5) < 1e-12 and abs(sd - math.sqrt(110 / 12)) < 1e-12
        assert abs(hw - t975_9 * math.sqrt(110 / 12) / math.sqrt(10)) < 1e-8
        log(f"  x = 1..10: media {m}, sd {sd:.6f} (= √(110/12)), semiancho {hw:.6f} (= 2.262157·sd/√10)")
        p = 0.50 + 0.01 * np.arange(10)
        dd = np.array([0.02, -0.01, 0.03, 0.00, 0.05, -0.02, 0.01, 0.04, 0.02, -0.03])
        g = p + dd
        st = paired_stats(p, g)
        d_sd = float(np.sqrt(((dd - dd.mean()) ** 2).sum() / 9))
        assert abs(st['diff'] - 0.011) < 1e-12 and abs(st['diff_hi'] - st['diff'] - t975_9 * d_sd / math.sqrt(10)) < 1e-8
        assert st['k'] == 6 and st['K'] == 10                    # 0.00 is not counted as PSA-1G > PSA
        assert abs(st['psa'] - 0.545) < 1e-12 and abs(st['g1'] - 0.556) < 1e-12
        log(f"  diferencia apareada: media {st['diff']:+.4f} (= 0.011), IC [{st['diff_lo']:+.5f}, {st['diff_hi']:+.5f}], "
            f"PSA-1G > PSA en {st['k']}/10 (un empate exacto no cuenta)")
        # dataset_stats / global_stats on a results frame with those numbers
        dg = dd.copy()
        dg[3] = 0.015                                  # no zero difference: rounding could make it positive
        gg = p + dg
        dg_sd = float(np.sqrt(((dg - dg.mean()) ** 2).sum() / 9))
        rows = []
        for k_, name in enumerate(PUBLIC):
            shift = 0.1 * k_
            rows.append({'dataset': name, 'condicion': NONE, 'epsilon': float('nan'), 'semilla': float('nan'),
                         'lambda': 0.0, 'ab': 0.3 + shift})
            for lam in LAMBDAS:
                rows.append({'dataset': name, 'condicion': PSA0, 'epsilon': float('nan'), 'semilla': float('nan'),
                             'lambda': lam, 'ab': 0.4 + shift})
                for eps in EPSILONS:
                    for s_, (a, b) in enumerate(zip(p, gg)):
                        # dataset k_ adds (k_ - 3) * 0.001 to the difference: the mean over the 7 datasets cancels it
                        rows.append({'dataset': name, 'condicion': PSA, 'epsilon': eps, 'semilla': float(s_),
                                     'lambda': lam, 'ab': a + shift})
                        rows.append({'dataset': name, 'condicion': G1, 'epsilon': eps, 'semilla': float(s_),
                                     'lambda': lam, 'ab': b + shift + (k_ - 3) * 0.001})
        fr = pd.DataFrame(rows)
        s2 = dataset_stats(fr, PUBLIC[6], 2.0, 4.0)
        assert abs(s2['diff'] - (dg.mean() + 0.003)) < 1e-12 and s2['k'] == int((dg + 0.003 > 0).sum()) == 7
        gl = global_stats(fr, PUBLIC, 1.0, 8.0)
        assert abs(gl['diff'] - dg.mean()) < 1e-12 and abs(gl['diff_hi'] - gl['diff'] - t975_9 * dg_sd / math.sqrt(10)) < 1e-8
        assert gl['k'] == 7 and abs(gl['floor'] - 0.6) < 1e-12 and abs(gl['psa0'] - 0.7) < 1e-12
        assert abs(gl['psa'] - 0.845) < 1e-12 and abs(gl['g1'] - (0.845 + dg.mean())) < 1e-12
        log(f"  tabla global: D_s = media sobre 7 datasets (los desplazamientos por dataset se cancelan); diferencia "
            f"{gl['diff']:+.4f} (= {dg.mean():+.4f}), IC = el apareado de las diferencias, "
            f"piso {gl['floor']:.2f}, PSA sin ruido {gl['psa0']:.2f}, PSA {gl['psa']:.3f}, PSA-1G {gl['g1']:.3f}")
        log("  OK")

        log("#" * 100)
        log(f"SELFTEST E — comparación con el piloto (--pilot-csv, tol {PILOT_TOL:g})")
        pil = res_full[res_full['dataset'].isin(PILOT_DATASETS)
                       & ((res_full['lambda'] == PILOT_LAMBDA) | (res_full['condicion'] == NONE))
                       & (res_full['semilla'].isna() | res_full['semilla'].isin(PILOT_SEEDS))].drop(columns=['ventana'])
        # the pilot also had lambda = -2 rows; they must be ignored
        neg = pil.assign(**{'lambda': -2.0, 'ab': 0.0})[pil['condicion'] != NONE]
        pilot_df = pd.concat([pil, neg], ignore_index=True)
        lines, nbad = compare_with_pilot(res_full, pilot_df)
        assert nbad == 0, lines
        log(lines[0])
        pert = pilot_df.copy()
        i_small = pert.index[(pert['condicion'] == PSA) & (pert['lambda'] == 2.0)][0]
        i_big = pert.index[(pert['condicion'] == G1) & (pert['lambda'] == 2.0)][0]
        pert.loc[i_small, 'ab'] += 5e-7
        pert.loc[i_big, 'ab'] += 2e-6
        lines, nbad = compare_with_pilot(res_full, pert)
        assert nbad == 1 and any('AVISO' in x and G1 in x for x in lines), lines
        log(f"  perturbación de 5e-7 ignorada, de 2e-6 avisada: {lines[1].strip()[:110]}")
        lines, nbad = compare_with_pilot(res_full, pilot_df.drop(index=i_big))
        assert nbad == 1, lines
        log("  OK")

        log("#" * 100)
        log("SELFTEST F — resumen.txt con los resultados de la corrida sintética")
        text = write_resumen(tmp / 'resumen.txt', ST_MODEL, state, res_full, checks, ['  (selftest: sin chequeo 4)'],
                             None, None)
        assert '[IN-SAMPLE]' in text and f"{OWN_DATASET} (IN-SAMPLE)" in text and 'AVISO: tofu-forget10 es IN-SAMPLE' in text
        assert 'recorte_total = False' in text and 'TABLA GLOBAL' in text
        for name in DATASETS:
            assert f"{name}{'  [IN-SAMPLE]' if name == OWN_DATASET else ''} | ventana {ST_WINDOWS[name]}" in text
        log("\n".join(text.splitlines()[-16:]))
        log("  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("#" * 100)
    log(f"SELFTEST PASSED in {time.time() - t0:.1f} s (temp dir removed; nothing written in the repo)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", choices=list(MODELS), help="sets the model and the default paths")
    parser.add_argument("--diffs-dir", default=None, help="default sweep_<tag>/diffs")
    parser.add_argument("--c-selection", default=None, help="default sweep_c_<tag>/c_selection.csv")
    parser.add_argument("--rbar-csv", default=None,
                        help="default resultados_rbar_<tag>/rbar_results.csv; mandatory (R̄ of every cache must match)")
    parser.add_argument("--out-dir", default=None, help="default nivel3_<tag>; if it exists, the run resumes")
    parser.add_argument("--pilot-csv", default=None,
                        help="1B only: pilot_1b/resultados.csv, compared at the end (λ=+2, seeds 0-1; warns above 1e-6)")
    parser.add_argument("--selftest", action='store_true', help="CPU, synthetic, no download, writes only to a temp dir")
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
        return
    if args.model_tag is None:
        parser.error("--model-tag is required")
    tag = args.model_tag
    args.diffs_dir = args.diffs_dir or f"sweep_{tag}/diffs"
    args.c_selection = args.c_selection or f"sweep_c_{tag}/c_selection.csv"
    args.rbar_csv = args.rbar_csv or f"resultados_rbar_{tag}/rbar_results.csv"
    args.out_dir = args.out_dir or f"nivel3_{tag}"
    if not Path(args.c_selection).is_file():
        raise SystemExit(f"{args.c_selection} not found")
    if args.pilot_csv is not None:
        if tag != '1b':
            parser.error("--pilot-csv only makes sense with --model-tag 1b")
        if not Path(args.pilot_csv).is_file():
            raise SystemExit(f"{args.pilot_csv} not found")
    run(args)


if __name__ == '__main__':
    main()
