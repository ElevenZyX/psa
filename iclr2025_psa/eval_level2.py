"""
Level 2 steering evaluation: the official TOFU metrics of open-unlearning (OU) on forget10, with the tofu-forget10 steering
vectors of the level 3, one model per job (--model-tag 1b / 3b / 8b).

Protocol (approved by Diego, 2026-10-22):
  * fp32 + sdpa: the models are loaded by utils.get_model_and_tokenizer(load_bfloat=False), as in the levels 1-3. Never bf16
    or flash_attention_2 (OU's model configs ask for both; their model_args are not used).
  * OU at commit OU_COMMIT, used from source through sys.path (--ou-dir, default ../open-unlearning), never pip-installed
    (OU pins transformers 5.x; steering-vectors 0.12.1 requires transformers < 5). OU's `trainer` package is never imported
    (it pulls deepspeed); its seed_everything is replicated below. The OU config is composed with hydra.compose
    (experiment=eval/tofu/default, model=<OU config of the model>, eval.tofu.batch_size=--batch-size); every locuslab/TOFU
    dataset node (23) is pinned at TOFU_REVISION.
  * Metrics (configs/eval/tofu.yaml): forget_truth_ratio, forget_quality, forget_Q_A_Prob, forget_Q_A_ROUGE, model_utility
    (harmonic mean of 9 components), privleak, extraction_strength. Generation for ROUGE: greedy, max_new_tokens 200.
  * Prompt: OU's template_args of the model (system "You are a helpful assistant.") with date_string = utils.CHAT_DATE
    ("26 Jul 2024"), passed to Evaluator.evaluate(template_args=...). Checked on item 0 of every dataset of the suite before
    any model is loaded: exactly one BOS, exactly one "Today Date: 26 Jul 2024" line, that system prompt; full and retain90
    must produce the same token ids. OU's collator and attention mask are kept (attention_mask = input_ids != pad, pad =
    <|eot_id|>: the <|eot_id|> closing the system and user turns are masked; level 3 masks nothing).
  * Reference: open-unlearning/tofu_<model>_retain90 evaluated here, without steering, with exactly the same protocol and
    GPU type. Its TOFU_EVAL.json is the retain_logs_path of every other condition of the model (forget_quality, privleak).
    OU's official logs (bf16 + flash_attention_2, "10 Apr 2025") are only a control: "sin steering" vs
    <model>_full/evals_forget10 and the reference vs <model>_retain90, AVISO above |Δ| 0.02 on [0, 1] metrics, 1 on
    log10(forget_quality), 2 points on privleak; never aborts.
  * Steering: SteeringVector.apply(model, multiplier=λ, min_token_index=0) around Evaluator.evaluate, on the tofu-forget10
    window of the model (c_selection.csv): every token, also in the generation used for ROUGE (prefill and every decode step).

Vectors, identical to eval_level3.py for the same seeds: eval_level3.read_c_selection / load_window_cache (R̄ cross-check
mandatory), sweep_c.paired_draws (one generator per seed), eval_pilot.build_vectors, check_priv_mean_replication and
PSA == run.py priv_mean with the same z.

Conditions: retain90 (referencia) and sin steering (λ = 0, shared by every λ); per λ: Mean Steering (paper), PSA sin ruido,
PSA-1G sin ruido, and PSA / PSA-1G for every ε in --epsilons (1 4 8) and seed in --seeds (0..4). Subsets: --conditions
(referencia sin-steering mean psa0 g1_0 psa g1), --only-reference, --only-noiseless (referencia, sin steering and the three
noiseless vectors), --lams (default 1), --seeds, --epsilons.

Checks (all abort unless marked AVISO): OU checkout at OU_COMMIT with a clean tree, OU packages imported from --ou-dir,
transformers < 5, steering_vectors 0.12.1, TF32 off; c_selection / caches / R̄ (as level 3); --level3-config (optional):
the tofu-forget10 inputs of the level 3 are the same files; composed suite (7 metrics, 23 TOFU nodes, generation args,
batch); prompts (above); model snapshots (config._commit_hash), fp32, sdpa, GPU type == protocol; first training prompt
rebuilt == the cache's (real models); check 4 (Mean Steering from the cache == train_steering_vector + run.py scaled_mean,
tol 1e-5) and a layer probe (the output of every window layer jumps by exactly λ·v_l on every token) before the first
steered condition; no steering hook left after every condition; OU warnings that mean "no reference" abort every
non-reference condition (tokenization mismatches are counted, AVISO); per condition, model_utility == hmean of its 9
components, forget_Q_A_ROUGE == mean of the per-item ROUGE-L recall, forget_truth_ratio and forget_quality (KS vs the
reference) and privleak recomputed from the per-item logs.

Outputs (--out-dir, default nivel2_<tag>; one per job): config.json (protocolo + first job), eval_config.yaml (the composed
OU suite), lanzamientos.jsonl (one line per launch), resultados.csv (append-only, atomic, one row = one finished
condition), one directory per condition (TOFU_EVAL.json and TOFU_SUMMARY.json of OU, config.json, tiempos.json),
resumen.txt (rebuilt from resultados.csv at the end of every launch). Relaunching the same command resumes: rows already in
resultados.csv are skipped; a condition directory without a row is deleted and recomputed (OU's own resumption is never
used: overwrite=True always). A different protocol (model, batch, prompt, GPU type, versions, inputs, ...) aborts.
merge_level2.py joins the resultados.csv of several jobs of the same model.

    python eval_level2.py --selftest      # CPU, tiny random Llama + local TOFU-like data, nothing downloaded, temp dir only
    python eval_level2.py --model-tag 3b --conditions referencia sin-steering psa0 --lams 1 2 --out-dir nivel2_3b_piloto
    python eval_level2.py --model-tag 8b --conditions psa g1 --seeds 0 1 2 \\
                          --reference nivel2_8b_a/referencia_retain90 --out-dir nivel2_8b_b
"""
import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval_level3
from cache_alignment import OWN_DATASET, load_rbar_csv
from eval_level3 import (CALIBRATION, DELTA, N_SAMPLES, S, SHUFFLE_SEED, dict_diff, expect_exit, fmt_float, fmt_seed,
                         library_versions, load_window_cache, read_c_selection, selection_line, sha256_file, sigma_lines)
from eval_pilot import (CHECK4_TOL, G1, G1_0, MEAN, NONE, PSA, PSA0, Report, build_vectors, check_psa_equals_priv_mean,
                        compare_mean_vectors, find_vec, log, n_forward_hooks, scaled_mean_as_in_run_py, to_steering_vector)
from measure_rbar import READ_TOKEN_INDEX
from sweep_c import MODELS, paired_draws
from sweep_epsilon import Accountant, check_priv_mean_replication, ci95, cosine

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
OU_COMMIT = 'b29929067732c40ee4ca58d5cc1d3c11fcf37de1'
TOFU_PATH = 'locuslab/TOFU'
TOFU_REVISION = '324592d84ae4f482ac7249b9285c2ecdb53e3a68'
TOFU_NODES = {'forget10_perturbed': 8, 'retain_perturbed': 4, 'real_authors_perturbed': 5, 'world_facts_perturbed': 5,
              'holdout10': 1}                       # hf_args nodes of the composed suite per TOFU config (23 in total)
OU_MODEL_CFG = {'1b': 'Llama-3.2-1B-Instruct', '3b': 'Llama-3.2-3B-Instruct', '8b': 'Llama-3.1-8B-Instruct'}
RETAIN_MODELS = {'1b': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90',
                 '3b': 'open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90',
                 '8b': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90'}
MODEL_SHAS = {   # HF snapshot commits (unchanged since 2025-02-24); model.config._commit_hash must be these
    'open-unlearning/tofu_Llama-3.2-1B-Instruct_full': '88e31200b97e4c0c04ae0d2f0b591f427046d192',
    'open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90': '7114300c0049527a71833f5683965c358ad9dcbf',
    'open-unlearning/tofu_Llama-3.2-3B-Instruct_full': '24f31ca19f6966dcb6f6b29abc511cce71222d4a',
    'open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90': '2c35819e9a731d096df9abd6efbcc361870c5333',
    'open-unlearning/tofu_Llama-3.1-8B-Instruct_full': '1a5c5b1a557f8c99bdadecd5168ebd03f640b00e',
    'open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90': 'f02956a9f25a18f38ad7808fbefb2f957347273e',
}
OFFICIAL_PREFIX = {'1b': 'tofu_Llama-3.2-1B-Instruct', '3b': 'tofu_Llama-3.2-3B-Instruct', '8b': 'tofu_Llama-3.1-8B-Instruct'}
LLAMA3_BOS, LLAMA3_EOT = 128000, 128009
OU_SYSTEM_PROMPT = 'You are a helpful assistant.'
SYSTEM_HEADER = '<|start_header_id|>system<|end_header_id|>\n\n'
ASSISTANT_HEADER = '<|start_header_id|>assistant<|end_header_id|>\n\n'
EOT = '<|eot_id|>'
IGNORE_INDEX = -100                                  # OU src/data/utils.py
EXPECTED_GENERATION = {'do_sample': False, 'top_p': None, 'temperature': None, 'max_new_tokens': 200, 'use_cache': True}

TOP_METRICS = ['forget_truth_ratio', 'forget_quality', 'forget_Q_A_Prob', 'forget_Q_A_ROUGE', 'model_utility', 'privleak',
               'extraction_strength']             # configs/eval/tofu.yaml, in OU's evaluation order
MU_COMPONENTS = ['retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio', 'ra_Q_A_Prob_normalised', 'ra_Q_A_ROUGE',
                 'ra_Truth_Ratio', 'wf_Q_A_Prob_normalised', 'wf_Q_A_ROUGE', 'wf_Truth_Ratio']   # model_utility.yaml order
MAIN_METRICS = ['forget_quality', 'model_utility', 'forget_Q_A_ROUGE']
OTHER_METRICS = ['forget_Q_A_Prob', 'forget_truth_ratio', 'privleak', 'extraction_strength']
METRIC_COLUMNS = MAIN_METRICS + OTHER_METRICS + MU_COMPONENTS
UNIT_KEYS = ['forget_Q_A_Prob', 'forget_Q_A_ROUGE', 'forget_truth_ratio', 'model_utility', 'extraction_strength'] + MU_COMPONENTS
RESULT_COLUMNS = (['modelo', 'ventana', 'condicion', 'epsilon', 'semilla', 'lambda'] + METRIC_COLUMNS
                  + ['vector_sha256', 'referencia_sha256', 'segundos'])
HEADER = ','.join(RESULT_COLUMNS) + '\n'
METRIC_LABEL = {m: m for m in METRIC_COLUMNS}
METRIC_LABEL['forget_quality'] = 'log10(forget_quality)'

REF = 'retain90 (referencia)'
CONDITION_ORDER = [REF, NONE, MEAN, PSA0, G1_0, PSA, G1]
CLI_CONDITIONS = {'referencia': REF, 'sin-steering': NONE, 'mean': MEAN, 'psa0': PSA0, 'g1_0': G1_0, 'psa': PSA, 'g1': G1}
NOISELESS = [REF, NONE, MEAN, PSA0, G1_0]
UNSTEERED = (REF, NONE)
PER_LAMBDA = (MEAN, PSA0, G1_0)
NOISY = (PSA, G1)
ALLOWED_EPSILONS = (1.0, 4.0, 8.0)
DEFAULT_EPSILONS = [1.0, 4.0, 8.0]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_LAMBDAS = [1.0]
DEFAULT_BATCH = 32
FORGET_N = 400
TOL_UNIT, TOL_LOG10FQ, TOL_PRIVLEAK = 0.02, 1.0, 2.0
RECOMPUTE_RTOL = 1e-9
FATAL_OU_WARNINGS = ('setting forget_quality to None', 'using default retain auc', 'evals not present in')
COUNTED_OU_WARNINGS = {'tokenization_mismatch': 'Tokenization mismatch', 'indice_0_con_loss': 'must not have loss',
                       'em_none': 'EM score for an instance is marked None',
                       'es_none': 'ES score for an instance is marked None'}
COND_FILES = {'TOFU_EVAL.json', 'TOFU_SUMMARY.json', 'config.json', 'tiempos.json'}


class Interrupted(Exception):
    """Raised by the selftest's fault injection to simulate a job killed in the middle of a condition."""


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_metric(x) -> str:
    return '' if is_missing(x) else repr(float(x))


def canon(obj):
    return json.loads(json.dumps(obj))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def vector_sha(vec: torch.Tensor, dtype=torch.float64) -> str:
    """sha256 of the (S, h) vector (shape included); float64 = as build_vectors makes it, float32 = as applied."""
    a = vec.detach().cpu().to(dtype).contiguous().numpy()
    return hashlib.sha256(json.dumps(list(a.shape)).encode() + a.tobytes()).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def lam_slug(lam: float) -> str:
    return f"lam{float(lam):+g}"


COND_SLUG = {REF: 'referencia_retain90', NONE: 'sin_steering', MEAN: 'mean', PSA0: 'psa_sin_ruido', G1_0: 'psa1g_sin_ruido',
             PSA: 'psa', G1: 'psa1g'}


def task_rel_dir(cond: str, eps, seed, lam) -> str:
    """Directory of a condition inside the job directory (relative, posix)."""
    if cond in UNSTEERED:
        return COND_SLUG[cond]
    if cond in PER_LAMBDA:
        return f"{lam_slug(lam)}/{COND_SLUG[cond]}"
    if cond in NOISY:
        return f"{lam_slug(lam)}/{COND_SLUG[cond]}/eps{float(eps):g}_s{int(seed)}"
    raise ValueError(f"unknown condition {cond!r}")


def task_rel_dir_from_key(key: tuple) -> str:
    _, cond, eps, seed, lam = key
    return task_rel_dir(cond, float(eps) if eps != '' else float('nan'), int(seed) if seed != '' else None, float(lam))


def task_key(model_tag: str, cond: str, eps, seed, lam) -> tuple:
    return model_tag, cond, fmt_float(eps), fmt_seed(seed), fmt_float(lam)


def key_from_strings(r: dict) -> tuple:
    e, s = r['epsilon'], r['semilla']
    return (r['modelo'], r['condicion'], '' if e == '' else repr(float(e)), '' if s == '' else str(int(float(s))),
            repr(float(r['lambda'])))


def row_sort_key(cond: str, eps, seed, lam) -> tuple:
    """Canonical order (reference, sin steering, then by λ: Mean, PSA s/r, PSA-1G s/r, then ε, seed, PSA before PSA-1G).
    eps / seed / lam may be numbers or the strings of resultados.csv."""
    if cond in UNSTEERED:
        return 0, CONDITION_ORDER.index(cond), 0.0, 0, 0.0, 0
    lam = float(lam)
    if cond in PER_LAMBDA:
        return 1, lam, 0, CONDITION_ORDER.index(cond), 0.0, 0
    return 1, lam, 1, float(eps), int(float(seed)), NOISY.index(cond)


def task_label(t: dict) -> str:
    c = t['condicion']
    if c in NOISY:
        return f"{c} ε={t['epsilon']:g} s={int(t['semilla'])} λ={t['lambda']:+g}"
    if c in PER_LAMBDA:
        return f"{c} λ={t['lambda']:+g}"
    return c


# --------------------------------------------------------------------------------------
# resultados.csv
# --------------------------------------------------------------------------------------
def _bad(label: str, msg: str):
    raise SystemExit(f"{label}: {msg}. Stopping.")


def validate_row(r: dict, model_tag: str, label: str) -> None:
    """Structural checks of one row of resultados.csv (strings, as csv.DictReader gives them)."""
    if None in r or any(v is None for v in r.values()):
        _bad(label, f"wrong number of fields {r}")
    try:
        if r['modelo'] != model_tag:
            _bad(label, f"modelo {r['modelo']!r} != {model_tag!r}")
        cond = r['condicion']
        if cond not in CONDITION_ORDER:
            _bad(label, f"condicion {cond!r} unknown")
        lam = float(r['lambda'])
        eps, seed = r['epsilon'], r['semilla']
        if cond in UNSTEERED:
            ok = eps == '' and seed == '' and lam == 0.0
        elif cond in PER_LAMBDA:
            ok = eps == '' and seed == '' and lam != 0.0 and math.isfinite(lam)
        else:
            ok = (eps != '' and seed != '' and lam != 0.0 and math.isfinite(lam) and float(eps) in ALLOWED_EPSILONS
                  and float(seed) == int(float(seed)) and int(float(seed)) >= 0)
        if not ok:
            _bad(label, f"epsilon {eps!r} / semilla {seed!r} / lambda {r['lambda']!r} do not fit {cond!r}")
        for m in METRIC_COLUMNS:
            v = r[m]
            if cond == REF and m in ('forget_quality', 'privleak'):
                if v != '':
                    _bad(label, f"{m} must be empty on the reference row (not defined against itself), got {v!r}")
                continue
            if v == '' or not math.isfinite(float(v)):
                _bad(label, f"{m} = {v!r} on {cond!r}")
        vs = r['vector_sha256']
        if (cond in UNSTEERED and vs != '') or (cond not in UNSTEERED and not re.fullmatch(r'[0-9a-f]{64}', vs)):
            _bad(label, f"vector_sha256 {vs!r} does not fit {cond!r}")
        if not re.fullmatch(r'[0-9a-f]{64}', r['referencia_sha256']):
            _bad(label, f"referencia_sha256 {r['referencia_sha256']!r}")
        if not float(r['segundos']) >= 0:
            _bad(label, f"segundos {r['segundos']!r}")
    except ValueError as e:
        _bad(label, f"unparseable value ({e})")


def parse_results_text(text: str, model_tag: str, label: str) -> dict:
    if not text.startswith(HEADER):
        raise SystemExit(f"{label}: header is not {HEADER.strip()!r}. Stopping.")
    if not text.endswith('\n'):
        raise SystemExit(f"{label}: the last line is incomplete. Stopping (fix the file by hand).")
    rows = {}
    for i, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        validate_row(r, model_tag, f"{label} line {i}")
        k = key_from_strings(r)
        if k in rows:
            raise SystemExit(f"{label} line {i}: duplicated key {k}. Stopping.")
        rows[k] = r
    return rows


def serialize_row(row: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(
        [row['modelo'], row['ventana'], row['condicion'], fmt_float(row['epsilon']), fmt_seed(row['semilla']),
         repr(float(row['lambda']))] + [fmt_metric(row[m]) for m in METRIC_COLUMNS]
        + [row['vector_sha256'] or '', row['referencia_sha256'], f"{float(row['segundos']):.1f}"])
    return buf.getvalue()


def strings_line(r: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow([r[c] for c in RESULT_COLUMNS])
    return buf.getvalue()


class ResultsStore:
    """resultados.csv of one job, append-only like eval_level3.ResultsStore: every append re-reads the file, checks that
    it is exactly what this process last wrote, and os.replace()s old text + the new line (never a partial line, rows
    never rewritten)."""

    def __init__(self, path: Path, model_tag: str):
        self.path, self.model_tag = path, model_tag
        if path.is_file():
            self.text = self._read()
        else:
            self.text = HEADER
            write_text_atomic(path, HEADER)
        self.rows = parse_results_text(self.text, model_tag, str(path))

    def _read(self) -> str:
        with open(self.path, encoding='utf-8', newline='') as f:
            return f.read()

    def __contains__(self, key: tuple) -> bool:
        return key in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def append(self, row: dict) -> None:
        line = serialize_row(row)
        r = next(csv.DictReader(io.StringIO(HEADER + line)))
        validate_row(r, self.model_tag, f"{self.path} (new row)")
        k = key_from_strings(r)
        if k in self.rows:
            raise SystemExit(f"refusing to write {k} twice")
        if self._read() != self.text:
            raise SystemExit(f"{self.path} was modified by someone else during the run. Stopping.")
        new = self.text + line
        write_text_atomic(self.path, new)
        self.text = new
        self.rows[k] = r


def read_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[''],
                       dtype={'modelo': str, 'ventana': str, 'condicion': str, 'vector_sha256': str,
                              'referencia_sha256': str})


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-tag", choices=sorted(OU_MODEL_CFG), help="model; sets the default paths")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--conditions", nargs='+', choices=list(CLI_CONDITIONS),
                   help="exact subset of conditions (default: all)")
    g.add_argument("--only-reference", action='store_true', help="only retain90 (referencia)")
    g.add_argument("--only-noiseless", action='store_true',
                   help="referencia, sin steering, Mean Steering, PSA sin ruido, PSA-1G sin ruido")
    p.add_argument("--lams", type=float, nargs='+', default=DEFAULT_LAMBDAS, help="steering multipliers (default 1)")
    p.add_argument("--seeds", type=int, nargs='+', default=DEFAULT_SEEDS, help="noise seeds (default 0 1 2 3 4)")
    p.add_argument("--epsilons", type=float, nargs='+', default=DEFAULT_EPSILONS, help="subset of 1 4 8 (default all)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="eval.tofu.batch_size of every metric (default 32); part of the protocol: the reference must match")
    p.add_argument("--reference", default=None,
                   help="referencia_retain90 directory of another job (default: <out-dir>/referencia_retain90)")
    p.add_argument("--out-dir", default=None, help="default nivel2_<tag>; one per job; if it exists, the job resumes")
    p.add_argument("--ou-dir", default='../open-unlearning', help="open-unlearning checkout at OU_COMMIT")
    p.add_argument("--official-logs", default=None, help="OU's saves/eval (setup_data.py --eval_logs); "
                                                          "default <ou-dir>/saves/eval")
    p.add_argument("--diffs-dir", default=None, help="default sweep_<tag>/diffs")
    p.add_argument("--c-selection", default=None, help="default sweep_c_<tag>/c_selection.csv")
    p.add_argument("--rbar-csv", default=None, help="default resultados_rbar_<tag>/rbar_results.csv (mandatory)")
    p.add_argument("--level3-config", default=None,
                   help="optional nivel3_<tag>/config.json: its tofu-forget10 inputs must be the same files")
    p.add_argument("--selftest", action='store_true', help="CPU, tiny random model, local data, temp dir only")
    return p


def resolve_args(args) -> argparse.Namespace:
    """Validates the flags and fills the defaults; adds args.conds (canonical order)."""
    if args.only_reference:
        conds = {REF}
    elif args.only_noiseless:
        conds = set(NOISELESS)
    elif args.conditions:
        if len(set(args.conditions)) != len(args.conditions):
            raise SystemExit(f"--conditions has repeated values {args.conditions}")
        conds = {CLI_CONDITIONS[c] for c in args.conditions}
    else:
        conds = set(CONDITION_ORDER)
    args.conds = [c for c in CONDITION_ORDER if c in conds]
    if args.reference is not None and REF in conds:
        raise SystemExit("--reference uses the reference of another job: choose the conditions with --conditions "
                         "without 'referencia'")
    if any(not math.isfinite(l) or l == 0.0 for l in args.lams) or len(set(args.lams)) != len(args.lams):
        raise SystemExit(f"--lams must be distinct, finite and non-zero, got {args.lams}")
    if len({lam_slug(l) for l in args.lams}) != len(args.lams):
        raise SystemExit(f"--lams {args.lams}: two values share the directory name {[lam_slug(l) for l in args.lams]}")
    if any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit(f"--seeds must be distinct and >= 0, got {args.seeds}")
    if any(e not in ALLOWED_EPSILONS for e in args.epsilons) or len(set(args.epsilons)) != len(args.epsilons):
        raise SystemExit(f"--epsilons must be distinct values of {ALLOWED_EPSILONS}, got {args.epsilons}")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size must be > 0, got {args.batch_size}")
    tag = args.model_tag
    for name, default in (('diffs_dir', 'sweep_{}/diffs'), ('c_selection', 'sweep_c_{}/c_selection.csv'),
                          ('rbar_csv', 'resultados_rbar_{}/rbar_results.csv'), ('out_dir', 'nivel2_{}')):
        if getattr(args, name) is None:
            if tag is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required without --model-tag")
            setattr(args, name, default.format(tag))
    if args.official_logs is None:
        args.official_logs = str(Path(args.ou_dir) / 'saves' / 'eval')
    return args


# --------------------------------------------------------------------------------------
# Model specification
# --------------------------------------------------------------------------------------
@dataclass
class Spec:
    tag: str                      # resultados.csv 'modelo' ('1b' / '3b' / '8b', 'selftest')
    cache_model: str              # model name in the caches / c_selection.csv / R̄ csv
    full: str                     # what utils.get_model_and_tokenizer loads
    retain: str
    ou_model_cfg: str             # OU configs/model/<name>.yaml (template_args only)
    full_sha: str | None
    retain_sha: str | None
    forget_n: int
    official_prefix: str | None
    bos_id: int
    eos_id: int
    selftest_data: dict | None = None     # {TOFU config: local jsonl}; None = locuslab/TOFU at TOFU_REVISION
    selftest_cache: str | None = None
    gen_max_new_tokens: int | None = None

    @property
    def real(self) -> bool:
        return self.selftest_data is None


def real_spec(tag: str) -> Spec:
    full, retain = MODELS[tag], RETAIN_MODELS[tag]
    return Spec(tag=tag, cache_model=full, full=full, retain=retain, ou_model_cfg=OU_MODEL_CFG[tag],
                full_sha=MODEL_SHAS[full], retain_sha=MODEL_SHAS[retain], forget_n=FORGET_N,
                official_prefix=OFFICIAL_PREFIX[tag], bos_id=LLAMA3_BOS, eos_id=LLAMA3_EOT)


# --------------------------------------------------------------------------------------
# Environment and open-unlearning
# --------------------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stdout, force=True)


def git(ou_dir: Path, *a: str) -> str:
    r = subprocess.run(['git', '-C', str(ou_dir), *a], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(a)} in {ou_dir} failed: {r.stderr.strip()}")
    return r.stdout


def check_ou_checkout(ou_dir: Path) -> str:
    if not (ou_dir / 'src' / 'evals').is_dir() or not (ou_dir / 'configs' / 'eval.yaml').is_file():
        raise SystemExit(f"{ou_dir} is not an open-unlearning checkout (no src/evals, configs/eval.yaml); "
                         f"run setup_tesis_ou.sh or pass --ou-dir")
    head = git(ou_dir, 'rev-parse', 'HEAD').strip()
    if head != OU_COMMIT:
        raise SystemExit(f"{ou_dir} is at {head}, expected {OU_COMMIT} (git -C {ou_dir} checkout {OU_COMMIT})")
    dirty = git(ou_dir, 'status', '--porcelain', '--untracked-files=all').strip()
    if dirty:
        raise SystemExit(f"{ou_dir} has local changes or untracked files; refusing to use it:\n{dirty[:2000]}")
    msg = f"open-unlearning {ou_dir} en {OU_COMMIT}, árbol limpio"
    log(msg)
    return msg


def import_ou(ou_dir: Path) -> None:
    """OU's src/ on sys.path right after this script's directory; its packages must come from there."""
    src = (ou_dir / 'src').resolve()
    if str(src) not in sys.path:
        sys.path.insert(1, str(src))
    try:
        import data          # noqa: F401  (OU)
        import evals         # noqa: F401  (OU; imports lm_eval)
        import evals.metrics  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"importing open-unlearning from {src} failed: {e!r}. Was setup_tesis_ou.sh run in this "
                         f"environment (hydra-core, rouge-score, lm-eval)?")
    for name in ('data', 'evals', 'evals.metrics'):
        f = Path(sys.modules[name].__file__).resolve()
        if src not in f.parents:
            raise SystemExit(f"module {name!r} was imported from {f}, not from {src}. Stopping.")
    if 'trainer' in sys.modules:
        raise SystemExit("open-unlearning's `trainer` package got imported (it pulls deepspeed); it must not be. Stopping.")


def check_environment() -> str:
    import steering_vectors
    import transformers
    problems = []
    if int(transformers.__version__.split('.')[0]) >= 5:
        problems.append(f"transformers {transformers.__version__} (steering-vectors 0.12.1 needs < 5)")
    if steering_vectors.__version__ != '0.12.1':
        problems.append(f"steering_vectors {steering_vectors.__version__} != 0.12.1")
    if torch.backends.cuda.matmul.allow_tf32:
        problems.append("torch.backends.cuda.matmul.allow_tf32 is True (fp32 protocol)")
    if torch.get_float32_matmul_precision() != 'highest':
        problems.append(f"float32_matmul_precision {torch.get_float32_matmul_precision()!r} != 'highest'")
    if problems:
        raise SystemExit("environment does not match the protocol: " + "; ".join(problems))
    msg = (f"entorno: transformers {transformers.__version__}, steering_vectors {steering_vectors.__version__}, "
           f"torch {torch.__version__}, TF32 apagado")
    log(msg)
    return msg


def seed_everything(seed: int = 0) -> None:
    """Verbatim open-unlearning src/trainer/utils.py:8-14 (not imported: OU's trainer package pulls deepspeed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def versions() -> dict:
    v = library_versions()
    for name in ('datasets', 'tokenizers', 'accelerate', 'huggingface-hub', 'hydra-core', 'omegaconf', 'rouge-score',
                 'nltk', 'lm-eval', 'scikit-learn'):
        try:
            v[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            v[name] = None
    return v


def device_info() -> dict:
    import utils
    if str(utils.device).startswith('cuda'):
        return {'tipo': 'cuda', 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version()}
    return {'tipo': 'cpu', 'gpu': 'cpu', 'cuda': None, 'cudnn': None}


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Instrumentation of OU's metric objects (timing; fault injection and hook probe for the selftest)
# --------------------------------------------------------------------------------------
class MetricInstrument:
    """Wraps, in memory, the `evaluate` method of every metric object of OU's METRICS_REGISTRY (the same singletons the
    evaluator and its pre_compute metrics use): wall time per metric with its nesting level. OU's source is not modified;
    uninstall() restores the class method. The selftest checks that results are identical with and without it."""

    def __init__(self):
        self.wrapped = []
        self.records: list[dict] = []
        self.depth = 0
        self.calls = 0
        self.fault_at = None
        self.probe = None

    def install(self) -> None:
        if self.wrapped:
            return
        from evals.metrics import METRICS_REGISTRY
        for m in {id(x): x for x in METRICS_REGISTRY.values()}.values():
            m.evaluate = self._wrap(m.evaluate)            # instance attribute shadows UnlearningMetric.evaluate
            self.wrapped.append(m)

    def uninstall(self) -> None:
        for m in self.wrapped:
            del m.evaluate
        self.wrapped = []

    def begin(self) -> None:
        self.records = []
        self.depth = 0

    def arm_fault(self, n_calls: int) -> None:
        self.fault_at = self.calls + n_calls

    def _wrap(self, orig):
        def timed(model, metric_name, *args, **kwargs):
            self.calls += 1
            if self.fault_at is not None and self.calls >= self.fault_at:
                self.fault_at = None
                raise Interrupted(f"selftest: interrupción simulada en {metric_name}")
            depth = self.depth
            self.depth += 1
            p0 = self.probe() if self.probe else None
            t0 = time.time()
            try:
                return orig(model, metric_name, *args, **kwargs)
            finally:
                dt = time.time() - t0
                self.depth = depth
                rec = {'metrica': metric_name, 'nivel': depth, 'segundos': round(dt, 3)}
                if self.probe:
                    rec['probe'] = self.probe() - p0
                self.records.append(rec)
                log(f"  [tiempo] {'  ' * depth}{metric_name}: {dt:.1f} s")
        return timed


INSTRUMENT = MetricInstrument()


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.name}: {record.getMessage()}")


@contextmanager
def capture_ou_warnings():
    """Logging WARNINGs that reach the root logger (OU's 'evaluator', 'metrics' and 'data' loggers) and Python warnings."""
    h = _WarningCapture()
    root = logging.getLogger()
    root.addHandler(h)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            yield h.records, w
    finally:
        root.removeHandler(h)


def classify_warnings(records: list[str], pywarn: list, is_ref: bool, label: str) -> dict:
    msgs = list(records) + [f"python {x.category.__name__}: {x.message}" for x in pywarn]
    fatal = [m for m in msgs if any(p in m for p in FATAL_OU_WARNINGS)]
    if fatal and not is_ref:
        raise SystemExit(f"{label}: open-unlearning warned {fatal[:3]}: forget_quality / privleak computed without the "
                         f"reference. Stopping.")
    for m in fatal:
        log(f"  (esperado en la referencia) {m[:200]}")
    counts = {k: sum(p in m for m in msgs) for k, p in COUNTED_OU_WARNINGS.items()}
    for k, c in counts.items():
        if c:
            log(f"  <-- AVISO: {label}: {c} × {k} (open-unlearning)")
    others = Counter(m[:200] for m in msgs
                     if m not in fatal and not any(p in m for p in COUNTED_OU_WARNINGS.values()))
    for m, c in others.most_common(8):
        log(f"  aviso de librería ×{c}: {m}")
    return counts


# --------------------------------------------------------------------------------------
# OU configuration
# --------------------------------------------------------------------------------------
def ou_nodes(node, key: str, path: str = 'eval.tofu') -> list:
    """(node, path) of every DictConfig under `node` (itself included) that has `key`, depth first, in config order."""
    from omegaconf import DictConfig
    out = []
    if isinstance(node, DictConfig):
        if key in node:
            out.append((node, path))
        for k in node.keys():
            child = node[k]
            if isinstance(child, DictConfig):
                out += ou_nodes(child, key, f"{path}.{k}")
    return out


def compose_eval_cfg(ou_dir: Path, spec: Spec, batch_size: int):
    """OU's eval config for the TOFU suite (as src/eval.py builds it with experiment=eval/tofu/default), with the TOFU
    revision pinned (or the selftest's local files) and the batch size; checked against what this script assumes.
    Returns (cfg, generation_args). The two hydra/*_logging overrides only replace OU's colorlog logging config (a plugin
    that is not installed); they do not touch the evaluation."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    overrides = ['experiment=eval/tofu/default', f'model={spec.ou_model_cfg}', f'eval.tofu.batch_size={batch_size}',
                 'task_name=nivel2', 'hydra/job_logging=default', 'hydra/hydra_logging=default']
    with initialize_config_dir(config_dir=str((ou_dir / 'configs').resolve()), version_base=None):
        cfg = compose(config_name='eval.yaml', overrides=overrides)
    if list(cfg.eval.keys()) != ['tofu'] or cfg.eval.tofu.handler != 'TOFUEvaluator':
        raise SystemExit(f"composed eval config has evaluators {list(cfg.eval.keys())}, expected only TOFUEvaluator")
    tofu = cfg.eval.tofu
    if list(tofu.metrics.keys()) != TOP_METRICS:
        raise SystemExit(f"OU's TOFU suite has metrics {list(tofu.metrics.keys())}, expected {TOP_METRICS}")
    if cfg.retain_logs_path is not None:
        raise SystemExit(f"retain_logs_path is {cfg.retain_logs_path!r} after composing, expected null")
    hf = ou_nodes(tofu, 'hf_args')
    names, bad = Counter(), []
    for node, path in hf:
        a = node.hf_args
        if a.path != TOFU_PATH or a.split != 'train' or 'revision' in a:
            bad.append(f"{path}: {OmegaConf.to_container(a, resolve=True)}")
        names[a.name] += 1
    if bad or dict(names) != TOFU_NODES:
        raise SystemExit(f"TOFU dataset nodes of the suite: {dict(names)} (expected {TOFU_NODES}); unexpected: {bad[:3]}")
    gens = ou_nodes(tofu, 'generation_args')
    with open_dict(cfg):
        for node, _ in hf:
            if spec.real:
                node.hf_args.revision = TOFU_REVISION
            else:
                node.hf_args = {'path': 'json', 'data_files': spec.selftest_data[node.hf_args.name], 'split': 'train',
                                'cache_dir': spec.selftest_cache}
        if spec.gen_max_new_tokens is not None:
            for node, _ in gens:
                node.generation_args.max_new_tokens = spec.gen_max_new_tokens
    gen = [OmegaConf.to_container(node.generation_args, resolve=True) for node, _ in gens]
    want = dict(EXPECTED_GENERATION)
    if spec.gen_max_new_tokens is not None:
        want['max_new_tokens'] = spec.gen_max_new_tokens
    if len(gen) != 4 or any(g != want for g in gen):
        raise SystemExit(f"generation_args of the 4 ROUGE metrics: {gen}, expected 4 x {want}")
    if spec.real:
        revs = {node.hf_args.revision for node, _ in hf}
        if revs != {TOFU_REVISION}:
            raise SystemExit(f"TOFU revisions after pinning: {revs}")
    bs = {node.batch_size for node, _ in ou_nodes(tofu, 'batch_size')}
    if bs != {batch_size}:
        raise SystemExit(f"batch sizes in the suite {bs}, expected {{{batch_size}}}")
    return cfg, gen[0]


def make_template_args(cfg) -> dict:
    from omegaconf import OmegaConf
    from utils import CHAT_DATE
    ta = OmegaConf.to_container(cfg.model.template_args, resolve=True)
    if ta.get('apply_chat_template') is not True or ta.get('system_prompt') != OU_SYSTEM_PROMPT:
        raise SystemExit(f"OU template_args {ta} do not apply the chat template with system {OU_SYSTEM_PROMPT!r}")
    ta['date_string'] = CHAT_DATE
    return ta


def eval_yaml(cfg) -> str:
    """The resolved eval suite with the reference path replaced by a placeholder (same text for every condition)."""
    from omegaconf import OmegaConf
    c = copy.deepcopy(cfg)
    c.retain_logs_path = 'RETAIN_LOGS_PATH'
    return OmegaConf.to_yaml(c.eval, resolve=True)


def make_evaluator(cfg, ref_json: Path | None):
    """OU's TOFUEvaluator with retain_logs_path = ref_json (None for the reference itself)."""
    from evals import get_evaluators
    c = copy.deepcopy(cfg)
    c.retain_logs_path = None if ref_json is None else str(Path(ref_json).resolve())
    for m in ('forget_quality', 'privleak'):
        got = c.eval.tofu.metrics[m].reference_logs.retain_model_logs.path
        if got != c.retain_logs_path:
            raise SystemExit(f"{m}.reference_logs path {got!r} != {c.retain_logs_path!r}")
    ev = get_evaluators(c.eval)
    if list(ev) != ['tofu']:
        raise SystemExit(f"get_evaluators returned {list(ev)}")
    return ev['tofu']


# --------------------------------------------------------------------------------------
# Prompt check
# --------------------------------------------------------------------------------------
def suite_dataset_nodes(cfg) -> list:
    """(name, dataset node) of every distinct dataset configuration the suite builds."""
    from omegaconf import OmegaConf
    seen, out = set(), []
    for node, path in ou_nodes(cfg.eval.tofu, 'datasets'):
        for dname in node.datasets.keys():
            d = node.datasets[dname]
            ident = json.dumps({'handler': d.handler, 'args': OmegaConf.to_container(d.args, resolve=True)}, sort_keys=True)
            if ident not in seen:
                seen.add(ident)
                out.append((f"{path.removeprefix('eval.tofu.metrics.')}.{dname}", d))
    return out


def check_tokenizer(tok, spec: Spec, label: str) -> str:
    ids = (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
    if ids != (spec.bos_id, spec.eos_id, spec.eos_id):
        raise SystemExit(f"[{label}] tokenizer bos/eos/pad ids {ids}, expected ({spec.bos_id}, {spec.eos_id}, "
                         f"{spec.eos_id}) (OU pads with <|eot_id|>)")
    if not isinstance(tok.chat_template, str) or not tok.chat_template:
        raise SystemExit(f"[{label}] tokenizer has no chat template")
    return sha256_text(tok.chat_template)


def check_prompt_item(lab: str, ids: list, labels: list, answer: str, gen: bool, tok) -> str:
    from utils import check_today_date
    bos = tok.bos_token_id
    if not ids or ids[0] != bos or ids.count(bos) != 1:
        raise SystemExit(f"{lab}: expected exactly one BOS ({bos}) at the start; ids[:4] = {ids[:4]}, "
                         f"{ids.count(bos)} BOS. Stopping.")
    text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    check_today_date(text, lab + ' ')                      # SystemExit unless exactly one "Today Date: CHAT_DATE" line
    if text.count(SYSTEM_HEADER) != 1:
        raise SystemExit(f"{lab}: {text.count(SYSTEM_HEADER)} system headers in {text[:300]!r}. Stopping.")
    system = text.split(SYSTEM_HEADER, 1)[1].split(EOT, 1)[0]
    if not system.endswith('\n\n' + OU_SYSTEM_PROMPT) or 'honest' in system:
        raise SystemExit(f"{lab}: system block {system!r} does not end in the system prompt {OU_SYSTEM_PROMPT!r}. Stopping.")
    if gen and not text.endswith(ASSISTANT_HEADER):
        raise SystemExit(f"{lab}: the generation prompt does not end in the assistant header: {text[-80:]!r}. Stopping.")
    target = tok.decode([t for t in labels if t != IGNORE_INDEX], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False)
    if not target.endswith(answer.strip() + EOT):
        log(f"  <-- AVISO: {lab}: the labels decode to {target[-120:]!r}, not ending in the answer + <|eot_id|> "
            f"{(answer.strip() + EOT)[-120:]!r}")
    return text


def prompt_check(cfg, tok, ta: dict, label: str, verbose: bool) -> tuple[str, list[dict]]:
    """Builds item 0 of every dataset of the suite with OU's own dataset classes and our template_args; checks it.
    Returns (fingerprint = sha256 of every checked token id and label, the items)."""
    from data import get_datasets          # OU
    fp, items = [], []
    for name, d in suite_dataset_nodes(cfg):
        ds = get_datasets(dataset_cfgs={name: d}, tokenizer=tok, template_args=ta)
        item = ds[0]
        if 'input_ids' not in item:                        # several answers (perturbed): their first one
            item = item[0]
        answer = ds.data[0][ds.answer_key]
        answer = answer[0] if isinstance(answer, list) else answer
        gen = bool(ds.predict_with_generate)
        ids, labels = item['input_ids'].tolist(), item['labels'].tolist()
        lab = f"[{label}] {name} ({'generación' if gen else 'probabilidad'})"
        text = check_prompt_item(lab, ids, labels, answer, gen, tok)
        if verbose:
            log(f"  {lab}: {text!r}")
        fp.append([name, gen, ids, labels])
        items.append({'name': name, 'gen': gen, 'ids': ids})
    return sha256_json(fp), items


# --------------------------------------------------------------------------------------
# Vectors (no model)
# --------------------------------------------------------------------------------------
def preflight_vectors(spec: Spec, paths: dict, epsilons: list, seeds: list, checks: list) -> dict:
    sel = read_c_selection(paths['c_selection'], spec.cache_model, [OWN_DATASET])[OWN_DATASET]
    log(selection_line(OWN_DATASET, sel))
    if not sel['recorte_total']:
        log(f"  <-- AVISO: recorte_total = False en c_selection.csv para {OWN_DATASET}; se evalúa igual")
    rbar_csv = load_rbar_csv(Path(paths['rbar_csv']))
    cache = load_window_cache(Path(paths['diffs_dir']), OWN_DATASET, sel['layers'], spec.cache_model, rbar_csv,
                              paths['rbar_csv'])
    h = cache['hidden']
    Z = paired_draws(seeds, S, h)
    acc = Accountant(CALIBRATION, cache['n'], DELTA, S)
    vecs, clipfrac = build_vectors(cache['d32'], sel['C'], sel['Cg'], acc, Z, epsilons, seeds)
    s_chk = acc.sigma_psa(epsilons[0])
    check_priv_mean_replication(cache['d32'][:, 0, :], C=sel['C'], sigma=s_chk)
    checks.append(check_psa_equals_priv_mean(cache['d32'], sel['C'], s_chk, Z[0],
                                             find_vec(vecs, PSA, epsilons[0], seeds[0])['vec'],
                                             f"[{OWN_DATASET}] PSA ε={epsilons[0]:g} s={seeds[0]} ventana {sel['ventana']}"))
    checks.append(f"c_selection, caché (metadatos, permutación, R̄ vs {paths['rbar_csv']}) y check_priv_mean_replication OK; "
                  f"ventana {sel['ventana']}, C_eval {sel['C']:g}, Cg_eval {sel['Cg']:.4f}")
    return {'sel': sel, 'cache': cache, 'h': h, 'acc': acc, 'vectors': vecs, 'clipfrac': clipfrac,
            'layers': sel['layers'], 'ventana': sel['ventana'], 'epsilons': list(epsilons), 'seeds': list(seeds)}


def inputs_block(spec: Spec, paths: dict, state: dict) -> dict:
    sel, cache = state['sel'], state['cache']
    return {'modelo_cache': spec.cache_model, 'dataset': OWN_DATASET,
            'c_selection': {'path': Path(paths['c_selection']).as_posix(), 'sha256': sha256_file(Path(paths['c_selection']))},
            'cache': {'path': Path(cache['path']).as_posix(), 'sha256': sha256_file(Path(cache['path']))},
            'rbar_csv': Path(paths['rbar_csv']).as_posix(), 'ventana': sel['ventana'], 'capas': sel['layers'],
            'C_eval': sel['C'], 'Cg_eval': sel['Cg'], 'recorte_total': sel['recorte_total'], 'n': cache['n'], 'S': S,
            'delta': DELTA, 'calibracion': CALIBRATION, 'read_token_index': READ_TOKEN_INDEX, 'n_samples': N_SAMPLES,
            'shuffle_seed': SHUFFLE_SEED, 'min_token_index': 0}


def check_level3_config(path: Path, entradas: dict) -> str:
    cfg3 = json.loads(path.read_text(encoding='utf-8'))
    e = cfg3.get('por_dataset', {}).get(OWN_DATASET)
    if e is None:
        raise SystemExit(f"{path} has no entry for {OWN_DATASET}")
    diffs = []
    if cfg3.get('model') != entradas['modelo_cache']:
        diffs.append(f"model {cfg3.get('model')!r} != {entradas['modelo_cache']!r}")
    if e['cache']['sha256'] != entradas['cache']['sha256']:
        diffs.append("sha256 of the tofu-forget10 cache")
    if cfg3['c_selection']['sha256'] != entradas['c_selection']['sha256']:
        diffs.append("sha256 of c_selection.csv")
    for k3, k2 in (('ventana', 'ventana'), ('C_eval', 'C_eval'), ('Cg_eval', 'Cg_eval')):
        if e[k3] != entradas[k2]:
            diffs.append(f"{k3} {e[k3]!r} != {entradas[k2]!r}")
    if diffs:
        raise SystemExit(f"{path}: the level 3 used other inputs for {OWN_DATASET}: {diffs}. Stopping.")
    msg = f"--level3-config {path}: mismo caché, c_selection, ventana, C_eval y Cg_eval que el nivel 3 para {OWN_DATASET}"
    log(msg)
    return msg


def build_tasks(conds: list, lams: list, epsilons: list, seeds: list, vecs: list) -> list[dict]:
    nan = float('nan')
    tasks = []

    def add(cond, eps, seed, lam, v):
        tasks.append({'condicion': cond, 'epsilon': eps, 'semilla': seed, 'lambda': lam, 'v': v,
                      'vec': None if v is None else v['vec']})

    for cond in UNSTEERED:
        if cond in conds:
            add(cond, nan, nan, 0.0, None)
    for lam in lams:
        for cond in PER_LAMBDA:
            if cond in conds:
                add(cond, nan, nan, lam, find_vec(vecs, cond))
        for eps in epsilons:
            for seed in seeds:
                for cond in NOISY:
                    if cond in conds:
                        add(cond, eps, seed, lam, find_vec(vecs, cond, eps, seed))
    return tasks


# --------------------------------------------------------------------------------------
# Job directory
# --------------------------------------------------------------------------------------
def check_or_write_job_config(out_dir: Path, protocol: dict, job: dict) -> bool:
    """True if resuming (config.json exists and its protocol is exactly this one); the job part may differ."""
    p = out_dir / 'config.json'
    if out_dir.exists():
        if not p.is_file():
            raise SystemExit(f"{out_dir} exists but has no config.json; cannot tell what it contains. Stopping.")
        saved = json.loads(p.read_text(encoding='utf-8'))
        if saved.get('script') != 'eval_level2.py':
            raise SystemExit(f"{p} was not written by eval_level2.py. Stopping.")
        diff = dict_diff(saved.get('protocolo', {}), protocol)
        if diff:
            raise SystemExit(f"{p}: the protocol does not match; refusing to mix results:\n  " + "\n  ".join(diff[:30]))
        log(f"{p}: mismo protocolo, se reanuda")
        return True
    out_dir.mkdir(parents=True)
    write_json_atomic(p, {'script': 'eval_level2.py', 'protocolo': protocol, 'creado': now_iso(), 'job_inicial': job})
    log(f"escrito {p}")
    return False


def record_launch(out_dir: Path, job: dict) -> None:
    line = json.dumps({'inicio': now_iso(), 'host': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                       'argv': sys.argv, 'job': job}, ensure_ascii=False)
    with open(out_dir / 'lanzamientos.jsonl', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def resolve_reference(ref_dir: Path, protocol: dict, model_tag: str, forget_n: int) -> dict:
    """A finished reference with exactly this protocol (GPU type included): its config.json, its TOFU_EVAL.json (sha256)
    and its row in the resultados.csv of its own job."""
    cfgp, evp = ref_dir / 'config.json', ref_dir / 'TOFU_EVAL.json'
    if not cfgp.is_file() or not evp.is_file():
        raise SystemExit(f"no finished reference in {ref_dir}: run the reference first (--only-reference or --conditions "
                         f"referencia ...) or pass --reference <dir>. Stopping.")
    rc = json.loads(cfgp.read_text(encoding='utf-8'))
    if rc.get('condicion') != REF:
        raise SystemExit(f"{cfgp} is not a reference (condicion {rc.get('condicion')!r}). Stopping.")
    diff = dict_diff(rc.get('protocolo', {}), protocol)
    if diff:
        raise SystemExit(f"the reference {ref_dir} was computed with another protocol (GPU type, versions, prompt, batch, "
                         f"inputs, ...):\n  " + "\n  ".join(diff[:30]))
    sha = sha256_file(evp)
    if sha != rc.get('tofu_eval_sha256'):
        raise SystemExit(f"{evp} changed after the reference was written (sha256 {sha} != {rc.get('tofu_eval_sha256')}). "
                         f"Stopping.")
    parent_csv = ref_dir.parent / 'resultados.csv'
    if not parent_csv.is_file():
        raise SystemExit(f"the reference {ref_dir} has no {parent_csv}. Stopping.")
    rows = parse_results_text(parent_csv.read_text(encoding='utf-8'), model_tag, str(parent_csv))
    k = task_key(model_tag, REF, float('nan'), float('nan'), 0.0)
    if k not in rows or rows[k]['referencia_sha256'] != sha:
        raise SystemExit(f"the reference {ref_dir} is not finished (no row in {parent_csv} with its sha256). Stopping.")
    logs = json.loads(evp.read_text(encoding='utf-8'))
    tr = (logs.get('forget_truth_ratio') or {}).get('value_by_index')
    if not tr or len(tr) != forget_n or (logs.get('mia_min_k') or {}).get('agg_value') is None:
        raise SystemExit(f"{evp}: forget_truth_ratio has {len(tr or {})} items (expected {forget_n}) or no mia_min_k. "
                         f"Stopping.")
    log(f"referencia: {ref_dir} (TOFU_EVAL.json sha256 {sha[:16]}…, mismo protocolo, {len(tr)} truth ratios de forget)")
    return {'dir': ref_dir, 'eval_json': evp, 'sha256': sha, 'logs': logs}


# --------------------------------------------------------------------------------------
# Model-side checks
# --------------------------------------------------------------------------------------
def load_model_checked(name: str, expected_sha: str | None, protocol: dict, state: dict, label: str):
    from utils import get_model_and_tokenizer
    log("=" * 100)
    log(f"cargando {label}: {name} (utils.get_model_and_tokenizer, load_bfloat=False)")
    t0 = time.time()
    model, tok = get_model_and_tokenizer(name, load_bfloat=False)
    model.eval()
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    problems = []
    if dtype != torch.float32:
        problems.append(f"dtype {dtype}, expected float32")
    attn = getattr(model.config, '_attn_implementation', None)
    if attn != 'sdpa':
        problems.append(f"attention {attn!r}, expected 'sdpa'")
    if expected_sha is not None and getattr(model.config, '_commit_hash', None) != expected_sha:
        problems.append(f"snapshot {getattr(model.config, '_commit_hash', None)!r} != {expected_sha!r}")
    if model.config.hidden_size != state['h'] or model.config.num_hidden_layers <= max(state['layers']):
        problems.append(f"hidden {model.config.hidden_size} / {model.config.num_hidden_layers} layers incompatible with "
                        f"the cache (hidden {state['h']}, window {state['layers']})")
    gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    if gpu != protocol['dispositivo']['gpu']:
        problems.append(f"device {gpu!r} != protocol {protocol['dispositivo']['gpu']!r}")
    if check_tokenizer(tok, Spec('', '', '', '', '', None, None, 0, None, protocol['prompts']['bos'],
                                 protocol['prompts']['eos']), label) != protocol['prompts']['chat_template_sha256']:
        problems.append("chat template differs from the one of the prompt check")
    if problems:
        raise SystemExit(f"[{label}] " + "; ".join(problems) + ". Stopping.")
    log(f"  {label}: {dev}, {dtype}, attention {attn}, {model.config.num_hidden_layers} capas, hidden "
        f"{model.config.hidden_size}, snapshot {getattr(model.config, '_commit_hash', None)} ({time.time() - t0:.0f} s)")
    return model, tok


def check_layer_jumps(model, ids: list, sv, lam: float, vec: torch.Tensor, layers: list, base_hooks: int) -> str:
    """In one forward pass, the output of every window layer right after the steering hook minus right before it must be
    λ·v_l on every token (forward hooks run in registration order: 'before' hooks are registered before sv.apply)."""
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    before, after = {}, {}

    def mk(store, i):
        def hook(_m, _a, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    hs = [model.model.layers[i].register_forward_hook(mk(before, i)) for i in layers]
    try:
        with torch.no_grad(), sv.apply(model, multiplier=lam, min_token_index=0):
            hs += [model.model.layers[i].register_forward_hook(mk(after, i)) for i in layers]
            model(x)
    finally:
        for h in hs:
            h.remove()
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit("layer probe: a hook was left on the model. Stopping.")
    worst = 0.0
    for j, l in enumerate(layers):
        b, a = before[l], after[l]
        target = (lam * vec[j].to(a.dtype)).to(a.device).expand_as(a)
        err = ((a - b) - target).abs()
        tol = 4 * torch.finfo(a.dtype).eps * (b.abs() + target.abs()) + 1e-30
        if bool((err > tol).any()):
            raise SystemExit(f"layer probe: the output of layer {l} did not jump by λ·v (max err {err.max().item():.3e}). "
                             f"Stopping.")
        worst = max(worst, err.max().item())
    msg = (f"sonda de capas: la salida de las capas {layers} salta exactamente λ·v_l (λ={lam:+g}) en los {len(ids)} tokens "
           f"(max err {worst:.1e}); hooks retirados")
    log(f"  {msg}")
    return msg


def check4_tofu(model, tok, state: dict, base_hooks: int) -> list[str]:
    from steering_vectors import train_steering_vector
    from utils import make_dataset
    layers, c = state['layers'], state['cache']
    log("=" * 100)
    log(f"CHEQUEO 4 — Mean Steering: caché vs train_steering_vector + scaled_mean (run.py:37-41), {OWN_DATASET}, ventana "
        f"{layers}, mismos {c['n']} pares")
    with torch.no_grad():
        sv_lib = train_steering_vector(model, tok, make_dataset(c['train_rows'], tok), read_token_index=READ_TOKEN_INDEX,
                                       show_progress=True, aggregator=scaled_mean_as_in_run_py, layers=layers)
    if sorted(sv_lib.layer_activations) != layers:
        raise SystemExit(f"train_steering_vector returned layers {sorted(sv_lib.layer_activations)}, expected {layers}")
    lib = torch.stack([sv_lib.layer_activations[l].detach().cpu().double() for l in layers])
    lines = compare_mean_vectors(lib, find_vec(state['vectors'], MEAN)['vec'], layers, CHECK4_TOL)
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit("train_steering_vector left hooks on the model. Stopping.")
    log(f"  assert OK (tol {CHECK4_TOL:g})")
    return [f"chequeo 4 OK en {OWN_DATASET} (ventana {state['ventana']}): max |diff| por capa ≤ {CHECK4_TOL:g}"] + lines


# --------------------------------------------------------------------------------------
# One evaluation
# --------------------------------------------------------------------------------------
def evaluate_once(evaluator, model, tok, ta: dict, out: Path, sv, lam: float, base_hooks: int, is_ref: bool,
                  label: str) -> dict:
    """OU's TOFU suite on `model` (steered by sv x lam if sv is given) into `out`, overwrite=True."""
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit(f"{label}: {n_forward_hooks(model)} forward hooks before the evaluation, expected {base_hooks}")
    seed_everything(0)
    INSTRUMENT.begin()
    calls0 = INSTRUMENT.calls
    with capture_ou_warnings() as (records, pywarn):
        t0 = time.time()
        if sv is None:
            summary = evaluator.evaluate(model=model, tokenizer=tok, template_args=ta, output_dir=str(out), overwrite=True)
        else:
            with sv.apply(model, multiplier=lam, min_token_index=0):
                summary = evaluator.evaluate(model=model, tokenizer=tok, template_args=ta, output_dir=str(out),
                                             overwrite=True)
        secs = time.time() - t0
    if n_forward_hooks(model) != base_hooks:
        raise SystemExit(f"{label}: {n_forward_hooks(model)} forward hooks after the evaluation, expected {base_hooks}: "
                         f"a steering hook was not removed. Stopping.")
    counts = classify_warnings(records, pywarn, is_ref, label)
    logs = json.loads((out / 'TOFU_EVAL.json').read_text(encoding='utf-8'))
    return {'summary': dict(summary), 'logs': logs, 'secs': secs, 'tiempos': list(INSTRUMENT.records),
            'avisos_ou': counts, 'llamadas': INSTRUMENT.calls - calls0}


def _close(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= RECOMPUTE_RTOL * max(abs(a), abs(b))


def extract_metrics(summary: dict, logs: dict, is_ref: bool, ref_logs: dict | None, forget_n: int,
                    label: str) -> tuple[dict, float | None]:
    """The CSV metrics of one evaluation, recomputed independently from the per-item logs where possible.
    Returns (values, privleak against OU's default 0.5 for the reference, else None)."""
    from scipy.stats import hmean, ks_2samp
    need = [m for m in TOP_METRICS if not (is_ref and m == 'forget_quality')]
    miss = [m for m in need if summary.get(m) is None]
    if miss:
        raise SystemExit(f"{label}: OU returned no {miss} (summary {sorted(summary)}). Stopping.")
    if is_ref and summary.get('forget_quality') is not None:
        raise SystemExit(f"{label}: the reference got a forget_quality ({summary['forget_quality']}); it must be computed "
                         f"without reference logs. Stopping.")
    vals = {m: float(summary[m]) for m in need}
    for c in MU_COMPONENTS:
        v = (logs.get(c) or {}).get('agg_value')
        if v is None:
            raise SystemExit(f"{label}: TOFU_EVAL.json has no {c}.agg_value. Stopping.")
        vals[c] = float(v)
    problems = []
    hm = float(hmean([vals[c] for c in MU_COMPONENTS]))
    if not _close(hm, vals['model_utility']):
        problems.append(f"model_utility {vals['model_utility']!r} != hmean of the 9 components {hm!r}")
    rg = [v['rougeL_recall'] for v in logs['forget_Q_A_ROUGE']['value_by_index'].values()
          if v.get('rougeL_recall') is not None]
    if not _close(float(np.mean(rg)), vals['forget_Q_A_ROUGE']):
        problems.append(f"forget_Q_A_ROUGE {vals['forget_Q_A_ROUGE']!r} != mean of the per-item ROUGE-L recall")
    tr = np.array([v['score'] for v in logs['forget_truth_ratio']['value_by_index'].values()], dtype=float)
    if len(tr) != forget_n:
        problems.append(f"forget_truth_ratio has {len(tr)} items, expected {forget_n}")
    agg = float(np.mean(np.minimum(tr, 1 / (tr + 1e-10))))
    if not _close(agg, vals['forget_truth_ratio']):
        problems.append(f"forget_truth_ratio {vals['forget_truth_ratio']!r} != mean(min(R, 1/R)) {agg!r}")
    pl05 = None
    if is_ref:
        pl05 = vals['privleak']
        vals['privleak'] = None
        vals['forget_quality'] = None
    else:
        ref_tr = np.array([v['score'] for v in ref_logs['forget_truth_ratio']['value_by_index'].values()], dtype=float)
        p = float(ks_2samp(tr, ref_tr).pvalue)
        if not _close(p, vals['forget_quality']):
            problems.append(f"forget_quality {vals['forget_quality']!r} != KS p-value vs the reference {p!r}")
        s, r = float(logs['mia_min_k']['agg_value']), float(ref_logs['mia_min_k']['agg_value'])
        pl = ((1 - s) - (1 - r)) / ((1 - r) + 1e-10) * 100           # OU privacy.privleak with the reference's AUC
        if not _close(pl, vals['privleak']):
            problems.append(f"privleak {vals['privleak']!r} != {pl!r} recomputed from the mia_min_k AUCs")
    if problems:
        raise SystemExit(f"{label}: " + "; ".join(problems) + ". Stopping.")
    return vals, pl05


def clean_incomplete(cond_dir: Path, out_dir: Path, label: str) -> None:
    cd, od = cond_dir.resolve(), out_dir.resolve()
    if od not in cd.parents:
        raise SystemExit(f"{cond_dir} is not inside {out_dir}. Stopping.")
    entries = list(cond_dir.iterdir())
    allowed = COND_FILES | {f + '.tmp' for f in COND_FILES}
    unknown = [e.name for e in entries if e.name not in allowed or not e.is_file()]
    if unknown:
        raise SystemExit(f"{cond_dir} holds files that are not a condition's ({unknown}); nothing deleted. Stopping.")
    log(f"  <-- AVISO: {label}: {cond_dir} existe sin fila en resultados.csv (corrida interrumpida); se borran "
        f"{sorted(e.name for e in entries)} y se recalcula")
    for e in entries:
        e.unlink()
    cond_dir.rmdir()


def run_condition(ctx: dict, t: dict, evaluator, model, tok, sv, ref: dict | None, base_hooks: int) -> dict:
    spec, state, out_dir, protocol = ctx['spec'], ctx['state'], ctx['out_dir'], ctx['protocol']
    cond, eps, seed, lam = t['condicion'], t['epsilon'], t['semilla'], t['lambda']
    is_ref = cond == REF
    rel = task_rel_dir(cond, eps, seed, lam)
    cond_dir = out_dir / rel
    label = task_label(t)
    log("-" * 100)
    log(f"{label} → {cond_dir}")
    if cond_dir.exists():
        clean_incomplete(cond_dir, out_dir, label)
    cond_dir.mkdir(parents=True)
    r = evaluate_once(evaluator, model, tok, ctx['ta'], cond_dir, sv, lam, base_hooks, is_ref, label)
    vals, pl05 = extract_metrics(r['summary'], r['logs'], is_ref, None if is_ref else ref['logs'], spec.forget_n, label)
    eval_sha = sha256_file(cond_dir / 'TOFU_EVAL.json')
    ref_sha = eval_sha if is_ref else ref['sha256']
    v = t['v']
    vinfo = None if v is None else {
        'sha256': vector_sha(v['vec']), 'sha256_fp32': vector_sha(v['vec'], torch.float32),
        'norma_por_capa': [float(x) for x in v['vec'].norm(dim=1)], 'sigma': v['sigma'], 'sigma_coord': v['sigma_coord']}
    write_json_atomic(cond_dir / 'tiempos.json', {'registros': r['tiempos'], 'segundos_total': r['secs'],
                                                  'llamadas_metricas': r['llamadas']})
    write_json_atomic(cond_dir / 'config.json', canon({
        'condicion': cond, 'epsilon': fmt_float(eps), 'semilla': fmt_seed(seed), 'lambda': float(lam), 'directorio': rel,
        'ventana': state['ventana'], 'capas': state['layers'], 'C_eval': state['sel']['C'], 'Cg_eval': state['sel']['Cg'],
        'vector': vinfo, 'protocolo': protocol, 'protocolo_sha256': sha256_json(protocol),
        'referencia': None if is_ref else {'dir': str(ref['dir']), 'sha256': ref['sha256']},
        'tofu_eval_sha256': eval_sha, 'metricas': vals, 'privleak_vs_0.5_de_OU': pl05, 'segundos': r['secs'],
        'llamadas_metricas': r['llamadas'], 'avisos_ou': r['avisos_ou'], 'fin': now_iso(), 'host': platform.node(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID')}))
    row = {'modelo': spec.tag, 'ventana': state['ventana'], 'condicion': cond, 'epsilon': eps, 'semilla': seed,
           'lambda': lam, **vals, 'vector_sha256': '' if vinfo is None else vinfo['sha256'], 'referencia_sha256': ref_sha,
           'segundos': r['secs']}
    ctx['store'].append(row)
    log(f"  {label}: {METRIC_LABEL['forget_quality']} {fmt_num('forget_quality', tr_metric('forget_quality', vals['forget_quality']))}"
        f" | model_utility {vals['model_utility']:.4f} | forget_Q_A_ROUGE {vals['forget_Q_A_ROUGE']:.4f} | "
        f"{r['secs']:.0f} s | {r['llamadas']} métricas")
    if is_ref:
        log(f"  (referencia) privleak de open-unlearning contra su default 0.5 = {pl05:.4f}; no va al CSV")
    return row


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def tr_metric(metric: str, x):
    """Value shown and averaged in the summary: log10 for forget_quality (None if missing or 0)."""
    if is_missing(x):
        return None
    x = float(x)
    if metric == 'forget_quality':
        return math.log10(x) if x > 0 else None
    return x


def fmt_num(metric: str, x, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    spec = {'forget_quality': '.3f', 'privleak': '.2f'}.get(metric, '.4f')
    return format(float(x), ('+' if signed else '') + spec)


def single_value(res: pd.DataFrame, cond: str, lam: float, metric: str):
    r = res[(res['condicion'] == cond) & (res['lambda'] == lam)]
    if len(r) > 1:
        raise SystemExit(f"{cond} λ={lam:g}: {len(r)} rows")
    return None if r.empty else r[metric].iloc[0]


def seed_values(res: pd.DataFrame, cond: str, lam: float, eps: float, metric: str) -> dict:
    r = res[(res['condicion'] == cond) & (res['lambda'] == lam) & (res['epsilon'] == eps)]
    return {int(s): v for s, v in zip(r['semilla'], r[metric])}


def ci_cell(metric: str, x: np.ndarray, signed: bool = False) -> str:
    if len(x) == 0:
        return '—'
    if len(x) == 1:
        return f"{fmt_num(metric, x[0], signed)} (1 sem.)"
    m, _, h = ci95(x)
    return f"{fmt_num(metric, m, signed)} [{fmt_num(metric, m - h, signed)}, {fmt_num(metric, m + h, signed)}]"


def paired_cells(metric: str, p: dict, g: dict, n_seeds: int) -> str:
    pv = {s: tr_metric(metric, v) for s, v in p.items()}
    gv = {s: tr_metric(metric, v) for s, v in g.items()}
    zeros = sum(v is None for v in list(pv.values()) + list(gv.values()))
    pa = np.array([v for v in pv.values() if v is not None])
    ga = np.array([v for v in gv.values() if v is not None])
    common = sorted(s for s in set(pv) & set(gv) if pv[s] is not None and gv[s] is not None)
    d = np.array([gv[s] - pv[s] for s in common])
    notes = []
    if len(common) < n_seeds:
        notes.append(f"{len(common)}/{n_seeds} semillas apareadas")
    if zeros:
        notes.append(f"{zeros} valor(es) = 0 excluidos")
    return (f"{ci_cell(metric, pa):>32s} | {ci_cell(metric, ga):>32s} | {ci_cell(metric, d, True):>32s} | "
            f"{int((d > 0).sum())}/{len(common)}" + (f"  ({'; '.join(notes)})" if notes else ''))


def fixed_rows(lam: float | None) -> list[tuple[str, float]]:
    rows = [(REF, 0.0), (NONE, 0.0)]
    if lam is not None:
        rows += [(c, lam) for c in PER_LAMBDA]
    return rows


def stats_block(P: Report, res: pd.DataFrame, metric: str, lam: float | None, epsilons: list, n_seeds: int) -> None:
    P(f"    [{METRIC_LABEL[metric]}]")
    for cond, lc in fixed_rows(lam):
        x = single_value(res, cond, lc, metric)
        if cond == REF and metric in ('forget_quality', 'privleak'):
            shown = 'no aplica (referencia)'
        elif x is None:
            shown = '— (no está en estos resultados)'
        else:
            y = tr_metric(metric, x)
            shown = fmt_num(metric, y) if y is not None else 'forget_quality = 0 (underflow)'
        P(f"      {cond + (' (λ=0)' if cond in UNSTEERED else ''):34s} {shown}")
    if lam is None or not epsilons:
        return
    P(f"      {'ε':>4s} | {'PSA media [IC 95%]':>32s} | {'PSA-1G media [IC 95%]':>32s} | {'PSA-1G − PSA [IC 95%]':>32s} | "
      f"PSA-1G > PSA")
    for eps in epsilons:
        P(f"      {eps:4g} | " + paired_cells(metric, seed_values(res, PSA, lam, eps, metric),
                                               seed_values(res, G1, lam, eps, metric), n_seeds))


def components_table(P: Report, res: pd.DataFrame, lam: float | None, epsilons: list) -> None:
    short = ['ret_Prob', 'ret_ROUGE', 'ret_TR', 'ra_Prob_n', 'ra_ROUGE', 'ra_TR', 'wf_Prob_n', 'wf_ROUGE', 'wf_TR']
    P("    [componentes de model_utility] (" + ", ".join(f"{s} = {c}" for s, c in zip(short, MU_COMPONENTS)) + ")")
    P(f"      {'':34s} " + " ".join(f"{s:>10s}" for s in short))
    for cond, lc in fixed_rows(lam):
        vals = [single_value(res, cond, lc, c) for c in MU_COMPONENTS]
        if all(v is None for v in vals):
            continue
        P(f"      {cond:34s} " + " ".join(f"{fmt_num(c, v):>10s}" for c, v in zip(MU_COMPONENTS, vals)))
    if lam is None:
        return
    for eps in epsilons:
        for cond in NOISY:
            means = []
            for c in MU_COMPONENTS:
                v = [x for x in seed_values(res, cond, lam, eps, c).values() if not is_missing(x)]
                means.append(float(np.mean(v)) if v else None)
            if all(m is None for m in means):
                continue
            n = len(seed_values(res, cond, lam, eps, MU_COMPONENTS[0]))
            P(f"      {f'{cond} ε={eps:g} (media {n} sem.)':34s} "
              + " ".join(f"{fmt_num(c, m):>10s}" for c, m in zip(MU_COMPONENTS, means)))


def official_control(res: pd.DataFrame, official_dir, prefix: str | None) -> tuple[list[str], int]:
    """'sin steering' vs OU's <prefix>_full/evals_forget10 and the reference vs <prefix>_retain90 (AVISO, never aborts)."""
    if official_dir is None or prefix is None:
        return ["  control oficial no disponible (sin --official-logs o modelo sin logs oficiales)"], 0
    official_dir = Path(official_dir)
    lines, n_av = [], 0
    for cond, rel, with_fq in ((NONE, f"{prefix}_full/evals_forget10/TOFU_EVAL.json", True),
                               (REF, f"{prefix}_retain90/TOFU_EVAL.json", False)):
        r = res[res['condicion'] == cond]
        p = official_dir / rel
        if r.empty:
            lines.append(f"  {cond}: no está en estos resultados (no se compara)")
            continue
        if not p.is_file():
            lines.append(f"  <-- AVISO: {cond}: no existe {p} (setup_data.py --eval_logs)")
            n_av += 1
            continue
        off = json.loads(p.read_text(encoding='utf-8'))
        r = r.iloc[0]
        lines.append(f"  {cond} vs oficial {rel}:")
        keys = [(k, TOL_UNIT) for k in UNIT_KEYS]
        if with_fq:
            keys += [('forget_quality', TOL_LOG10FQ), ('privleak', TOL_PRIVLEAK)]
        for k, tol in keys:
            theirs = off[k].get('agg_value') if isinstance(off.get(k), dict) else None
            ours = r[k]
            if theirs is None or is_missing(ours):
                lines.append(f"    {METRIC_LABEL[k]:26s} sin dato (propio {ours}, oficial {theirs})")
                continue
            a, b = tr_metric(k, ours), tr_metric(k, theirs)
            if a is None or b is None:
                lines.append(f"    <-- AVISO: {METRIC_LABEL[k]}: valor 0 (propio {ours}, oficial {theirs})")
                n_av += 1
                continue
            d = a - b
            bad = abs(d) > tol
            n_av += int(bad)
            lines.append(f"    {METRIC_LABEL[k]:26s} propio {fmt_num(k, a):>9s} | oficial {fmt_num(k, b):>9s} | "
                         f"Δ {fmt_num(k, d, True):>9s}" + (f"  <-- AVISO (|Δ| > {tol:g})" if bad else ''))
    return lines, n_av


def vector_lines_l2(P: Report, vecs: list, layers: list, epsilons: list) -> None:
    """eval_level3.vector_lines for the job's epsilons."""
    ref = find_vec(vecs, MEAN)['vec']
    P(f"  {'condición':30s} " + " ".join(f"{'L' + str(l) + ' ‖v‖':>9s} {'cos':>6s}" for l in layers))

    def row(label, group):
        nrm = torch.stack([v['vec'].norm(dim=1) for v in group]).mean(dim=0)
        cs = torch.stack([cosine(v['vec'], ref) for v in group]).mean(dim=0)
        P(f"  {label:30s} " + " ".join(f"{nrm[j].item():9.4f} {cs[j].item():6.3f}" for j in range(len(layers))))

    for cond in PER_LAMBDA:
        row(cond, [find_vec(vecs, cond)])
    for cond in NOISY:
        for eps in epsilons:
            group = [v for v in vecs if v['condicion'] == cond and v['epsilon'] == eps]
            if group:
                row(f"{cond} ε={eps:g} (media {len(group)} sem.)", group)
    P("  (cos = coseno con Mean Steering (paper) en la misma capa; ‖v‖ antes de λ; diagnóstico)")


def build_resumen(res: pd.DataFrame, protocol: dict, *, origen: str, cond_dirs: dict, checks=(), state=None,
                  official_dir=None, official_prefix=None, ref_info: str | None = None, elapsed=None) -> str:
    P = Report()
    e = protocol['entradas']
    ta = protocol['template_args']
    P("=" * 110)
    P(f"RESUMEN — nivel 2 (TOFU forget10, métricas de open-unlearning) — {protocol['modelo']} — {origen}")
    P("=" * 110)
    P(f"protocolo: fp32 + sdpa | batch {protocol['batch_size']} | system {ta['system_prompt']!r} | fecha "
      f"{ta['date_string']!r} | OU {str(protocol['ou_commit'])[:10]} | TOFU {str(protocol['tofu']['revision'])[:10]} | "
      f"GPU {protocol['dispositivo']['gpu']}")
    P(f"steering: vectores de {OWN_DATASET}, ventana {e['ventana']} (capas {e['capas']}), C_eval {e['C_eval']:g}, "
      f"Cg_eval {e['Cg_eval']:.4f}, todos los tokens (min_token_index=0), también en la generación de ROUGE")
    P("PSA: d/max(C,‖d‖) por capa, media + gaussiano por capa a (ε/5, δ/5), sin ×C. PSA-1G: D/max(Cg,‖D‖) sobre las 5 capas "
      "concatenadas, media + UN gaussiano a (ε, δ), ×√5. Mean Steering (paper) = scaled_mean de run.py.")
    P("IC 95%: t de Student sobre las semillas (k−1 g.l.); PSA-1G − PSA: diferencia apareada por semilla (mismo z). "
      "forget_quality se resume en log10 (el valor crudo está en resultados.csv).")
    P("")
    P("AVISOS:")
    P(f"  - forget10 es IN-SAMPLE en las preguntas: los pares de {OWN_DATASET} (de donde salen los vectores) son las 400 "
      f"preguntas de forget10. retain, real authors, world facts y holdout10 no lo son.")
    P("  - referencia PROPIA: retain90 evaluado con este mismo protocolo (fp32 + sdpa, fecha "
      f"{ta['date_string']}, batch {protocol['batch_size']}, misma GPU). Las logs oficiales de open-unlearning "
      "(bf16 + flash_attention_2, 10 Apr 2025) son solo un control.")
    P("  - máscara de atención de open-unlearning: attention_mask = input_ids != pad (<|eot_id|>); los <|eot_id|> que "
      "cierran el system y el user quedan enmascarados (el nivel 3 no enmascara nada).")
    P("  - en la fila de la referencia forget_quality y privleak no aplican (open-unlearning calcula privleak contra 0.5; "
      "ese valor está solo en el log y en su config.json).")
    if ref_info:
        P(f"  - referencia usada: {ref_info}")
    totals = Counter()
    for d in cond_dirs.values():
        p = Path(d) / 'config.json'
        if p.is_file():
            totals.update(json.loads(p.read_text(encoding='utf-8')).get('avisos_ou') or {})
    for k, c in sorted(totals.items()):
        if c:
            P(f"  - open-unlearning avisó {c} × {k} en total (ver los logs de cada condición)")
    P("")
    if checks:
        P("CHEQUEOS:")
        for x in checks:
            P(f"  - {x.strip()}")
        P("")
    P("CONTROL contra las logs oficiales de open-unlearning (solo AVISO; difieren precisión y fecha, se esperan "
      f"diferencias; umbrales |Δ| > {TOL_UNIT:g} en [0,1], > {TOL_LOG10FQ:g} en log10(FQ), > {TOL_PRIVLEAK:g} pts en "
      "privleak; forget_quality y privleak oficiales usan la referencia oficial):")
    lines, n_av = official_control(res, official_dir, official_prefix)
    for x in lines:
        P(x)
    if n_av:
        P(f"  <-- AVISO: {n_av} diferencia(s) por encima del umbral")
    lams = sorted({float(x) for x in res.loc[~res['condicion'].isin(UNSTEERED), 'lambda']})
    epsilons = sorted({float(x) for x in res.loc[res['condicion'].isin(NOISY), 'epsilon']})
    n_seeds = len({int(x) for x in res.loc[res['condicion'].isin(NOISY), 'semilla']})
    for lam in (lams or [None]):
        P("")
        P("=" * 110)
        P(f"λ = {lam:+g}" if lam is not None else "sin condiciones con steering en estos resultados")
        for m in MAIN_METRICS + OTHER_METRICS:
            stats_block(P, res, m, lam, epsilons, n_seeds)
        components_table(P, res, lam, epsilons)
    if state is not None:
        P("")
        P("=" * 110)
        P(f"VECTORES ({OWN_DATASET}, ventana {state['ventana']})")
        P("  σ por coordenada:")
        sigma_lines(P, state['acc'], state['h'], state['epsilons'])
        fp, fg = state['clipfrac']['frac_psa'], state['clipfrac']['frac_g1']
        P(f"  Recorte: PSA (C={state['sel']['C']:g}) fracción de filas recortadas por capa "
          + " ".join(f"L{l} {fp[j]:.3f}" for j, l in enumerate(state['layers']))
          + f" | PSA-1G (Cg={state['sel']['Cg']:.3f}) fracción de filas concatenadas recortadas {fg:.3f}")
        vector_lines_l2(P, state['vectors'], state['layers'], state['epsilons'])
    P("")
    P("=" * 110)
    P("TIEMPOS (segundos)")
    for cond in CONDITION_ORDER:
        s = res.loc[res['condicion'] == cond, 'segundos']
        if len(s):
            P(f"  {cond:28s} media {s.mean():9.1f} por evaluación completa | n={len(s)}")
    per_metric = {}
    for d in cond_dirs.values():
        p = Path(d) / 'tiempos.json'
        if p.is_file():
            for rec in json.loads(p.read_text(encoding='utf-8'))['registros']:
                if rec['nivel'] == 0:
                    per_metric.setdefault(rec['metrica'], []).append(rec['segundos'])
    if per_metric:
        P("  por métrica de primer nivel (media sobre las condiciones):")
        for m in TOP_METRICS:
            if m in per_metric:
                P(f"    {m:24s} {np.mean(per_metric[m]):9.1f} | n={len(per_metric[m])}")
    if elapsed is not None:
        P(f"  tiempo de esta ejecución {elapsed:.0f} s")
    return P.text()


# --------------------------------------------------------------------------------------
# One job
# --------------------------------------------------------------------------------------
def run(args, spec: Spec) -> dict:
    t_start = time.time()
    setup_logging()
    out_dir, ou_dir = Path(args.out_dir), Path(args.ou_dir)
    log("=" * 100)
    log(f"NIVEL 2 — TOFU forget10 con las métricas de open-unlearning — {spec.full}")
    log(f"condiciones {args.conds} | λ {args.lams} | ε {args.epsilons} | semillas {args.seeds} | batch {args.batch_size}")
    log(f"salida {out_dir} | OU {ou_dir} | referencia {args.reference or '(la de esta carpeta)'} | logs oficiales "
        f"{args.official_logs}")
    log(f"AVISO: forget10 es IN-SAMPLE en las preguntas (los pares de {OWN_DATASET} son las preguntas de forget10)")
    checks: list[str] = []

    # ---------------- environment, OU, vectors (no model) ----------------
    checks.append(check_environment())
    checks.append(check_ou_checkout(ou_dir))
    import_ou(ou_dir)
    INSTRUMENT.install()
    paths = {'diffs_dir': args.diffs_dir, 'c_selection': args.c_selection, 'rbar_csv': args.rbar_csv}
    state = preflight_vectors(spec, paths, args.epsilons, args.seeds, checks)
    entradas = inputs_block(spec, paths, state)
    if args.level3_config:
        checks.append(check_level3_config(Path(args.level3_config), entradas))
    tasks = build_tasks(args.conds, args.lams, args.epsilons, args.seeds, state['vectors'])

    # ---------------- OU config, tokenizers, prompts ----------------
    cfg, gen = compose_eval_cfg(ou_dir, spec, args.batch_size)
    ta = make_template_args(cfg)
    yaml_text = eval_yaml(cfg)
    checks.append(f"suite de open-unlearning: {TOP_METRICS}; 23 nodos de TOFU "
                  + (f"fijados en {TOFU_REVISION[:10]}" if spec.real else "(datos locales del selftest)")
                  + f"; batch {args.batch_size}; generación {gen}")
    from transformers import AutoConfig, AutoTokenizer
    tok_full, tok_ret = AutoTokenizer.from_pretrained(spec.full), AutoTokenizer.from_pretrained(spec.retain)
    ct_full, ct_ret = check_tokenizer(tok_full, spec, 'full'), check_tokenizer(tok_ret, spec, 'retain90')
    if ct_full != ct_ret:
        raise SystemExit("full and retain90 tokenizers have different chat templates. Stopping.")
    log("=" * 100)
    log(f"CHEQUEO DE PROMPTS — ítem 0 de cada dataset de la suite, con las clases de open-unlearning y template_args {ta}")
    fp_full, items = prompt_check(cfg, tok_full, ta, 'full', verbose=True)
    fp_ret, _ = prompt_check(cfg, tok_ret, ta, 'retain90', verbose=False)
    if fp_full != fp_ret:
        raise SystemExit("full and retain90 build different token ids for the same prompts. Stopping.")
    checks.append(f"prompts: {len(items)} configuraciones de dataset × 2 tokenizers, un BOS, una línea 'Today Date: "
                  f"{ta['date_string']}', system {OU_SYSTEM_PROMPT!r}; full y retain90 con los mismos ids "
                  f"(huella {fp_full[:16]}…)")
    full_sha = AutoConfig.from_pretrained(spec.full)._commit_hash
    ret_sha = AutoConfig.from_pretrained(spec.retain)._commit_hash
    for name, got, want in ((spec.full, full_sha, spec.full_sha), (spec.retain, ret_sha, spec.retain_sha)):
        if want is not None and got != want:
            raise SystemExit(f"{name}: snapshot {got!r}, expected {want!r}. Stopping.")
    if spec.real:
        from utils import make_pos_neg_pair
        c = state['cache']
        if make_pos_neg_pair(c['train_rows'][0], tok_full) != (c['example_pos'], c['example_neg']):
            raise SystemExit(f"[{OWN_DATASET}] the first training prompt rebuilt now differs from the one stored in the "
                             f"cache: the extraction format changed. Stopping.")
        checks.append("primer prompt de entrenamiento reconstruido == example_pos/neg del caché")

    # ---------------- protocol, job directory ----------------
    protocol = canon({
        'script': 'eval_level2.py', 'modelo_tag': spec.tag, 'modelo': spec.full, 'modelo_sha': full_sha,
        'retain90': spec.retain, 'retain90_sha': ret_sha, 'ou_commit': OU_COMMIT,
        'tofu': {'path': TOFU_PATH if spec.real else 'json local (selftest)', 'revision': TOFU_REVISION if spec.real else None},
        'metricas': TOP_METRICS, 'eval_yaml_sha256': sha256_text(yaml_text), 'template_args': ta,
        'batch_size': args.batch_size, 'generation_args': gen,
        'precision': {'dtype': 'float32', 'attn_implementation': 'sdpa', 'tf32_matmul': False,
                      'float32_matmul_precision': 'highest'},
        'dispositivo': device_info(), 'versiones': versions(),
        'prompts': {'huella_sha256': fp_full, 'chat_template_sha256': ct_full, 'bos': spec.bos_id, 'eos': spec.eos_id,
                    'pad': spec.eos_id, 'n_items': len(items)},
        'entradas': entradas, 'selftest': not spec.real})
    job = canon({'condiciones': args.conds, 'lams': args.lams, 'epsilons': args.epsilons, 'semillas': args.seeds,
                 'referencia': args.reference, 'official_logs': args.official_logs, 'level3_config': args.level3_config,
                 'ou_dir': str(ou_dir), 'out_dir': str(out_dir)})
    resumed = check_or_write_job_config(out_dir, protocol, job)
    if not (out_dir / 'eval_config.yaml').is_file():
        write_text_atomic(out_dir / 'eval_config.yaml', yaml_text)
    record_launch(out_dir, job)
    store = ResultsStore(out_dir / 'resultados.csv', spec.tag)
    pending = [t for t in tasks
               if task_key(spec.tag, t['condicion'], t['epsilon'], t['semilla'], t['lambda']) not in store]
    log("=" * 100)
    log(f"{len(tasks)} condiciones pedidas, {len(tasks) - len(pending)} ya en resultados.csv, {len(pending)} por calcular"
        + (" (reanudación)" if resumed else ""))
    for t in pending:
        log(f"  - {task_label(t)} → {task_rel_dir(t['condicion'], t['epsilon'], t['semilla'], t['lambda'])}")
    ctx = {'spec': spec, 'state': state, 'out_dir': out_dir, 'protocol': protocol, 'store': store, 'ta': ta}

    # ---------------- reference ----------------
    own_ref = out_dir / 'referencia_retain90'
    ref_dir = Path(args.reference) if args.reference else own_ref
    ref_task = [t for t in pending if t['condicion'] == REF]
    others = [t for t in pending if t['condicion'] != REF]
    ref = None
    if others and not ref_task:
        ref = resolve_reference(ref_dir, protocol, spec.tag, spec.forget_n)
    n_new = 0
    if ref_task:
        model, tok = load_model_checked(spec.retain, ret_sha if spec.retain_sha else None, protocol, state, 'retain90')
        base_hooks = n_forward_hooks(model)
        evaluator = make_evaluator(cfg, None)
        run_condition(ctx, ref_task[0], evaluator, model, tok, None, None, base_hooks)
        n_new += 1
        del model, tok, evaluator
        release_memory()
        ref = resolve_reference(own_ref, protocol, spec.tag, spec.forget_n)

    # ---------------- full model ----------------
    if others:
        model, tok = load_model_checked(spec.full, full_sha if spec.full_sha else None, protocol, state, 'full')
        base_hooks = n_forward_hooks(model)
        dtype = next(model.parameters()).dtype
        dev = next(model.parameters()).device
        steered = [t for t in others if t['vec'] is not None]
        if steered:
            if spec.real:
                checks.extend(check4_tofu(model, tok, state, base_hooks))
            t0 = steered[0]
            ids = next(it['ids'] for it in items if not it['gen'])
            checks.append(check_layer_jumps(model, ids, to_steering_vector(t0['vec'], state['layers'], dtype, dev),
                                            t0['lambda'], t0['vec'], state['layers'], base_hooks))
        evaluator = make_evaluator(cfg, ref['eval_json'])
        done_secs = []
        for i, t in enumerate(others):
            sv = None if t['vec'] is None else to_steering_vector(t['vec'], state['layers'], dtype, dev)
            row = run_condition(ctx, t, evaluator, model, tok, sv, ref, base_hooks)
            n_new += 1
            done_secs.append(row['segundos'])
            left = len(others) - i - 1
            log(f"  [{i + 1}/{len(others)}] quedan {left} condiciones ≈ {left * float(np.mean(done_secs)) / 3600:.1f} h")
        checks.append(f"ningún hook de steering quedó en el modelo después de cada condición ({len(others)} en esta "
                      f"ejecución)")
        del model, tok, evaluator
        release_memory()

    # ---------------- summary ----------------
    res = read_results(store.path)
    cond_dirs = {k: out_dir / task_rel_dir_from_key(k) for k in store.rows}
    if ref is None and (out_dir / 'referencia_retain90' / 'config.json').is_file():
        ref_info = str(out_dir / 'referencia_retain90')
    else:
        ref_info = None if ref is None else f"{ref['dir']} (TOFU_EVAL.json sha256 {ref['sha256'][:16]}…)"
    text = build_resumen(res, protocol, origen=f"job {out_dir}", cond_dirs=cond_dirs, checks=checks, state=state,
                         official_dir=args.official_logs, official_prefix=spec.official_prefix, ref_info=ref_info,
                         elapsed=time.time() - t_start)
    write_text_atomic(out_dir / 'resumen.txt', text)
    log(f"\n{n_new} condiciones nuevas | resultados.csv ({len(store)} filas), resumen.txt y config.json en {out_dir} | "
        f"{time.time() - t_start:.0f} s")
    return {'out_dir': out_dir, 'n_new': n_new, 'protocol': protocol, 'state': state, 'items': items}


# --------------------------------------------------------------------------------------
# Self-test (CPU, tiny random Llama, local TOFU-like data; nothing downloaded, temp dir only)
# --------------------------------------------------------------------------------------
ST_N = {'forget10_perturbed': 12, 'holdout10': 12, 'retain_perturbed': 12, 'real_authors_perturbed': 8,
        'world_facts_perturbed': 8}
ST_WORDS = ("the a an author book wrote born city river garden mother father novel poem award year early life works style "
            "genre writes story famous young old house light dark music painter teacher doctor small large green blue red "
            "quiet loud friend family village mountain winter summer letter school").split()
ST_BATCH = 4
ST_GEN_TOKENS = 8
ST_LAMS = ['1', '3']
ST_EPS = ['1', '8']
ST_SEEDS = ['0', '1']
ST_CHAT_TEMPLATE = (
    r"""{{- bos_token }}
{%- if not date_string is defined %}{%- if strftime_now is defined %}{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}{%- set date_string = "26 Jul 2024" %}{%- endif %}{%- endif %}
{%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}{%- else %}{%- set system_message = "" %}{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}""")
# leaf metrics of the suite that run the model (the selftest checks the steering hook fires inside every one of them)
MODEL_LEAVES = ['forget_Q_A_PARA_Prob', 'forget_Q_A_PERT_Prob', 'forget_Q_A_Prob', 'forget_Q_A_ROUGE', 'retain_Q_A_Prob',
                'retain_Q_A_ROUGE', 'retain_Q_A_PARA_Prob', 'retain_Q_A_PERT_Prob', 'ra_Q_A_Prob', 'ra_Q_A_PERT_Prob',
                'ra_Q_A_ROUGE', 'wf_Q_A_Prob', 'wf_Q_A_PERT_Prob', 'wf_Q_A_ROUGE', 'mia_min_k', 'extraction_strength']
ROUGE_LEAVES = {'forget_Q_A_ROUGE': 'forget10_perturbed', 'retain_Q_A_ROUGE': 'retain_perturbed',
                'ra_Q_A_ROUGE': 'real_authors_perturbed', 'wf_Q_A_ROUGE': 'world_facts_perturbed'}


def st_sentence(rng, n: int, end: str = '.') -> str:
    return ' '.join(ST_WORDS[i] for i in rng.integers(0, len(ST_WORDS), n)).capitalize() + end


def st_write_tofu(root: Path) -> dict:
    root.mkdir(parents=True)
    rng = np.random.default_rng(7)
    files = {}
    for name, n in ST_N.items():
        rows = []
        for i in range(n):
            q = st_sentence(rng, 6 + i % 4, '?')
            if name in ('forget10_perturbed', 'retain_perturbed'):
                r = {'question': q, 'answer': st_sentence(rng, 8), 'paraphrased_answer': st_sentence(rng, 9),
                     'perturbed_answer': [st_sentence(rng, 8) for _ in range(5)], 'paraphrased_question': st_sentence(rng, 7, '?')}
            elif name in ('real_authors_perturbed', 'world_facts_perturbed'):
                r = {'question': q, 'answer': st_sentence(rng, 2), 'perturbed_answer': [st_sentence(rng, 2) for _ in range(3)]}
            else:
                r = {'question': q, 'answer': st_sentence(rng, 10)}
            rows.append(r)
        p = root / f"{name}.jsonl"
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        files[name] = str(p.resolve())
    return files


def st_tokenizer(texts: list):
    """Byte-level BPE trained on the selftest texts, Llama-3 special tokens, a Llama-3.2-like chat template (same system
    header, 'Cutting Knowledge Date' and 'Today Date' lines, strftime_now fallback when no date_string is passed)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    specials = ['<|begin_of_text|>', '<|end_of_text|>', '<|start_header_id|>', '<|end_header_id|>', EOT]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=1000, special_tokens=specials, show_progress=False,
                                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    bos = tk.token_to_id('<|begin_of_text|>')
    tk.post_processor = processors.TemplateProcessing(single='<|begin_of_text|> $A',
                                                      special_tokens=[('<|begin_of_text|>', bos)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token='<|begin_of_text|>', eos_token=EOT, pad_token=EOT,
                                  model_max_length=4096, clean_up_tokenization_spaces=False)
    tok.chat_template = ST_CHAT_TEMPLATE
    return tok


def st_assets(root: Path) -> dict:
    from transformers import LlamaConfig, LlamaForCausalLM
    files = st_write_tofu(root / 'tofu')
    texts = ['Cutting Knowledge Date: December 2023', 'Today Date: 26 Jul 2024', OU_SYSTEM_PROMPT,
             'You are a helpful, honest assistant.', 'system user assistant', '0123456789 Jan Feb Mar Apr May Jun Jul Aug '
             'Sep Oct Nov Dec']
    for p in files.values():
        for line in Path(p).read_text(encoding='utf-8').splitlines():
            for v in json.loads(line).values():
                texts += v if isinstance(v, list) else [v]
    tok = st_tokenizer(texts)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=eval_level3.ST_H, intermediate_size=128,
                      num_hidden_layers=eval_level3.ST_L, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id, tie_word_embeddings=False)
    dirs = {}
    for name, seed in (('full', 1), ('retain90', 2)):
        torch.manual_seed(seed)
        m = LlamaForCausalLM(cfg).eval()
        d = root / name
        m.save_pretrained(d)
        tok.save_pretrained(d)
        dirs[name] = str(d.resolve())
    return {'files': files, 'tok': tok, 'full': dirs['full'], 'retain': dirs['retain90']}


def st_spec(assets: dict, cache_dir: Path) -> Spec:
    tok = assets['tok']
    return Spec(tag='selftest', cache_model=eval_level3.ST_MODEL, full=assets['full'], retain=assets['retain'],
                ou_model_cfg='Llama-3.2-1B-Instruct', full_sha=None, retain_sha=None,
                forget_n=ST_N['forget10_perturbed'], official_prefix='tofu_selftest', bos_id=tok.bos_token_id,
                eos_id=tok.eos_token_id, selftest_data=assets['files'], selftest_cache=str(cache_dir),
                gen_max_new_tokens=ST_GEN_TOKENS)


def st_rows(out_dir: Path, tag: str = 'selftest') -> dict:
    return parse_results_text((out_dir / 'resultados.csv').read_text(encoding='utf-8'), tag, str(out_dir))


def st_no_seconds(rows: dict) -> dict:
    return {k: {c: v for c, v in r.items() if c != 'segundos'} for k, r in rows.items()}


def st_fake_protocol() -> dict:
    """A minimal protocol dict with the fields build_resumen reads (merge_level2's selftest)."""
    return canon({'script': 'eval_level2.py', 'modelo_tag': 'selftest', 'modelo': 'synthetic/full', 'ou_commit': OU_COMMIT,
                  'tofu': {'path': 'x', 'revision': None}, 'batch_size': 4,
                  'template_args': {'system_prompt': OU_SYSTEM_PROMPT, 'date_string': '26 Jul 2024'},
                  'dispositivo': {'gpu': 'cpu'},
                  'entradas': {'ventana': '12-16', 'capas': [12, 13, 14, 15, 16], 'C_eval': 2.0, 'Cg_eval': 2 * math.sqrt(5)}})


def selftest(args) -> None:
    t0 = time.time()
    setup_logging()
    if not Path('datasets').is_dir():
        raise SystemExit("run the selftest from iclr2025_psa (it reads datasets/ like the other selftests)")
    import utils
    utils.device = 'cpu'           # the tiny model runs on CPU even on a GPU node (get_model_and_tokenizer reads it)
    ou_dir = Path(args.ou_dir)
    tmp = Path(tempfile.mkdtemp(prefix='eval_level2_selftest_'))
    parser = build_parser()

    def st_args(argv: list):
        return resolve_args(parser.parse_args(argv))

    try:
        # ------------------------------------------------------------------ 0
        log("#" * 100)
        log("SELFTEST 0 — nombres, claves, flags, plan de condiciones, resultados.csv, IC con t de tabla")
        assert task_rel_dir(REF, None, None, 0.0) == 'referencia_retain90' and task_rel_dir(NONE, None, None, 0.0) == 'sin_steering'
        assert task_rel_dir(PSA0, None, None, 1.0) == 'lam+1/psa_sin_ruido'
        assert task_rel_dir(PSA, 1.0, 0, 1.0) == 'lam+1/psa/eps1_s0'
        assert task_rel_dir(G1, 8.0, 3, -0.5) == 'lam-0.5/psa1g/eps8_s3'
        k = task_key('selftest', G1, 8.0, 3, -0.5)
        assert task_rel_dir_from_key(k) == 'lam-0.5/psa1g/eps8_s3' and k == ('selftest', G1, '8.0', '3', '-0.5')
        need = ['--diffs-dir', 'x', '--c-selection', 'x', '--rbar-csv', 'x', '--out-dir', 'x']
        assert st_args(need).conds == CONDITION_ORDER
        assert st_args(need + ['--only-noiseless']).conds == NOISELESS
        assert st_args(need + ['--only-reference']).conds == [REF]
        assert st_args(need + ['--conditions', 'psa0', 'sin-steering', 'referencia']).conds == [REF, NONE, PSA0]
        a = st_args(['--model-tag', '3b'])
        assert (a.diffs_dir, a.c_selection, a.rbar_csv, a.out_dir) == ('sweep_3b/diffs', 'sweep_c_3b/c_selection.csv',
                                                                        'resultados_rbar_3b/rbar_results.csv', 'nivel2_3b')
        assert a.official_logs == str(Path('../open-unlearning') / 'saves' / 'eval') and a.lams == [1.0]
        assert a.seeds == [0, 1, 2, 3, 4] and a.epsilons == [1.0, 4.0, 8.0] and a.batch_size == 32
        for label, extra in (('--conditions con --only-noiseless', ['--conditions', 'psa', '--only-noiseless']),
                             ('λ = 0', ['--lams', '0']), ('λ repetido', ['--lams', '1', '1']),
                             ('λ con el mismo directorio', ['--lams', '1', '1.0000001']), ('ε = 2', ['--epsilons', '2']),
                             ('semilla negativa', ['--seeds', '-1']), ('batch 0', ['--batch-size', '0']),
                             ('--reference con la referencia', ['--reference', 'r']),
                             ('condición repetida', ['--conditions', 'psa', 'psa'])):
            expect_exit(lambda: st_args(need + extra), label)
        expect_exit(lambda: st_args([]), 'sin --model-tag ni rutas')
        fake = [{'condicion': c, 'epsilon': float('nan'), 'semilla': float('nan'), 'vec': torch.zeros(1)} for c in PER_LAMBDA]
        fake += [{'condicion': c, 'epsilon': e, 'semilla': s, 'vec': torch.zeros(1)} for c in NOISY for e in DEFAULT_EPSILONS
                 for s in range(10)]
        assert len(build_tasks(CONDITION_ORDER, [1.0], DEFAULT_EPSILONS, DEFAULT_SEEDS, fake)) == 35
        assert len(build_tasks(CONDITION_ORDER, [1.0, 2.0], DEFAULT_EPSILONS, DEFAULT_SEEDS, fake)) == 68
        first = build_tasks([REF, NONE, PSA0], [1.0, 2.0], DEFAULT_EPSILONS, DEFAULT_SEEDS, fake)
        assert [(t['condicion'], t['lambda']) for t in first] == [(REF, 0.0), (NONE, 0.0), (PSA0, 1.0), (PSA0, 2.0)]
        assert len(build_tasks(NOISELESS, [1.0, 2.0], DEFAULT_EPSILONS, DEFAULT_SEEDS, fake)) == 2 + 3 * 2
        keys = [(c, e, s, l) for (c, e, s, l) in [(G1, 8.0, 1, 2.0), (PSA, 8.0, 1, 2.0), (MEAN, '', '', 2.0), (G1_0, '', '', 1.0),
                                                  (NONE, '', '', 0.0), (PSA, 1.0, 0, 1.0), (REF, '', '', 0.0)]]
        order = sorted(keys, key=lambda x: row_sort_key(*x))
        assert [x[0] for x in order] == [REF, NONE, G1_0, PSA, MEAN, PSA, G1], order
        # resultados.csv: round trip and corrupt files
        csv_dir = tmp / 'csv'
        csv_dir.mkdir()
        sha = 'a' * 64
        base_row = {'modelo': 'selftest', 'ventana': '12-16', 'epsilon': float('nan'), 'semilla': float('nan'),
                    'vector_sha256': '', 'referencia_sha256': sha, 'segundos': 1.25, **{m: 0.5 for m in METRIC_COLUMNS}}
        st = ResultsStore(csv_dir / 'resultados.csv', 'selftest')
        st.append({**base_row, 'condicion': REF, 'lambda': 0.0, 'forget_quality': None, 'privleak': None})
        st.append({**base_row, 'condicion': NONE, 'lambda': 0.0, 'forget_quality': 1e-22})
        st.append({**base_row, 'condicion': PSA, 'lambda': 1.0, 'epsilon': 1.0, 'semilla': 0, 'vector_sha256': 'b' * 64})
        again = ResultsStore(csv_dir / 'resultados.csv', 'selftest')
        assert set(again.rows) == set(st.rows) and len(again) == 3
        assert task_key('selftest', PSA, 1.0, 0, 1.0) in again and task_key('selftest', REF, float('nan'), float('nan'), 0.0) in again
        expect_exit(lambda: st.append({**base_row, 'condicion': NONE, 'lambda': 0.0}), 'fila repetida')
        expect_exit(lambda: st.append({**base_row, 'condicion': REF, 'lambda': 0.0}), 'referencia con forget_quality')
        expect_exit(lambda: st.append({**base_row, 'condicion': MEAN, 'lambda': 1.0}), 'Mean sin vector_sha256')
        expect_exit(lambda: st.append({**base_row, 'condicion': PSA, 'lambda': 1.0, 'epsilon': 2.0, 'semilla': 0,
                                       'vector_sha256': 'b' * 64}), 'ε fuera de 1/4/8')
        text = (csv_dir / 'resultados.csv').read_text(encoding='utf-8')
        for label, bad in (('última línea incompleta', text[:-1]), ('fila duplicada', text + text.splitlines(True)[2]),
                           ('otro modelo', text.replace('\nselftest,', '\notro,', 1)),
                           ('número inválido', text.replace(',0.5,', ',abc,', 1))):
            expect_exit(lambda: parse_results_text(bad, 'selftest', label), label)
        (csv_dir / 'resultados.csv').write_text(text + text.splitlines(True)[1], encoding='utf-8')
        expect_exit(lambda: st.append({**base_row, 'condicion': MEAN, 'lambda': 2.0, 'vector_sha256': 'c' * 64}),
                    'archivo modificado por otro proceso')
        t975_4 = 2.776445105                              # t_{0.975, 4} from a table (9 decimals), not from scipy
        m, sd, hw = ci95(np.arange(1.0, 6.0))
        assert abs(m - 3.0) < 1e-12 and abs(sd - math.sqrt(2.5)) < 1e-12 and abs(hw - t975_4 * math.sqrt(2.5) / math.sqrt(5)) < 1e-8
        cell = paired_cells('model_utility', {0: 0.5, 1: 0.6, 2: 0.7}, {0: 0.55, 1: 0.6, 2: 0.9}, 3)
        assert cell.strip().endswith('2/3'), cell
        assert paired_cells('forget_quality', {0: 1e-3, 1: 0.0}, {0: 1e-2, 1: 1e-1}, 2).endswith('(1/2 semillas apareadas; 1 valor(es) = 0 excluidos)')
        assert abs(tr_metric('forget_quality', 1e-22) + 22.0) < 1e-12 and tr_metric('forget_quality', 0.0) is None
        log("  OK")

        # ------------------------------------------------------------------ A
        log("#" * 100)
        log("SELFTEST A — open-unlearning: checkout, import, composición real de la suite para 1b / 3b / 8b")
        check_environment()
        check_ou_checkout(ou_dir)
        import_ou(ou_dir)
        INSTRUMENT.install()
        for tag in ('1b', '3b', '8b'):
            cfg, gen = compose_eval_cfg(ou_dir, real_spec(tag), 32)
            assert gen == EXPECTED_GENERATION
            ta = make_template_args(cfg)
            assert ta['date_string'] == utils.CHAT_DATE and ta['system_prompt'] == OU_SYSTEM_PROMPT
            hf = ou_nodes(cfg.eval.tofu, 'hf_args')
            assert len(hf) == 23 and all(n.hf_args.revision == TOFU_REVISION for n, _ in hf)
            assert len(suite_dataset_nodes(cfg)) >= 10
            cfg16, _ = compose_eval_cfg(ou_dir, real_spec(tag), 16)
            assert eval_yaml(cfg) != eval_yaml(cfg16) and eval_yaml(cfg) == eval_yaml(compose_eval_cfg(ou_dir, real_spec(tag), 32)[0])
            assert 'RETAIN_LOGS_PATH' in eval_yaml(cfg)
            ev = make_evaluator(cfg, tmp / 'x' / 'TOFU_EVAL.json')
            assert list(ev.metrics) == TOP_METRICS
            make_evaluator(cfg, None)
            log(f"  {tag}: {OU_MODEL_CFG[tag]}, 7 métricas, 23 nodos de TOFU en {TOFU_REVISION[:10]}, generación {gen}, "
                f"{len(suite_dataset_nodes(cfg))} configuraciones de dataset distintas")
        log("  OK")

        # ------------------------------------------------------------------ B
        log("#" * 100)
        log("SELFTEST B — datos y modelos diminutos, preflight, chequeo de prompts (casos buenos y malos)")
        inputs = eval_level3.st_write_inputs(tmp / 'inputs')
        assets = st_assets(tmp / 'assets')
        spec = st_spec(assets, tmp / 'hf_datasets')
        tok = assets['tok']
        cfg, gen = compose_eval_cfg(ou_dir, spec, ST_BATCH)
        assert gen['max_new_tokens'] == ST_GEN_TOKENS
        ta = make_template_args(cfg)
        fp, items = prompt_check(cfg, tok, ta, 'selftest', verbose=True)
        from transformers import AutoTokenizer
        fp_ret, _ = prompt_check(cfg, AutoTokenizer.from_pretrained(spec.retain), ta, 'selftest retain', verbose=False)
        assert fp == fp_ret, "same tokenizer saved twice must give the same fingerprint"
        expect_exit(lambda: prompt_check(cfg, tok, {k: v for k, v in ta.items() if k != 'date_string'}, 'sin fecha', False),
                    'template_args sin date_string (el template pone la fecha de hoy)')
        expect_exit(lambda: prompt_check(cfg, tok, {**ta, 'date_string': '10 Apr 2025'}, 'otra fecha', False), 'fecha 10 Apr 2025')
        expect_exit(lambda: prompt_check(cfg, tok, {**ta, 'system_prompt': 'You are a helpful, honest assistant.'}, 'honest', False),
                    'system prompt del nivel 3')
        try:
            tok.chat_template = ST_CHAT_TEMPLATE.replace("{{- bos_token }}", "{{- bos_token }}{{- bos_token }}", 1)
            expect_exit(lambda: prompt_check(cfg, tok, ta, 'doble BOS', False), 'doble BOS')
            tok.chat_template = ST_CHAT_TEMPLATE.replace('December 2023\\n', 'December 2023 \\n', 1)
            fp2, _ = prompt_check(cfg, tok, ta, 'template distinto', False)
            assert fp2 != fp, "a different chat template must change the fingerprint"
        finally:
            tok.chat_template = ST_CHAT_TEMPLATE
        st_checks = []
        state = preflight_vectors(spec, inputs, [1.0, 8.0], [0, 1], st_checks)
        assert state['ventana'] == eval_level3.ST_WINDOWS[OWN_DATASET] and state['h'] == eval_level3.ST_H
        Z10 = paired_draws(list(range(10)), S, state['h'])
        v10, _ = build_vectors(state['cache']['d32'], state['sel']['C'], state['sel']['Cg'], state['acc'], Z10, [1.0, 8.0],
                               list(range(10)))
        for cond in NOISY:
            for eps in (1.0, 8.0):
                for s in (0, 1):
                    assert torch.equal(find_vec(v10, cond, eps, s)['vec'], find_vec(state['vectors'], cond, eps, s)['vec'])
        for cond in PER_LAMBDA:
            assert torch.equal(find_vec(v10, cond)['vec'], find_vec(state['vectors'], cond)['vec'])
        log("  OK: prompts verificados; vectores con semillas 0,1 == los mismos con semillas 0..9 (como el nivel 3)")

        # ------------------------------------------------------------------ C
        log("#" * 100)
        log("SELFTEST C — job A completo: referencia, sin steering, Mean / PSA s/r / PSA-1G s/r, PSA / PSA-1G (ε 1, 8 × "
            "semillas 0, 1) con λ +1 y +3")
        base_argv = ['--ou-dir', str(ou_dir), '--diffs-dir', inputs['diffs_dir'], '--c-selection', inputs['c_selection'],
                     '--rbar-csv', inputs['rbar_csv'], '--batch-size', str(ST_BATCH), '--epsilons', *ST_EPS,
                     '--seeds', *ST_SEEDS]
        A = tmp / 'job_A'
        out_a = run(st_args(base_argv + ['--lams', *ST_LAMS, '--out-dir', str(A)]), spec)
        rows_a = st_rows(A)
        n_expected = 2 + len(ST_LAMS) * (3 + len(NOISY) * len(ST_EPS) * len(ST_SEEDS))
        assert len(rows_a) == n_expected == out_a['n_new'], (len(rows_a), n_expected)
        kref = task_key('selftest', REF, float('nan'), float('nan'), 0.0)
        ref_sha = rows_a[kref]['referencia_sha256']
        assert rows_a[kref]['forget_quality'] == '' and rows_a[kref]['privleak'] == ''
        assert all(r['referencia_sha256'] == ref_sha for r in rows_a.values())
        assert ref_sha == sha256_file(A / 'referencia_retain90' / 'TOFU_EVAL.json')
        st_state = out_a['state']
        for kk, r in rows_a.items():
            cond = r['condicion']
            d = A / task_rel_dir_from_key(kk)
            assert all((d / f).is_file() for f in COND_FILES), d
            assert {x['metrica'] for x in json.loads((d / 'tiempos.json').read_text())['registros'] if x['nivel'] == 0} == set(TOP_METRICS)
            if cond in PER_LAMBDA:
                assert r['vector_sha256'] == vector_sha(find_vec(st_state['vectors'], cond)['vec'])
            elif cond in NOISY:
                assert r['vector_sha256'] == vector_sha(find_vec(st_state['vectors'], cond, float(r['epsilon']),
                                                                 int(r['semilla']))['vec'])
            if cond != REF:
                assert r['forget_quality'] != '' and r['privleak'] != ''
        res_txt = (A / 'resumen.txt').read_text(encoding='utf-8')
        for s in ('IN-SAMPLE', 'referencia PROPIA', 'log10(forget_quality)', 'λ = +1', 'λ = +3', 'componentes de model_utility',
                  'CONTROL', 'TIEMPOS', 'sonda de capas', 'PSA-1G − PSA [IC 95%]'):
            assert s in res_txt, s
        text_a = (A / 'resultados.csv').read_bytes()
        again = run(st_args(base_argv + ['--lams', *ST_LAMS, '--out-dir', str(A)]), spec)
        assert again['n_new'] == 0 and (A / 'resultados.csv').read_bytes() == text_a
        assert len((A / 'lanzamientos.jsonl').read_text(encoding='utf-8').splitlines()) == 2
        log(f"  OK: {len(rows_a)} filas; referencia sin forget_quality/privleak; sha de vectores == recomputados; "
            f"relanzar no evalúa nada y no toca resultados.csv")

        # ------------------------------------------------------------------ D
        log("#" * 100)
        log("SELFTEST D — λ = 0 idéntico a sin steering; instrumentación sin efecto; el hook actúa en todas las métricas "
            "(también en cada paso de la generación); sonda de capas; sin referencia aborta")
        protocol_a = out_a['protocol']
        ref = resolve_reference(A / 'referencia_retain90', protocol_a, 'selftest', spec.forget_n)
        cfg, _ = compose_eval_cfg(ou_dir, spec, ST_BATCH)
        ta = make_template_args(cfg)
        model, mtok = load_model_checked(spec.full, None, protocol_a, st_state, 'full (selftest D)')
        base_hooks = n_forward_hooks(model)
        ev = make_evaluator(cfg, ref['eval_json'])
        D = tmp / 'st_d'
        layers = st_state['layers']
        v0 = find_vec(st_state['vectors'], PSA0)['vec']
        sv0 = to_steering_vector(v0, layers, torch.float32, 'cpu')
        outs = {}
        for name, sv, lam in (('none', None, 0.0), ('zero', sv0, 0.0)):
            (D / name).mkdir(parents=True)
            outs[name] = evaluate_once(ev, model, mtok, ta, D / name, sv, lam, base_hooks, False, f"D {name}")
        b_none = (D / 'none' / 'TOFU_EVAL.json').read_bytes()
        assert (D / 'zero' / 'TOFU_EVAL.json').read_bytes() == b_none, "λ = 0 must be identical to no steering"
        assert (A / 'sin_steering' / 'TOFU_EVAL.json').read_bytes() == b_none, "same model + reference must reproduce job A"
        INSTRUMENT.uninstall()
        try:
            (D / 'raw').mkdir()
            evaluate_once(ev, model, mtok, ta, D / 'raw', None, 0.0, base_hooks, False, 'D sin instrumentación')
        finally:
            INSTRUMENT.install()
        assert (D / 'raw' / 'TOFU_EVAL.json').read_bytes() == b_none, "the timing wrapper must not change any result"
        counter = [0]
        probe = model.model.layers[layers[0]].register_forward_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
        INSTRUMENT.probe = lambda: counter[0]
        try:
            (D / 'steer').mkdir()
            steer = evaluate_once(ev, model, mtok, ta, D / 'steer', sv0, 3.0, base_hooks + 1, False, 'D λ=+3')
        finally:
            INSTRUMENT.probe = None
            probe.remove()
        rec = {}
        for x in steer['tiempos']:
            rec.setdefault(x['metrica'], x)
        for leaf in MODEL_LEAVES:
            assert leaf in rec and rec[leaf]['probe'] > 0, (leaf, rec.get(leaf))
        n_batches = {leaf: math.ceil(ST_N[dname] / ST_BATCH) for leaf, dname in ROUGE_LEAVES.items()}
        assert all(rec[leaf]['probe'] >= n for leaf, n in n_batches.items()), {leaf: rec[leaf] for leaf in n_batches}
        # more forward passes than prefills: the decode steps ran through the steered layer too
        assert sum(rec[leaf]['probe'] for leaf in n_batches) > sum(n_batches.values()), {leaf: rec[leaf] for leaf in n_batches}
        base_logs, st_logs = outs['none']['logs'], steer['logs']
        diff_prob = sum(base_logs['forget_Q_A_Prob']['value_by_index'][i]['prob'] != v['prob']
                        for i, v in st_logs['forget_Q_A_Prob']['value_by_index'].items())
        diff_mia = sum(base_logs['mia_min_k']['forget']['value_by_index'][i]['score'] != v['score']
                       for i, v in st_logs['mia_min_k']['forget']['value_by_index'].items())
        diff_gen = sum(base_logs[m]['value_by_index'][i]['generation'] != v['generation']
                       for m in ROUGE_LEAVES for i, v in st_logs[m]['value_by_index'].items())
        assert diff_prob > 0 and diff_mia > 0 and diff_gen > 0, (diff_prob, diff_mia, diff_gen)
        ids = next(it['ids'] for it in out_a['items'] if not it['gen'])
        check_layer_jumps(model, ids, sv0, 3.0, v0, layers, base_hooks)
        expect_exit(lambda: check_layer_jumps(model, ids, sv0, 3.0, v0 * 1.01, layers, base_hooks), 'sonda con otro vector')
        ev_none = make_evaluator(cfg, None)
        (D / 'noref').mkdir()
        expect_exit(lambda: evaluate_once(ev_none, model, mtok, ta, D / 'noref', None, 0.0, base_hooks, False, 'D sin ref'),
                    'condición sin referencia')
        assert n_forward_hooks(model) == base_hooks
        log(f"  OK: con λ=+3 cambian {diff_prob} probabilidades, {diff_mia} puntajes MIA y {diff_gen} generaciones; la capa "
            f"{layers[0]} corre dentro de las {len(MODEL_LEAVES)} métricas que usan el modelo")
        del model, mtok, ev, ev_none
        release_memory()

        # ------------------------------------------------------------------ E
        log("#" * 100)
        log("SELFTEST E — job B interrumpido a mitad de la 3.ª condición, relanzado, y comparado con el job A")
        per_eval = json.loads((A / 'sin_steering' / 'config.json').read_text(encoding='utf-8'))['llamadas_metricas']
        B = tmp / 'job_B'
        argv_b = base_argv[:-len(ST_EPS) - len(ST_SEEDS) - 2] + ['--epsilons', '1', '--seeds', '0', '--lams', '1',
                                                                    '--conditions', 'referencia', 'sin-steering', 'psa0',
                                                                    'psa', '--out-dir', str(B)]
        INSTRUMENT.arm_fault(2 * per_eval + per_eval // 2)
        try:
            run(st_args(argv_b), spec)
        except Interrupted as e:
            log(f"  (esperado) {e}")
        else:
            raise AssertionError("the fault injection was supposed to interrupt job B")
        finally:
            INSTRUMENT.fault_at = None
        assert len(st_rows(B)) == 2
        partial = B / 'lam+1' / 'psa_sin_ruido'
        assert (partial / 'TOFU_EVAL.json').is_file() and not (partial / 'config.json').is_file()
        out_b = run(st_args(argv_b), spec)
        assert out_b['n_new'] == 2 and out_b['protocol'] == protocol_a
        rows_b = st_rows(B)
        nb, na = st_no_seconds(rows_b), st_no_seconds(rows_a)
        assert set(nb) <= set(na) and all(nb[kk] == na[kk] for kk in nb), "job B must reproduce job A row by row"
        assert run(st_args(argv_b), spec)['n_new'] == 0
        log(f"  OK: {len(rows_b)} filas idénticas a las del job A (salvo segundos); el directorio incompleto se borró y "
            f"se recalculó; un tercer lanzamiento no evalúa nada")

        # ------------------------------------------------------------------ F
        log("#" * 100)
        log("SELFTEST F — abortos: otro batch, sin referencia, referencia de otra GPU / modificada, carpeta sin config.json")
        expect_exit(lambda: run(st_args([x if x != str(ST_BATCH) else '2' for x in argv_b]), spec), 'otro --batch-size')
        expect_exit(lambda: run(st_args(base_argv + ['--conditions', 'sin-steering', '--out-dir', str(tmp / 'job_C')]), spec),
                    'sin referencia')
        for label, tamper in (('referencia calculada en otra GPU', 'gpu'), ('TOFU_EVAL.json de la referencia modificado', 'sha')):
            cp = tmp / f"A_{tamper}"
            shutil.copytree(A, cp)
            if tamper == 'gpu':
                p = cp / 'referencia_retain90' / 'config.json'
                c = json.loads(p.read_text(encoding='utf-8'))
                c['protocolo']['dispositivo']['gpu'] = 'NVIDIA A100-SXM4-40GB'
                p.write_text(json.dumps(c), encoding='utf-8')
            else:
                p = cp / 'referencia_retain90' / 'TOFU_EVAL.json'
                p.write_text(p.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            expect_exit(lambda: run(st_args(base_argv + ['--conditions', 'sin-steering', '--reference',
                                                         str(cp / 'referencia_retain90'),
                                                         '--out-dir', str(tmp / f"job_{tamper}")]), spec), label)
        (tmp / 'job_vacio').mkdir()
        expect_exit(lambda: run(st_args(base_argv + ['--out-dir', str(tmp / 'job_vacio')]), spec), 'carpeta sin config.json')
        log("  OK")

        # ------------------------------------------------------------------ G
        log("#" * 100)
        log("SELFTEST G — job D2 con la referencia del job A (como el 8B repartido en dos GPU) y merge_level2")
        D2 = tmp / 'job_D2'
        run(st_args(base_argv[:-len(ST_EPS) - len(ST_SEEDS) - 2] + ['--epsilons', '1', '--seeds', *ST_SEEDS, '--lams', '1',
                                                                      '--conditions', 'psa', 'g1', '--reference',
                                                                      str(A / 'referencia_retain90'), '--out-dir', str(D2)]),
            spec)
        rows_d2 = st_no_seconds(st_rows(D2))
        assert len(rows_d2) == 4 and all(rows_d2[kk] == na[kk] for kk in rows_d2)
        import merge_level2
        m1 = merge_level2.merge([A, D2], tmp / 'merge_1')
        assert m1['n_rows'] == len(rows_a) and m1['n_dup'] == 4
        assert st_no_seconds(st_rows(tmp / 'merge_1')) == na
        bad = tmp / 'job_D2_bad'
        shutil.copytree(D2, bad)
        p = bad / 'resultados.csv'
        lines = p.read_text(encoding='utf-8').splitlines(True)
        cells = next(csv.reader([lines[1]]))
        cells[RESULT_COLUMNS.index('model_utility')] = repr(float(cells[RESULT_COLUMNS.index('model_utility')]) + 1e-3)
        lines[1] = strings_line(dict(zip(RESULT_COLUMNS, cells)))
        p.write_text(''.join(lines), encoding='utf-8')
        expect_exit(lambda: merge_level2.merge([A, bad], tmp / 'merge_2'), 'duplicado con otro valor')
        log("  OK: D2 reproduce las filas de A; la unión A + D2 acepta los 4 duplicados idénticos (AVISO) y aborta si uno difiere")

        # ------------------------------------------------------------------ H
        log("#" * 100)
        log("SELFTEST H — control contra logs oficiales (sintéticas) y resumen")
        off = tmp / 'official'
        (off / 'tofu_selftest_full' / 'evals_forget10').mkdir(parents=True)
        (off / 'tofu_selftest_retain90').mkdir(parents=True)
        shutil.copy(A / 'sin_steering' / 'TOFU_EVAL.json', off / 'tofu_selftest_full' / 'evals_forget10' / 'TOFU_EVAL.json')
        shutil.copy(A / 'referencia_retain90' / 'TOFU_EVAL.json', off / 'tofu_selftest_retain90' / 'TOFU_EVAL.json')
        res_a = read_results(A / 'resultados.csv')
        lines, n_av = official_control(res_a, off, 'tofu_selftest')
        assert n_av == 0, lines
        p = off / 'tofu_selftest_full' / 'evals_forget10' / 'TOFU_EVAL.json'
        o = json.loads(p.read_text(encoding='utf-8'))
        o['ra_Q_A_ROUGE']['agg_value'] += 0.05
        o['forget_quality']['agg_value'] *= 1e2
        o['privleak']['agg_value'] += 3.0
        o['retain_Q_A_Prob']['agg_value'] += 0.01                 # below the threshold: no AVISO
        p.write_text(json.dumps(o), encoding='utf-8')
        lines, n_av = official_control(res_a, off, 'tofu_selftest')
        assert n_av == 3, lines
        txt = build_resumen(res_a, protocol_a, origen='selftest', cond_dirs={}, official_dir=off,
                            official_prefix='tofu_selftest')
        assert '3 diferencia(s) por encima del umbral' in txt
        log("  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("#" * 100)
    log(f"SELFTEST PASSED in {time.time() - t0:.1f} s (temp dir removed; nothing written in the repo)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    sys.modules.setdefault('eval_level2', sys.modules[__name__])   # merge_level2's `import eval_level2` = this module
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = build_parser()
    args = parser.parse_args()
    if args.selftest:
        selftest(args)
        return
    if args.model_tag is None:
        parser.error("--model-tag is required")
    resolve_args(args)
    run(args, real_spec(args.model_tag))


if __name__ == '__main__':
    main()
