"""
Joins the resultados.csv of several eval_level2.py jobs of the SAME model and protocol (for instance the 8B split over two
GPUs, or a reference / noiseless job plus a noisy job) into one directory: resultados.csv, config.json and resumen.txt.

Rules (approved by Diego, 2026-10-22):
  * every job must have exactly the same protocol (eval_level2's config.json 'protocolo': model and snapshots, OU commit,
    TOFU revision, suite, prompt, batch, GPU type, library versions, inputs); otherwise abort;
  * a key (modelo, condicion, epsilon, semilla, lambda) present in several jobs is accepted, with an AVISO, only if the rows
    are bit-identical (every column but 'segundos'); if they differ, abort;
  * every row must use the same reference (referencia_sha256), and the same (condicion, epsilon, semilla) must have the same
    vector_sha256 under every λ;
  * --out-dir is created, or replaced only if it holds a previous output of this script.
resumen.txt is eval_level2.build_resumen on the joined rows (without the vector section: see each job's resumen.txt).

    python merge_level2.py nivel2_8b_a nivel2_8b_b --out-dir nivel2_8b [--official-logs ../open-unlearning/saves/eval]
    python merge_level2.py --selftest     # synthetic jobs, no model, temp dir only
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import eval_level2 as el2
from eval_level3 import dict_diff, expect_exit
from eval_pilot import log


def load_job(d: Path) -> dict:
    cfgp, csvp = d / 'config.json', d / 'resultados.csv'
    if not cfgp.is_file() or not csvp.is_file():
        raise SystemExit(f"{d} is not an eval_level2.py job directory (no config.json / resultados.csv). Stopping.")
    cfg = json.loads(cfgp.read_text(encoding='utf-8'))
    if cfg.get('script') != 'eval_level2.py' or 'protocolo' not in cfg:
        raise SystemExit(f"{cfgp} was not written by eval_level2.py. Stopping.")
    protocol = cfg['protocolo']
    rows = el2.parse_results_text(csvp.read_text(encoding='utf-8'), protocol['modelo_tag'], str(csvp))
    return {'dir': d, 'protocol': protocol, 'rows': rows}


def merge(dirs: list, out_dir: Path, official_dir=None) -> dict:
    el2.setup_logging()
    out_dir = Path(out_dir)
    log("=" * 100)
    log(f"UNIÓN de resultados del nivel 2: {[str(d) for d in dirs]} → {out_dir}")
    if not dirs:
        raise SystemExit("no job directories given")
    if len({str(Path(d).resolve()) for d in dirs}) != len(dirs):
        raise SystemExit(f"a job directory is repeated in {dirs}. Stopping.")
    if any(Path(d).resolve() == out_dir.resolve() for d in dirs):
        raise SystemExit(f"--out-dir {out_dir} is one of the inputs. Stopping.")
    jobs = [load_job(Path(d)) for d in dirs]
    base = jobs[0]['protocol']
    for j in jobs[1:]:
        diff = dict_diff(base, j['protocol'])
        if diff:
            raise SystemExit(f"{j['dir']} has another protocol than {jobs[0]['dir']}; refusing to join:\n  "
                             + "\n  ".join(diff[:30]))
    merged, origin, n_dup = {}, {}, 0
    for j in jobs:
        for k, r in j['rows'].items():
            if k in merged:
                a = {c: v for c, v in merged[k].items() if c != 'segundos'}
                b = {c: v for c, v in r.items() if c != 'segundos'}
                if a != b:
                    bad = [c for c in a if a[c] != b[c]]
                    raise SystemExit(f"key {k} is in {origin[k]} and in {j['dir']} with different values in {bad}. "
                                     f"Stopping.")
                n_dup += 1
                log(f"  <-- AVISO: {k} está en {origin[k]} y en {j['dir']} con valores idénticos bit a bit; se usa la "
                    f"primera")
                continue
            merged[k] = r
            origin[k] = j['dir']
    refs = sorted({r['referencia_sha256'] for r in merged.values()})
    if len(refs) != 1:
        raise SystemExit(f"the rows use {len(refs)} different references {[x[:16] for x in refs]}. Stopping.")
    vec = {}
    for k, r in merged.items():
        if r['condicion'] not in el2.UNSTEERED:
            vec.setdefault((k[1], k[2], k[3]), set()).add(r['vector_sha256'])
    bad = {vk: s for vk, s in vec.items() if len(s) > 1}
    if bad:
        raise SystemExit(f"the same vector has different sha256 under different λ: {list(bad)[:5]}. Stopping.")
    if out_dir.exists():
        prev = out_dir / 'config.json'
        if not prev.is_file() or json.loads(prev.read_text(encoding='utf-8')).get('script') != 'merge_level2.py':
            raise SystemExit(f"{out_dir} exists and is not a previous output of merge_level2.py. Stopping.")
        log(f"  <-- AVISO: {out_dir} ya tenía una unión anterior; se reemplazan resultados.csv, config.json y resumen.txt")
    else:
        out_dir.mkdir(parents=True)
    order = sorted(merged, key=lambda k: el2.row_sort_key(k[1], k[2], k[3], k[4]))
    text = el2.HEADER + ''.join(el2.strings_line(merged[k]) for k in order)
    tag = base['modelo_tag']
    joined = el2.parse_results_text(text, tag, f"{out_dir} (unión)")
    if set(joined) != set(merged):
        raise SystemExit("the joined csv does not round-trip. Stopping.")
    el2.write_text_atomic(out_dir / 'resultados.csv', text)
    el2.write_json_atomic(out_dir / 'config.json', {
        'script': 'merge_level2.py', 'protocolo': base, 'entradas': [str(j['dir']) for j in jobs],
        'filas_por_entrada': {str(j['dir']): len(j['rows']) for j in jobs}, 'n_filas': len(order),
        'duplicados_identicos': n_dup, 'referencia_sha256': refs[0], 'fecha': el2.now_iso()})
    res = el2.read_results(out_dir / 'resultados.csv')
    cond_dirs = {k: Path(origin[k]) / el2.task_rel_dir_from_key(k) for k in order}
    checks = [f"{len(jobs)} jobs con el mismo protocolo; {len(order)} filas; {n_dup} claves repetidas con valores idénticos; "
              f"una sola referencia ({refs[0][:16]}…); mismo vector_sha256 para cada (condición, ε, semilla) en todos los λ"]
    checks += [f"{j['dir']}: {len(j['rows'])} filas" for j in jobs]
    summary = el2.build_resumen(res, base, origen=f"unión de {len(jobs)} jobs", cond_dirs=cond_dirs, checks=checks,
                                official_dir=official_dir, official_prefix=el2.OFFICIAL_PREFIX.get(tag))
    el2.write_text_atomic(out_dir / 'resumen.txt', summary)
    log(f"\nunión escrita en {out_dir}: {len(order)} filas ({n_dup} duplicados idénticos)")
    return {'out_dir': out_dir, 'n_rows': len(order), 'n_dup': n_dup}


# --------------------------------------------------------------------------------------
# Self-test (synthetic jobs, no model)
# --------------------------------------------------------------------------------------
def st_vsha(cond: str, eps, seed) -> str:
    return hashlib.sha256(f"{cond}|{eps}|{seed}".encode()).hexdigest()


def st_row(rng, cond: str, lam: float = 0.0, eps=float('nan'), seed=float('nan'), ref: str = 'a' * 64) -> dict:
    metrics = {m: float(rng.uniform(0.05, 0.95)) for m in el2.METRIC_COLUMNS}
    if cond == el2.REF:
        metrics['forget_quality'] = None
        metrics['privleak'] = None
    vsha = '' if cond in el2.UNSTEERED else st_vsha(cond, eps, seed)
    return {'modelo': 'selftest', 'ventana': '12-16', 'condicion': cond, 'epsilon': eps, 'semilla': seed, 'lambda': lam,
            **metrics, 'vector_sha256': vsha, 'referencia_sha256': ref, 'segundos': float(rng.uniform(1, 9))}


def st_job(root: Path, name: str, protocol: dict, rows: list) -> Path:
    d = root / name
    d.mkdir(parents=True)
    el2.write_json_atomic(d / 'config.json', {'script': 'eval_level2.py', 'protocolo': protocol, 'creado': 'selftest',
                                              'job_inicial': {}})
    store = el2.ResultsStore(d / 'resultados.csv', protocol['modelo_tag'])
    for r in rows:
        store.append(r)
    return d


def selftest() -> None:
    t0 = time.time()
    el2.setup_logging()
    tmp = Path(tempfile.mkdtemp(prefix='merge_level2_selftest_'))
    try:
        rng = np.random.default_rng(0)
        P = el2.st_fake_protocol()
        REF, NONE, MEAN, PSA0, G1_0, PSA, G1 = el2.CONDITION_ORDER
        rows_a = [st_row(rng, REF), st_row(rng, NONE)] + [st_row(rng, c, 1.0) for c in (MEAN, PSA0, G1_0)]
        rows_b = [st_row(rng, c, 1.0, 1.0, s) for s in (0, 1) for c in (PSA, G1)] + [st_row(rng, MEAN, 2.0)]
        A = st_job(tmp, 'A', P, rows_a)
        B = st_job(tmp, 'B', P, rows_b)

        log("#" * 100)
        log("SELFTEST merge 1 — dos jobs disjuntos")
        m = merge([A, B], tmp / 'M')
        assert m['n_rows'] == 10 and m['n_dup'] == 0
        joined = el2.parse_results_text((tmp / 'M' / 'resultados.csv').read_text(encoding='utf-8'), 'selftest', 'M')
        keys = list(joined)
        assert keys == sorted(keys, key=lambda k: el2.row_sort_key(k[1], k[2], k[3], k[4]))
        assert keys[0][1] == REF and keys[1][1] == NONE
        both = {**el2.parse_results_text((A / 'resultados.csv').read_text(encoding='utf-8'), 'selftest', 'A'),
                **el2.parse_results_text((B / 'resultados.csv').read_text(encoding='utf-8'), 'selftest', 'B')}
        assert joined == both
        txt = (tmp / 'M' / 'resumen.txt').read_text(encoding='utf-8')
        assert 'λ = +1' in txt and 'λ = +2' in txt and 'unión de 2 jobs' in txt
        log("  OK")

        log("#" * 100)
        log("SELFTEST merge 2 — duplicados idénticos (salvo segundos) se aceptan con AVISO; distintos abortan")
        C = st_job(tmp, 'C', P, [{**r, 'segundos': r['segundos'] + 10} for r in rows_b])
        m = merge([A, B, C], tmp / 'M')            # replaces the previous union in M
        assert m['n_rows'] == 10 and m['n_dup'] == len(rows_b)
        D = st_job(tmp, 'D', P, [{**r, 'model_utility': r['model_utility'] + (1e-12 if i == 2 else 0.0)}
                                 for i, r in enumerate(rows_b)])
        expect_exit(lambda: merge([A, B, D], tmp / 'M2'), 'duplicado con un valor distinto')
        log("  OK")

        log("#" * 100)
        log("SELFTEST merge 3 — protocolo, referencia y vectores inconsistentes; carpetas inválidas")
        P_gpu = json.loads(json.dumps(P))
        P_gpu['dispositivo']['gpu'] = 'NVIDIA A100-SXM4-40GB'
        E = st_job(tmp, 'E', P_gpu, [st_row(rng, MEAN, 3.0)])
        expect_exit(lambda: merge([A, E], tmp / 'M3'), 'otro protocolo (GPU)')
        F = st_job(tmp, 'F', P, [st_row(rng, MEAN, 3.0, ref='b' * 64)])
        expect_exit(lambda: merge([A, F], tmp / 'M4'), 'otra referencia')
        G = st_job(tmp, 'G', P, [{**st_row(rng, MEAN, 3.0), 'vector_sha256': 'c' * 64}])
        expect_exit(lambda: merge([A, G], tmp / 'M5'), 'mismo vector con otro sha256 en otro λ')
        expect_exit(lambda: merge([A, A], tmp / 'M6'), 'carpeta repetida')
        expect_exit(lambda: merge([A, B], A), '--out-dir es una entrada')
        (tmp / 'otra').mkdir()
        (tmp / 'otra' / 'config.json').write_text(json.dumps({'script': 'otro'}), encoding='utf-8')
        expect_exit(lambda: merge([A, B], tmp / 'otra'), '--out-dir que no es una unión anterior')
        expect_exit(lambda: merge([A, tmp / 'otra'], tmp / 'M7'), 'entrada que no es un job de eval_level2')
        log("  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("#" * 100)
    log(f"SELFTEST PASSED in {time.time() - t0:.1f} s (temp dir removed; nothing written in the repo)")


def main():
    sys.modules.setdefault('merge_level2', sys.modules[__name__])
    for stream in (sys.stdout, sys.stderr):  # non-UTF-8 consoles (Windows cp1252) must not crash the prints
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs='*', help="eval_level2.py job directories (same model and protocol)")
    parser.add_argument("--out-dir", default=None, help="output directory (new, or a previous output of this script)")
    parser.add_argument("--official-logs", default=None, help="OU's saves/eval, for the control section of resumen.txt")
    parser.add_argument("--selftest", action='store_true', help="synthetic jobs, no model, temp dir only")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.dirs or args.out_dir is None:
        parser.error("give the job directories and --out-dir")
    merge([Path(d) for d in args.dirs], Path(args.out_dir), args.official_logs)


if __name__ == '__main__':
    main()
