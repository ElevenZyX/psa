#!/usr/bin/env bash
# One-time setup for eval_level2.py on the cluster. Run from ~/psa:
#     bash setup_tesis_ou.sh              # creates tesis-ou if missing, or verifies it and continues
#     bash setup_tesis_ou.sh --recreate   # deletes tesis-ou and clones it again from tesis
# The env "tesis" is only read (pip list / pip freeze), never modified.
#   1. conda env tesis-ou = clone of tesis. A fresh clone must have exactly tesis's packages and versions; an existing
#      tesis-ou must still have every package of tesis at tesis's version (it may only have extra packages), otherwise
#      it stops and asks for --recreate. `pip check` must be clean before installing anything.
#   2. open-unlearning cloned in ~/psa/open-unlearning and pinned at OU_COMMIT (an existing clone is verified, and checked
#      out at OU_COMMIT only if its tree is clean).
#   3. hydra-core / rouge-score / lm-eval at fixed versions and datasets (whatever version resolves), with
#      pip install -c <constraints> where <constraints> = tesis's pip freeze without editable or local lines and without
#      the packages of ALLOWED_CHANGES (~/psa/tesis-ou_restricciones.txt). `pip install --dry-run` first (full pip output
#      kept in ~/psa/tesis-ou_pip_dryrun.log): STOPS if any package already installed would change version (or be
#      reinstalled), except those of ALLOWED_CHANGES (AVISO with the old and new version). After installing: every package
#      installed before still at its version (same exception), the requested ones installed, `pip check` clean, and
#      open-unlearning's evals imported from its src/ (not installed).
#   4. OU's official eval logs: python setup_data.py --eval_logs (-> ~/psa/open-unlearning/saves/eval); the 6 TOFU_EVAL.json
#      of the 1B / 3B / 8B full and retain90 models are checked.
# Re-running it is safe: every step verifies what already exists and continues.
set -eo pipefail

RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        *) echo "uso: bash setup_tesis_ou.sh [--recreate]" >&2; exit 2 ;;
    esac
done

BASE_ENV=tesis
ENV_NAME=tesis-ou
OU_URL=https://github.com/locuslab/open-unlearning.git
OU_COMMIT=b29929067732c40ee4ca58d5cc1d3c11fcf37de1
PSA_DIR="$HOME/psa"
OU_DIR="$PSA_DIR/open-unlearning"
PKGS=("hydra-core==1.3.2" "rouge-score==0.1.2" "lm-eval==0.4.11" "datasets")
# Packages of tesis that MAY change version when PKGS are installed: they are left out of the constraints, and a change
# is reported as an AVISO; every other package of tesis must keep its exact version. Keep this list minimal and justify
# every entry here:
#   fsspec: file I/O library (local / remote filesystems for datasets and huggingface_hub), no numerical effect.
#           tesis has fsspec 2026.7.0 and every datasets installable next to huggingface-hub < 1.0 (transformers 4.57.6)
#           requires an older fsspec (datasets 5.0.1 -> fsspec 2026.6.0).
ALLOWED_CHANGES="fsspec"
CONSTRAINTS="$PSA_DIR/tesis-ou_restricciones.txt"
DRYRUN_LOG="$PSA_DIR/tesis-ou_pip_dryrun.log"
INSTALL_LOG="$PSA_DIR/tesis-ou_pip_install.log"

die() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== $*"; }

[ "$ENV_NAME" != "$BASE_ENV" ] || die "ENV_NAME == BASE_ENV"
[ "$(pwd -P)" = "$(cd "$PSA_DIR" && pwd -P)" ] || die "lanzar desde ~/psa (ahora en $(pwd))"
[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ] || die "no existe ~/miniconda/etc/profile.d/conda.sh"
source "$HOME/miniconda/etc/profile.d/conda.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
H="$WORK/helper.py"
cat > "$H" <<'EOF'
"""Helpers of setup_tesis_ou.sh. Package names are compared normalised (PEP 503)."""
import json
import re
import subprocess
import sys


def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_pkgs(path):
    return {norm(n): v for n, v in load(path).items()}


def cmd_list(out):
    """{name: version} of every distribution pip sees in the active env."""
    txt = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json', '--disable-pip-version-check'],
                         check=True, capture_output=True, text=True).stdout
    pkgs = {norm(d['name']): d['version'] for d in json.loads(txt)}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(pkgs, f, indent=1, sort_keys=True)
    print(f"    {len(pkgs)} paquetes -> {out}")


def allowed_set(allowed_csv):
    """The ALLOWED_CHANGES of setup_tesis_ou.sh: comma or space separated names."""
    return {norm(x) for x in re.split(r'[,\s]+', allowed_csv) if x}


def cmd_compare(base_path, env_path, mode, label_base, label_env, allowed_csv=''):
    """identical: same packages and versions; superset: every package of base at the same version (extras allowed).
    A package of allowed_csv may change version (AVISO) but not disappear."""
    base, env = load_pkgs(base_path), load_pkgs(env_path)
    allowed = allowed_set(allowed_csv)
    changed = [(n, base[n], env.get(n)) for n in sorted(base) if env.get(n) != base[n]]
    bad_changes = [c for c in changed if c[0] not in allowed or c[2] is None]
    for n, a, b in changed:
        if (n, a, b) in bad_changes:
            print(f"    {n}: {label_base} {a} -> {label_env} {b if b is not None else '(no está)'}")
        else:
            print(f"    AVISO: {n} cambió de versión (excepción permitida): {label_base} {a} -> {label_env} {b}")
    extra = sorted(set(env) - set(base))
    if extra:
        print(f"    {len(extra)} paquete(s) en {label_env} que no están en {label_base}"
              f"{'' if mode == 'superset' else ' (no permitido)'}: "
              + ", ".join(f"{n} {env[n]}" for n in extra))
    bad = bool(bad_changes) or (mode == 'identical' and bool(extra))
    if not bad:
        print(f"    {label_env}: {'mismos paquetes y versiones que' if mode == 'identical' else 'todo paquete de'} "
              f"{label_base}{'' if mode == 'identical' else ' en la misma versión'}"
              + (f" (salvo las excepciones permitidas: {', '.join(sorted(allowed))})" if allowed and changed else ''))
    sys.exit(1 if bad else 0)


def cmd_constraints(freeze_path, out, allowed_csv=''):
    """tesis's pip freeze without editable (-e ...) or local / direct-reference (name @ url) lines, and without the
    packages allowed to change (ALLOWED_CHANGES)."""
    allowed = allowed_set(allowed_csv)
    keep, skipped, excepted = [], [], []
    with open(freeze_path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('-e') or ' @ ' in s or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@]+', s):
                skipped.append(s)
            elif norm(s.split('==', 1)[0]) in allowed:
                excepted.append(s)
            else:
                keep.append(s)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(keep) + '\n')
    print(f"    {len(keep)} restricciones name==version -> {out}; {len(skipped)} línea(s) excluida(s) (editables o locales)"
          + (':' if skipped else ''))
    for s in skipped:
        print(f"      {s[:160]}")
    print(f"    sin restricción por excepción permitida (puede cambiar de versión): {', '.join(excepted) or '(ninguno)'}")
    missing = sorted(allowed - {norm(s.split('==', 1)[0]) for s in excepted})
    if missing:
        print(f"    (excepciones que no aparecen como name==version en el freeze: {', '.join(missing)})")


def cmd_requested(env_path, *specs):
    """specs 'name==version' or 'name' (any version). exit 0: all installed (pinned ones at their version);
    10: some missing; 20: some pinned one installed at another version."""
    env = load_pkgs(env_path)
    missing = conflict = False
    for spec in specs:
        name, _, want = spec.partition('==')
        have = env.get(norm(name))
        print(f"    {name}: instalado {have}, pedido {want or 'cualquier versión'}")
        missing |= have is None
        conflict |= bool(want) and have is not None and have != want
    sys.exit(20 if conflict else 10 if missing else 0)


def cmd_report(report_path, before_path, allowed_csv=''):
    """pip --dry-run --report: STOP (exit 3) if any package installed before would be installed again (any version),
    except the packages of allowed_csv (AVISO with the old and new version)."""
    before = load_pkgs(before_path)
    allowed = allowed_set(allowed_csv)
    new, changes = [], []
    for it in load(report_path).get('install', []):
        n, v = norm(it['metadata']['name']), it['metadata']['version']
        (changes if n in before else new).append((n, before.get(n), v))
    print(f"    el dry-run instalaría {len(new)} paquete(s) nuevo(s): " + ", ".join(f"{n} {v}" for n, _, v in new))
    for n, a, v in changes:
        if n in allowed:
            print(f"    AVISO: {n} cambiaría de versión (excepción permitida): {a} -> {v}")
    bad = [c for c in changes if c[0] not in allowed]
    if bad:
        print(f"    y CAMBIARÍA {len(bad)} paquete(s) ya instalado(s) que no son excepciones permitidas:")
        for n, a, v in bad:
            print(f"      {n}: {a} -> {v}")
        sys.exit(3)
    print("    ningún otro paquete ya instalado cambia" if changes else "    ningún paquete ya instalado cambia")


def cmd_versions(*names):
    import importlib.metadata as md
    for n in names:
        v = None
        for cand in (n, n.replace('-', '_')):
            try:
                v = md.version(cand)
                break
            except md.PackageNotFoundError:
                pass
        print(f"    {n:18s} {v}")


if __name__ == '__main__':
    {'list': cmd_list, 'compare': cmd_compare, 'constraints': cmd_constraints, 'requested': cmd_requested,
     'report': cmd_report, 'versions': cmd_versions}[sys.argv[1]](*sys.argv[2:])
EOF

env_exists() { conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"; }
pip_failure() {  # $1 = log of a failed pip command: pip's explanation (from its first ERROR line), complete in the log
    echo
    echo "---- explicación de pip (salida completa en $1) ----"
    if grep -q "ERROR" "$1"; then sed -n '/ERROR/,$p' "$1"; else tail -n 80 "$1"; fi
    echo "---- fin de la explicación de pip ----"
}
active_python_is_env() {
    case "$(command -v python)" in
        *"/envs/$ENV_NAME/"*) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------------------------------------------------
step "1. entorno conda $ENV_NAME (clon de $BASE_ENV)"
env_exists "$BASE_ENV" || die "no existe el entorno $BASE_ENV"
conda activate "$BASE_ENV"
echo "paquetes de $BASE_ENV (solo lectura):"
python "$H" list "$WORK/tesis.json"
python -m pip freeze --disable-pip-version-check > "$WORK/tesis_freeze.txt"
conda deactivate
if [ "$RECREATE" = 1 ] && env_exists "$ENV_NAME"; then
    echo "--recreate: se borra $ENV_NAME y se vuelve a clonar desde $BASE_ENV"
    for _ in 1 2 3 4 5 6 7 8; do [ "${CONDA_SHLVL:-0}" -gt 0 ] && conda deactivate || break; done
    [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ] || die "no se pudo desactivar $ENV_NAME para borrarlo"
    conda remove -y -n "$ENV_NAME" --all
    ! env_exists "$ENV_NAME" || die "no se pudo borrar $ENV_NAME"
fi
FRESH=0
if env_exists "$ENV_NAME"; then
    echo "$ENV_NAME ya existe: se verifica (para rehacerlo: bash setup_tesis_ou.sh --recreate)"
else
    conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
    FRESH=1
fi
conda activate "$ENV_NAME"
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && active_python_is_env || die "no se pudo activar $ENV_NAME ($(command -v python))"
echo "python $(python -c 'import sys; print(sys.version.split()[0])') en $(command -v python)"
echo "paquetes de $ENV_NAME:"
python "$H" list "$WORK/before.json"
if [ "$FRESH" = 1 ]; then
    # a fresh clone must be exactly tesis (no exception)
    python "$H" compare "$WORK/tesis.json" "$WORK/before.json" identical "$BASE_ENV" "$ENV_NAME" "" \
        || die "el clon recién hecho $ENV_NAME no coincide con $BASE_ENV (ver arriba)"
else
    # an existing tesis-ou: every package of tesis at its version, except ALLOWED_CHANGES (AVISO); extras allowed
    python "$H" compare "$WORK/tesis.json" "$WORK/before.json" superset "$BASE_ENV" "$ENV_NAME" "$ALLOWED_CHANGES" \
        || die "$ENV_NAME no coincide con $BASE_ENV (ver arriba); para rehacerlo: bash setup_tesis_ou.sh --recreate"
fi
echo "pip check de $ENV_NAME antes de instalar nada:"
python -m pip check \
    || die "pip check ya reporta conflictos ANTES de instalar nada (vienen de $BASE_ENV, o de un $ENV_NAME roto: --recreate); no se sigue"

# ---------------------------------------------------------------------------------------------------------------------
step "2. open-unlearning en $OU_DIR @ $OU_COMMIT"
if [ -d "$OU_DIR/.git" ]; then
    echo "el clon ya existe: se verifica"
    url="$(git -C "$OU_DIR" remote get-url origin)"
    case "$url" in
        *locuslab/open-unlearning*) ;;
        *) die "$OU_DIR apunta a $url, no a locuslab/open-unlearning" ;;
    esac
elif [ -e "$OU_DIR" ]; then
    die "$OU_DIR existe pero no es un clon git; no se toca"
else
    git clone "$OU_URL" "$OU_DIR"
fi
dirty="$(git -C "$OU_DIR" status --porcelain --untracked-files=all)"
if [ -n "$dirty" ]; then
    echo "$dirty"
    die "$OU_DIR tiene cambios locales o archivos sin seguimiento; no se toca"
fi
if [ "$(git -C "$OU_DIR" rev-parse HEAD)" != "$OU_COMMIT" ]; then
    git -C "$OU_DIR" cat-file -e "${OU_COMMIT}^{commit}" 2>/dev/null || git -C "$OU_DIR" fetch origin
    git -C "$OU_DIR" -c advice.detachedHead=false checkout --quiet "$OU_COMMIT"
fi
[ "$(git -C "$OU_DIR" rev-parse HEAD)" = "$OU_COMMIT" ] || die "no se pudo fijar $OU_COMMIT"
echo "HEAD = $OU_COMMIT, árbol limpio"

# ---------------------------------------------------------------------------------------------------------------------
step "3. ${PKGS[*]} en $ENV_NAME, con las versiones de $BASE_ENV como restricciones"
active_python_is_env || die "el python activo no es el de $ENV_NAME: $(command -v python)"
echo "excepciones permitidas (pueden cambiar de versión, con AVISO): $ALLOWED_CHANGES"
python "$H" constraints "$WORK/tesis_freeze.txt" "$CONSTRAINTS" "$ALLOWED_CHANGES"
rc=0
python "$H" requested "$WORK/before.json" "${PKGS[@]}" || rc=$?
case "$rc" in
    0)  echo "ya están instalados (los fijados, en sus versiones): no se instala nada" ;;
    10) echo "pip install --dry-run -c $CONSTRAINTS (todavía no se instala nada; salida completa en $DRYRUN_LOG):"
        if ! python -m pip install --dry-run --disable-pip-version-check --report "$WORK/report.json" \
                -c "$CONSTRAINTS" "${PKGS[@]}" 2>&1 | tee "$DRYRUN_LOG"; then
            pip_failure "$DRYRUN_LOG"
            die "pip no encontró versiones de ${PKGS[*]} compatibles con las restricciones de $BASE_ENV (excepciones permitidas: $ALLOWED_CHANGES; --report necesita pip >= 22.2); no se instaló nada"
        fi
        python "$H" report "$WORK/report.json" "$WORK/before.json" "$ALLOWED_CHANGES" \
            || die "el --dry-run cambiaría paquetes ya instalados que no son excepciones permitidas (ver arriba); no se instaló nada"
        echo "pip install -c $CONSTRAINTS (salida completa en $INSTALL_LOG):"
        if ! python -m pip install --disable-pip-version-check -c "$CONSTRAINTS" "${PKGS[@]}" 2>&1 | tee "$INSTALL_LOG"; then
            pip_failure "$INSTALL_LOG"
            die "pip install falló (ver arriba): $ENV_NAME puede haber quedado a medias, rehacerlo con --recreate"
        fi ;;
    20) die "un paquete pedido ya está instalado en $ENV_NAME en otra versión (ver arriba); cambiarlo cambiaría un paquete ya instalado; no se instaló nada" ;;
    *)  die "helper requested falló (código $rc)" ;;
esac
echo "paquetes de $ENV_NAME después:"
python "$H" list "$WORK/after.json"
python "$H" compare "$WORK/before.json" "$WORK/after.json" superset "antes" "después" "$ALLOWED_CHANGES" \
    || die "cambió la versión de un paquete ya instalado que no es excepción permitida (ver arriba): $ENV_NAME quedó roto, rehacerlo con --recreate"
python "$H" requested "$WORK/after.json" "${PKGS[@]}" || die "los paquetes pedidos no quedaron instalados (o no en las versiones pedidas)"
echo "pip check:"
python -m pip check || die "pip check reporta conflictos después de instalar (ver arriba): $ENV_NAME quedó roto, rehacerlo con --recreate"
echo "prueba de importación (open-unlearning desde su src/, sin instalarlo):"
python - "$OU_DIR" <<'EOF'
import sys
from pathlib import Path
src = (Path(sys.argv[1]) / 'src').resolve()
sys.path.insert(0, str(src))
import data    # open-unlearning
import evals   # open-unlearning (imports lm_eval)
import evals.metrics
for mod in (data, evals, evals.metrics):
    assert src in Path(mod.__file__).resolve().parents, f"{mod.__name__} from {mod.__file__}"
assert 'trainer' not in sys.modules, "open-unlearning's trainer package got imported"
import hydra, lm_eval, rouge_score, steering_vectors, torch, transformers  # noqa: F401
assert int(transformers.__version__.split('.')[0]) < 5, transformers.__version__
assert steering_vectors.__version__ == '0.12.2', steering_vectors.__version__  # = eval_level2.STEERING_VECTORS_VERSION
print(f"    evals desde {evals.__file__}: OK")
EOF

# ---------------------------------------------------------------------------------------------------------------------
step "4. logs oficiales de open-unlearning (setup_data.py --eval_logs -> $OU_DIR/saves/eval)"
( cd "$OU_DIR" && python setup_data.py --eval_logs )
for m in tofu_Llama-3.2-1B-Instruct tofu_Llama-3.2-3B-Instruct tofu_Llama-3.1-8B-Instruct; do
    for f in "${m}_full/evals_forget10/TOFU_EVAL.json" "${m}_retain90/TOFU_EVAL.json"; do
        [ -s "$OU_DIR/saves/eval/$f" ] || die "falta $OU_DIR/saves/eval/$f"
        echo "    ok $f"
    done
done
dirty="$(git -C "$OU_DIR" status --porcelain --untracked-files=all)"
if [ -n "$dirty" ]; then
    echo "$dirty"
    die "setup_data.py dejó archivos no ignorados en $OU_DIR (eval_level2.py exige el árbol limpio)"
fi

# ---------------------------------------------------------------------------------------------------------------------
step "5. versiones finales de $ENV_NAME"
echo "frente a $BASE_ENV:"
python "$H" compare "$WORK/tesis.json" "$WORK/after.json" superset "$BASE_ENV" "$ENV_NAME" "$ALLOWED_CHANGES" \
    || die "$ENV_NAME no coincide con $BASE_ENV fuera de las excepciones permitidas (ver arriba)"
python "$H" versions torch transformers numpy steering-vectors huggingface-hub tokenizers datasets fsspec pyarrow \
    accelerate hydra-core omegaconf rouge-score nltk lm-eval scikit-learn
echo
echo "LISTO. Restricciones usadas: $CONSTRAINTS"
echo "Siguiente paso (desde ~/psa):  sbatch selftest_level2.slurm"
