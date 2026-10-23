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
#   3. hydra-core / rouge-score / lm-eval / datasets at fixed versions, with pip install -c <constraints> where
#      <constraints> = tesis's pip freeze without editable or local lines (~/psa/tesis-ou_restricciones.txt).
#      `pip install --dry-run` first: STOPS if ANY package already installed would change version (or be reinstalled).
#      After installing: every package installed before still at its version, the requested ones at their versions,
#      `pip check` clean, and open-unlearning's evals imported from its src/ (not installed).
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
PKGS=("hydra-core==1.3.2" "rouge-score==0.1.2" "lm-eval==0.4.11" "datasets==3.0.1")
CONSTRAINTS="$PSA_DIR/tesis-ou_restricciones.txt"

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


def cmd_compare(base_path, env_path, mode, label_base, label_env):
    """identical: same packages and versions; superset: every package of base at the same version (extras allowed)."""
    base, env = load_pkgs(base_path), load_pkgs(env_path)
    changed = [(n, base[n], env.get(n)) for n in sorted(base) if env.get(n) != base[n]]
    extra = sorted(set(env) - set(base))
    for n, a, b in changed:
        print(f"    {n}: {label_base} {a} -> {label_env} {b if b is not None else '(no está)'}")
    if extra:
        print(f"    {len(extra)} paquete(s) en {label_env} que no están en {label_base}"
              f"{'' if mode == 'superset' else ' (no permitido)'}: "
              + ", ".join(f"{n} {env[n]}" for n in extra))
    bad = bool(changed) or (mode == 'identical' and bool(extra))
    if not bad:
        print(f"    {label_env}: {'mismos paquetes y versiones que' if mode == 'identical' else 'todo paquete de'} "
              f"{label_base}{'' if mode == 'identical' else ' en la misma versión'}")
    sys.exit(1 if bad else 0)


def cmd_constraints(freeze_path, out):
    """tesis's pip freeze without editable (-e ...) or local / direct-reference (name @ url) lines."""
    keep, skipped = [], []
    with open(freeze_path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('-e') or ' @ ' in s or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@]+', s):
                skipped.append(s)
            else:
                keep.append(s)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(keep) + '\n')
    print(f"    {len(keep)} restricciones name==version -> {out}; {len(skipped)} línea(s) excluida(s) (editables o locales)"
          + (':' if skipped else ''))
    for s in skipped:
        print(f"      {s[:160]}")


def cmd_requested(env_path, *specs):
    """exit 0: all installed at the requested versions; 10: some missing; 20: some installed at another version."""
    env = load_pkgs(env_path)
    missing = conflict = False
    for spec in specs:
        name, want = spec.split('==')
        have = env.get(norm(name))
        print(f"    {name}: instalado {have}, pedido {want}")
        missing |= have is None
        conflict |= have is not None and have != want
    sys.exit(20 if conflict else 10 if missing else 0)


def cmd_report(report_path, before_path):
    """pip --dry-run --report: STOP (exit 3) if any package installed before would be installed again (any version)."""
    before = load_pkgs(before_path)
    new, changes = [], []
    for it in load(report_path).get('install', []):
        n, v = norm(it['metadata']['name']), it['metadata']['version']
        (changes if n in before else new).append((n, before.get(n), v))
    print(f"    el dry-run instalaría {len(new)} paquete(s) nuevo(s): " + ", ".join(f"{n} {v}" for n, _, v in new))
    if changes:
        print(f"    y CAMBIARÍA {len(changes)} paquete(s) ya instalado(s):")
        for n, a, v in changes:
            print(f"      {n}: {a} -> {v}")
        sys.exit(3)
    print("    ningún paquete ya instalado cambia")


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
if [ "$FRESH" = 1 ]; then MODE=identical; else MODE=superset; fi
python "$H" compare "$WORK/tesis.json" "$WORK/before.json" "$MODE" "$BASE_ENV" "$ENV_NAME" \
    || die "$ENV_NAME no coincide con $BASE_ENV (ver arriba); para rehacerlo: bash setup_tesis_ou.sh --recreate"
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
python "$H" constraints "$WORK/tesis_freeze.txt" "$CONSTRAINTS"
rc=0
python "$H" requested "$WORK/before.json" "${PKGS[@]}" || rc=$?
case "$rc" in
    0)  echo "ya están instalados en esas versiones: no se instala nada" ;;
    10) echo "pip install --dry-run -c $CONSTRAINTS (todavía no se instala nada):"
        python -m pip install --dry-run --quiet --disable-pip-version-check --report "$WORK/report.json" \
            -c "$CONSTRAINTS" "${PKGS[@]}" \
            || die "pip no encontró versiones de ${PKGS[*]} compatibles con las de $BASE_ENV (ver el error de pip arriba; --report necesita pip >= 22.2); no se instaló nada"
        python "$H" report "$WORK/report.json" "$WORK/before.json" \
            || die "el --dry-run cambiaría paquetes ya instalados (ver arriba); no se instaló nada"
        python -m pip install --disable-pip-version-check -c "$CONSTRAINTS" "${PKGS[@]}" ;;
    20) die "un paquete pedido ya está instalado en $ENV_NAME en otra versión (ver arriba); cambiarlo cambiaría un paquete ya instalado; no se instaló nada" ;;
    *)  die "helper requested falló (código $rc)" ;;
esac
echo "paquetes de $ENV_NAME después:"
python "$H" list "$WORK/after.json"
python "$H" compare "$WORK/before.json" "$WORK/after.json" superset "antes" "después" \
    || die "cambió la versión de un paquete que ya estaba instalado (ver arriba): $ENV_NAME quedó roto, rehacerlo con --recreate"
python "$H" requested "$WORK/after.json" "${PKGS[@]}" || die "los paquetes pedidos no quedaron en las versiones pedidas"
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
assert steering_vectors.__version__ == '0.12.1', steering_vectors.__version__
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
python "$H" versions torch transformers numpy steering-vectors huggingface-hub tokenizers datasets fsspec pyarrow \
    accelerate hydra-core omegaconf rouge-score nltk lm-eval scikit-learn
echo
echo "LISTO. Restricciones usadas: $CONSTRAINTS"
echo "Siguiente paso (desde ~/psa):  sbatch selftest_level2.slurm"
