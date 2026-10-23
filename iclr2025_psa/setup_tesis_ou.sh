#!/usr/bin/env bash
# One-time setup for eval_level2.py on the cluster. Run from ~/psa:   bash setup_tesis_ou.sh
#   1. conda env "tesis-ou" = clone of "tesis" (if it already exists it is verified, never recreated);
#   2. hydra-core / rouge-score / lm-eval: `pip install --dry-run` first; STOPS if torch, transformers, numpy or
#      steering-vectors would change; otherwise installs them (nothing is installed if they are already at these versions);
#   3. open-unlearning cloned in ~/psa/open-unlearning and pinned at OU_COMMIT (if the clone exists: verified, and checked
#      out at OU_COMMIT only if its tree is clean);
#   4. OU's official eval logs: python setup_data.py --eval_logs (-> ~/psa/open-unlearning/saves/eval), the 6 TOFU_EVAL.json
#      of the 1B / 3B / 8B full and retain90 models are checked;
#   5. smoke test: OU's evals imported from its src/ with this environment; versions printed.
# Re-running it is safe: every step verifies what already exists and continues.
set -eo pipefail

BASE_ENV=tesis
ENV_NAME=tesis-ou
OU_URL=https://github.com/locuslab/open-unlearning.git
OU_COMMIT=b29929067732c40ee4ca58d5cc1d3c11fcf37de1
PSA_DIR="$HOME/psa"
OU_DIR="$PSA_DIR/open-unlearning"
PKGS=("hydra-core==1.3.2" "rouge-score==0.1.2" "lm-eval==0.4.11")

die() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== $*"; }

[ "$(pwd -P)" = "$(cd "$PSA_DIR" && pwd -P)" ] || die "lanzar desde ~/psa (ahora en $(pwd))"
[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ] || die "no existe ~/miniconda/etc/profile.d/conda.sh"
source "$HOME/miniconda/etc/profile.d/conda.sh"

# "name version" of the packages that must not change (torch, transformers, numpy, steering-vectors), active env
protected_versions() {
    python - <<'EOF'
import importlib.metadata as md
for n in ('torch', 'transformers', 'numpy', 'steering-vectors'):
    v = 'NO-INSTALADO'
    for cand in (n, n.replace('-', '_')):
        try:
            v = md.version(cand)
            break
        except md.PackageNotFoundError:
            pass
    print(n, v)
EOF
}

env_exists() { conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"; }

# ---------------------------------------------------------------------------------------------------------------------
step "1. entorno conda $ENV_NAME (clon de $BASE_ENV)"
env_exists "$BASE_ENV" || die "no existe el entorno $BASE_ENV"
conda activate "$BASE_ENV"
BASE_VERSIONS="$(protected_versions)"
conda deactivate
echo "versiones protegidas en $BASE_ENV:"
echo "$BASE_VERSIONS" | sed 's/^/    /'
if env_exists "$ENV_NAME"; then
    echo "$ENV_NAME ya existe: se verifica (no se recrea)"
else
    conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
fi
conda activate "$ENV_NAME"
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || die "no se pudo activar $ENV_NAME"
case "$(command -v python)" in
    *"/envs/$ENV_NAME/"*) ;;
    *) die "el python activo no es el de $ENV_NAME: $(command -v python)" ;;
esac
ENV_VERSIONS="$(protected_versions)"
if [ "$ENV_VERSIONS" != "$BASE_VERSIONS" ]; then
    echo "$ENV_NAME:"; echo "$ENV_VERSIONS" | sed 's/^/    /'
    die "torch / transformers / numpy / steering-vectors de $ENV_NAME difieren de $BASE_ENV; no se sigue"
fi
echo "python $(python -c 'import sys; print(sys.version.split()[0])') en $(command -v python); versiones protegidas = $BASE_ENV"

# ---------------------------------------------------------------------------------------------------------------------
step "2. ${PKGS[*]} en $ENV_NAME"
if python - "${PKGS[@]}" <<'EOF'
import importlib.metadata as md
import sys
ok = True
for spec in sys.argv[1:]:
    name, want = spec.split('==')
    have = None
    for cand in (name, name.replace('-', '_')):
        try:
            have = md.version(cand)
            break
        except md.PackageNotFoundError:
            pass
    print(f"    {name}: instalado {have}, pedido {want}")
    ok = ok and have == want
sys.exit(0 if ok else 1)
EOF
then
    echo "ya están instalados en esas versiones: no se instala nada"
else
    REPORT="$(mktemp)"
    echo "pip install --dry-run (todavía no se instala nada):"
    python -m pip install --dry-run --quiet --report "$REPORT" "${PKGS[@]}" \
        || die "pip install --dry-run falló (--report necesita pip >= 22.2: python -m pip install -U pip en $ENV_NAME)"
    python - "$REPORT" <<'EOF' || die "el --dry-run cambiaría paquetes protegidos: no se instaló nada"
import importlib.metadata as md
import json
import re
import sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
protected = {'torch', 'transformers', 'numpy', 'steering-vectors'}
norm = lambda s: re.sub(r'[-_.]+', '-', s).lower()
def have(name):
    for cand in (name, name.replace('-', '_')):
        try:
            return md.version(cand)
        except md.PackageNotFoundError:
            pass
    return None
items = report.get('install', [])
print(f"    el dry-run instalaría o cambiaría {len(items)} paquete(s):")
bad = []
for it in items:
    name, new = norm(it['metadata']['name']), it['metadata']['version']
    flag = ''
    if name in protected:
        bad.append(name)
        flag = '   <-- PROTEGIDO'
    print(f"      {name}: {have(name) or '(nuevo)'} -> {new}{flag}")
if bad:
    print(f"    PROTEGIDOS que cambiarían: {bad}")
sys.exit(3 if bad else 0)
EOF
    rm -f "$REPORT"
    python -m pip install "${PKGS[@]}"
    [ "$(protected_versions)" = "$BASE_VERSIONS" ] || die "después de instalar cambiaron paquetes protegidos"
fi

# ---------------------------------------------------------------------------------------------------------------------
step "3. open-unlearning en $OU_DIR @ $OU_COMMIT"
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
step "5. prueba de importación (open-unlearning desde su src/, sin instalarlo)"
python - "$OU_DIR" <<'EOF'
import importlib.metadata as md
import sys
from pathlib import Path
src = (Path(sys.argv[1]) / 'src').resolve()
sys.path.insert(0, str(src))
import data    # open-unlearning
import evals   # open-unlearning (imports lm_eval)
for mod in (data, evals):
    assert src in Path(mod.__file__).resolve().parents, f"{mod.__name__} from {mod.__file__}"
import hydra, lm_eval, rouge_score, steering_vectors, torch, transformers  # noqa: F401
assert int(transformers.__version__.split('.')[0]) < 5, transformers.__version__
assert steering_vectors.__version__ == '0.12.1', steering_vectors.__version__
print(f"    evals desde {evals.__file__}")
for n in ('torch', 'transformers', 'numpy', 'steering-vectors', 'datasets', 'accelerate', 'hydra-core', 'omegaconf',
          'rouge-score', 'nltk', 'lm-eval', 'scikit-learn'):
    v = None
    for cand in (n, n.replace('-', '_')):
        try:
            v = md.version(cand)
            break
        except md.PackageNotFoundError:
            pass
    print(f"    {n:18s} {v}")
EOF

echo
echo "LISTO. Siguiente paso (desde ~/psa):  sbatch selftest_level2.slurm"
