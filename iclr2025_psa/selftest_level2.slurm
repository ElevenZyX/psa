#!/bin/bash
#SBATCH --job-name=selftest_level2
#SBATCH --partition=compute-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=144:00:00
#SBATCH --output=selftest_level2_%j.out
#SBATCH --error=selftest_level2_%j.err
# Selftests of eval_level2.py and merge_level2.py (CPU: tiny random model, local TOFU-like data, temp dir only).
# Launch from ~/psa (the --output / --error paths are relative to the submission directory):
#     cd ~/psa && sbatch selftest_level2.slurm
# Requires setup_tesis_ou.sh (env tesis-ou, ~/psa/open-unlearning at the pinned commit).

module swap gnu14 gnu12/12.2.0 2>/dev/null || module load gnu12/12.2.0
module load cuda/12.6
source ~/miniconda/etc/profile.d/conda.sh && conda activate tesis-ou
set -eo pipefail

cd ~/psa/iclr2025_psa
echo "host $(hostname) | job ${SLURM_JOB_ID} | $(date) | python $(command -v python)"
python eval_level2.py --selftest
python merge_level2.py --selftest
echo "selftests de eval_level2.py y merge_level2.py OK | $(date)"
