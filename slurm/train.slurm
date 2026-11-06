#!/bin/bash
# Generic SLURM job for the three training scripts.
#
# Usage (from the repository root):
#   sbatch slurm/train.slurm diffusion --data-root /path/to/data
#   sbatch slurm/train.slurm vit_gan   --data-root /path/to/data
#   sbatch slurm/train.slurm cvae      --data-root /path/to/data/celeba
#
# The scripts resume from their latest checkpoint, so long trainings can be
# split across several jobs when the cluster has a wall-time limit.
# Adapt partition / account / resources to your cluster.

#SBATCH --job-name=face-gen
#SBATCH --partition=gpuq
#SBATCH --account=<your-account>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1

set -euo pipefail
MODEL=${1:?"usage: sbatch slurm/train.slurm <cvae|diffusion|vit_gan> [args...]"}
shift

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
PYTHON=${PYTHON:-python}

echo "Job started: $(date) on $(hostname)"
nvidia-smi
"$PYTHON" -m "$MODEL.train" "$@"
echo "Job finished: $(date)"
