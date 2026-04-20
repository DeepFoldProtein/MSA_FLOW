#!/bin/bash
#SBATCH --job-name=msaflow-overfit
#SBATCH --nodes=1
#SBATCH --nodelist=ada-004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1            # 1 GPU is enough for 16 samples
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/overfit_%j.out
#SBATCH --error=logs/overfit_%j.err

module load python/3.11.14
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_merged.lmdb}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/decoder_overfit}
CONFIG=$REPO_DIR/msaflow/configs/decoder_overfit.yaml

# Single-GPU accelerate config (no distributed)
ACCEL_CONFIG=$REPO_DIR/msaflow/configs/accelerate_1gpu.yaml

source $REPO_DIR/.venv/bin/activate

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Purpose    : OVERFIT SANITY CHECK (16 MSAs, 500 epochs)"
echo "Output dir : $OUTPUT_DIR"
date

accelerate launch \
    --config_file $ACCEL_CONFIG \
    $REPO_DIR/msaflow/training/train_decoder.py \
    --config $CONFIG \
    --lmdb_path $LMDB_PATH \
    --output_dir $OUTPUT_DIR

echo "Done: $(date)"
