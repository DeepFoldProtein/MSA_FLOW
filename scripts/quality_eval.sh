#!/bin/bash
#SBATCH --job-name=msaflow-qeval
#SBATCH --nodes=1
#SBATCH --nodelist=ada-004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/qeval_%j.out
#SBATCH --error=logs/qeval_%j.err

# ── 환경 ──────────────────────────────────────────────────────────────────────
module load python/3.11.14
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
LMDB_PATH=${LMDB_PATH:-/gpfs/deepfold/users/yjlee4/msaflow_merged.lmdb}
DECODER_CKPT=${DECODER_CKPT:-$REPO_DIR/runs/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/quality_eval}

N_PROTEINS=${N_PROTEINS:-50}   # 단백질 수 (빠른 테스트: 10, 논문 수준: 200)
N_SEQS=${N_SEQS:-32}           # 단백질당 생성 시퀀스 수
N_STEPS=${N_STEPS:-100}        # ODE 스텝 수 (빠른 테스트: 50)
TEMPERATURE=${TEMPERATURE:-0.0}  # 0 = deterministic ODE

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "LMDB           : $LMDB_PATH"
echo "Decoder ckpt   : $DECODER_CKPT"
echo "Latent FM ckpt : $LATENT_FM_CKPT"
echo "N proteins     : $N_PROTEINS"
echo "N seqs/protein : $N_SEQS"
echo "ODE steps      : $N_STEPS"
date

python $REPO_DIR/msaflow/inference/quality_eval.py \
    --lmdb_path      $LMDB_PATH \
    --decoder_ckpt   $DECODER_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output_dir     $OUTPUT_DIR \
    --n_proteins     $N_PROTEINS \
    --n_seqs         $N_SEQS \
    --n_steps        $N_STEPS \
    --temperature    $TEMPERATURE \
    --device         cuda \
    --verbose

echo "Done: $(date)"
echo "Results: $OUTPUT_DIR/quality_eval.csv"
