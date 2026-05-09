#!/bin/bash
#SBATCH --job-name=msaflow-fewshot-all
#SBATCH --nodes=1
#SBATCH --nodelist=ada-003
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/fewshot_all_%j.out
#SBATCH --error=logs/fewshot_all_%j.err
# fewshot — 332개 전체, colabfold MSA 사용
# sbatch scripts/run_fewshot_all.sh

module load python/3.11.14
module load cuda/13.0.2

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/latest.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
PROTENIX_CKPT=${PROTENIX_CKPT:-$REPO_DIR/checkpoint/protenix_base_default_v1.0.0.pt}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
MSA_DIR=${MSA_DIR:-/gpfs/deepfold/users/yjlee4/foldbench_search}
ALL_FASTA=${ALL_FASTA:-$REPO_DIR/data/foldbench_monomer.fasta}
BASE_DIR=${BASE_DIR:-$REPO_DIR/runs/benchmark_all}
USALIGN_BIN=${USALIGN_BIN:-$HOME/.local/bin/USalign}

N_SEQS=${N_SEQS:-32}
N_SEEDS=${N_SEEDS:-5}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}
MAX_REC_DEPTH=${MAX_REC_DEPTH:-128}

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH
export PATH="$HOME/.local/bin:$PATH"
export PROTENIX_ROOT_DIR=$REPO_DIR

mkdir -p $BASE_DIR/fewshot_all $REPO_DIR/logs

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "FASTA    : $ALL_FASTA  ($(grep -c '^>' $ALL_FASTA) proteins)"
echo "MSA dir  : $MSA_DIR"
date

# GPU 확인
IFS=',' read -ra ALL_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
HEALTHY_GPUS=()
for _GPU in "${ALL_GPUS[@]}"; do
    if CUDA_VISIBLE_DEVICES=$_GPU python -c \
        "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        HEALTHY_GPUS+=("$_GPU")
        echo "  GPU $_GPU: OK"
    else
        echo "  GPU $_GPU: 불량 — 제외"
    fi
done
NUM_SHARDS=${#HEALTHY_GPUS[@]}
echo "사용 GPU: ${HEALTHY_GPUS[*]}  (shards=$NUM_SHARDS)"

is_done() { [ -s "$1/benchmark_results.csv" ]; }

merge_shards() {
    local mode_dir=$1
    python - << PYEOF
import csv, glob, math
output_dir = "$mode_dir"
rows, header = [], None
for shard_csv in sorted(glob.glob(f"{output_dir}/shard_*.csv")):
    with open(shard_csv) as fh:
        reader = csv.DictReader(fh)
        if header is None:
            header = reader.fieldnames
        rows.extend(reader)
if rows:
    out_path = f"{output_dir}/benchmark_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tm_vals = [float(r["tm_score"]) for r in rows
               if r.get("tm_score") not in ("", "nan", None)
               and not math.isnan(float(r["tm_score"]))]
    print(f"  merged {len(rows)} rows → {out_path}")
    if tm_vals:
        print(f"  TM-score  n={len(tm_vals)}  mean={sum(tm_vals)/len(tm_vals):.4f}")
else:
    print(f"  No shard CSVs found in {output_dir}")
PYEOF
}

echo ""
echo "=== fewshot — all 332 ($(date)) ==="
if is_done $BASE_DIR/fewshot_all; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=${HEALTHY_GPUS[$SHARD_ID]} \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta           $ALL_FASTA \
            --decoder_ckpt    $DECODER_CKPT \
            --latent_fm_ckpt  $LATENT_FM_CKPT \
            --output_dir      $BASE_DIR/fewshot_all \
            --mode            fewshot \
            --protenix_model  $PROTENIX_MODEL \
            --protenix_ckpt   $PROTENIX_CKPT \
            --shallow_msa_dir $MSA_DIR \
            --max_rec_depth   $MAX_REC_DEPTH \
            --ref_cif_dir     $REF_CIF_DIR \
            --usalign_bin     $USALIGN_BIN \
            --device          cuda \
            --num_shards      $NUM_SHARDS \
            --shard_id        $SHARD_ID \
            --n_seqs          $N_SEQS \
            --n_seeds         $N_SEEDS \
            --n_steps         $N_STEPS \
            --temperature     $TEMPERATURE \
            > $BASE_DIR/fewshot_all/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  fewshot_all done: $(date)"
    merge_shards $BASE_DIR/fewshot_all
fi

echo ""
echo "결과 → $BASE_DIR/fewshot_all/benchmark_results.csv"
echo "완료: $(date)"
