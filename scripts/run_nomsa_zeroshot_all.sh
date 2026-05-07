#!/bin/bash
#SBATCH --job-name=msaflow-nz-all
#SBATCH --nodes=1
#SBATCH --nodelist=ada-003
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/nz_all_%j.out
#SBATCH --error=logs/nz_all_%j.err
# nomsa + zeroshot — 332개 전체 단백질 대상 오버나이트 실행
# sbatch scripts/run_nomsa_zeroshot_all.sh

module load python/3.11.14
module load cuda/13.0.2

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/latest.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
FASTA_DIR=${FASTA_DIR:-$REPO_DIR/data/foldbench_groups}
BASE_DIR=${BASE_DIR:-$REPO_DIR/runs/benchmark_all}
USALIGN_BIN=${USALIGN_BIN:-$HOME/.local/bin/USalign}

N_SEQS=${N_SEQS:-32}
N_SEEDS=${N_SEEDS:-5}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH
export PATH="$HOME/.local/bin:$PATH"

mkdir -p $BASE_DIR/nomsa_all $BASE_DIR/zeroshot_all $REPO_DIR/logs

# 332개 전체 FASTA 생성 (orphan + shallow + full)
ALL_FASTA=$BASE_DIR/foldbench_all.fasta
cat $FASTA_DIR/foldbench_orphan.fasta \
    $FASTA_DIR/foldbench_shallow.fasta \
    $FASTA_DIR/foldbench_full.fasta > $ALL_FASTA
echo "Total proteins: $(grep -c '^>' $ALL_FASTA)"

# ── GPU 확인 ────────────────────────────────────────────────────────────────────
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

is_done() { [ -s "$1/benchmark_results.csv" ]; }

# ── Stage 1: nomsa — 전체 332개 ─────────────────────────────────────────────────
echo ""
echo "=== Stage 1/2: nomsa — all $(date) ==="
if is_done $BASE_DIR/nomsa_all; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=${HEALTHY_GPUS[$SHARD_ID]} \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $ALL_FASTA \
            --decoder_ckpt   $LATENT_FM_CKPT \
            --latent_fm_ckpt $LATENT_FM_CKPT \
            --output_dir     $BASE_DIR/nomsa_all \
            --mode           nomsa \
            --protenix_model $PROTENIX_MODEL \
            --ref_cif_dir    $REF_CIF_DIR \
            --usalign_bin    $USALIGN_BIN \
            --device         cuda \
            --num_shards     $NUM_SHARDS \
            --shard_id       $SHARD_ID \
            > $BASE_DIR/nomsa_all/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  nomsa_all done: $(date)"
    merge_shards $BASE_DIR/nomsa_all
fi

# ── Stage 2: zeroshot — 전체 332개 ──────────────────────────────────────────────
echo ""
echo "=== Stage 2/2: zeroshot — all $(date) ==="
if is_done $BASE_DIR/zeroshot_all; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=${HEALTHY_GPUS[$SHARD_ID]} \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $ALL_FASTA \
            --decoder_ckpt   $DECODER_CKPT \
            --latent_fm_ckpt $LATENT_FM_CKPT \
            --output_dir     $BASE_DIR/zeroshot_all \
            --mode           zeroshot \
            --protenix_model $PROTENIX_MODEL \
            --ref_cif_dir    $REF_CIF_DIR \
            --usalign_bin    $USALIGN_BIN \
            --device         cuda \
            --num_shards     $NUM_SHARDS \
            --shard_id       $SHARD_ID \
            --n_seqs         $N_SEQS \
            --n_seeds        $N_SEEDS \
            --n_steps        $N_STEPS \
            --temperature    $TEMPERATURE \
            > $BASE_DIR/zeroshot_all/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  zeroshot_all done: $(date)"
    merge_shards $BASE_DIR/zeroshot_all
fi

echo ""
echo "=== 완료: $(date) ==="
echo "결과:"
echo "  nomsa_all   → $BASE_DIR/nomsa_all/benchmark_results.csv"
echo "  zeroshot_all → $BASE_DIR/zeroshot_all/benchmark_results.csv"
