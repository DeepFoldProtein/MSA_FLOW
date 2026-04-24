#!/bin/bash
#SBATCH --job-name=msaflow-nomsa
#SBATCH --nodes=1
#SBATCH --nodelist=ada-003
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/nomsa_%j.out
#SBATCH --error=logs/nomsa_%j.err

# ── Python / CUDA 모듈 로드 ────────────────────────────────────────────────────
module load python/3.11.14
module load cuda/13.0.2

# ── CUDA_HOME 설정 ─────────────────────────────────────────────────────────────
export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ── uv PATH ───────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-$REPO_DIR/runs/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
FASTA=${FASTA:-$REPO_DIR/data/foldbench_monomer.fasta}
BASELINE_DIR=${BASELINE_DIR:-$REPO_DIR/runs/fold_benchmark_nomsa}
REF_PDB_DIR=${REF_PDB_DIR:-}
TMSCORE_BIN=${TMSCORE_BIN:-TMscore}

PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

# ── 환경 활성화 ────────────────────────────────────────────────────────────────
source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH

# ── 로그 디렉터리 ──────────────────────────────────────────────────────────────
mkdir -p $BASELINE_DIR $REPO_DIR/logs

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Python        : $(python --version)"
echo "FASTA         : $FASTA"
echo "Baseline dir  : $BASELINE_DIR"
echo "Protenix model: $PROTENIX_MODEL"
date

NUM_SHARDS=2

REF_ARG=""
if [ -n "$REF_PDB_DIR" ]; then
    REF_ARG="--ref_pdb_dir $REF_PDB_DIR"
fi

# ── No-MSA baseline (2 GPU 병렬) ───────────────────────────────────────────────
echo "=== Launching No-MSA baseline shards ==="
for SHARD_ID in 0 1; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID \
    python $REPO_DIR/msaflow/inference/fold_benchmark.py \
        --fasta          $FASTA \
        --decoder_ckpt   $DECODER_CKPT \
        --latent_fm_ckpt $LATENT_FM_CKPT \
        --output_dir     $BASELINE_DIR \
        --mode           nomsa \
        --device         cuda \
        --protenix_model $PROTENIX_MODEL \
        --tmscore_bin    $TMSCORE_BIN \
        --shard_id       $SHARD_ID \
        --num_shards     $NUM_SHARDS \
        $REF_ARG \
        > $BASELINE_DIR/shard_${SHARD_ID}.log 2>&1 &
done

echo "Launched No-MSA baseline shards (PIDs: $(jobs -p))"
wait
echo "No-MSA baseline done: $(date)"

# ── 결과 병합 ──────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import csv, glob, os

output_dir = os.environ.get("BASELINE_DIR", "runs/fold_benchmark_nomsa")
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
    print(f"[No-MSA] Merged {len(rows)} rows → {out_path}")
else:
    print(f"[No-MSA] No shard CSVs found in {output_dir}")
PYEOF

echo "All done: $(date)"
