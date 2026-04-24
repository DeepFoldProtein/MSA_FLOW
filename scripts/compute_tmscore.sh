#!/bin/bash
#SBATCH --job-name=msaflow-tmscore
#SBATCH --nodes=1
#SBATCH --nodelist=ada-001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/tmscore_%j.out
#SBATCH --error=logs/tmscore_%j.err

# ── 선행 job 완료 후 실행하려면 ────────────────────────────────────────────────
# sbatch --dependency=afterok:<BENCH_JOB_ID> scripts/compute_tmscore.sh

module load python/3.11.14
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
source $REPO_DIR/.venv/bin/activate

ZEROSHOT_DIR=${ZEROSHOT_DIR:-$REPO_DIR/runs/fold_benchmark}
NOMSA_DIR=${NOMSA_DIR:-$REPO_DIR/runs/fold_benchmark_nomsa}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
USALIGN_BIN=${USALIGN_BIN:-USalign}

mkdir -p $REPO_DIR/logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Ref CIF dir: $REF_CIF_DIR"
echo "USalign    : $(which $USALIGN_BIN 2>/dev/null || echo NOT FOUND)"
date

# ── USalign 없으면 설치 ────────────────────────────────────────────────────────
if ! command -v $USALIGN_BIN &> /dev/null; then
    echo "USalign not found — downloading prebuilt binary..."
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O /tmp/USalign.cpp
    g++ -static -O3 -ffast-math -lm -o $HOME/.local/bin/USalign /tmp/USalign.cpp
    USALIGN_BIN=$HOME/.local/bin/USalign
    echo "Built USalign at $USALIGN_BIN"
fi

# ── zero-shot TM-score ─────────────────────────────────────────────────────────
ZEROSHOT_CSV=$ZEROSHOT_DIR/benchmark_results.csv
if [ -f "$ZEROSHOT_CSV" ]; then
    echo "=== Computing TM-scores: MSAFlow zero-shot ==="
    python $REPO_DIR/scripts/compute_tmscore.py \
        --results_csv  $ZEROSHOT_CSV \
        --fold_dir     $ZEROSHOT_DIR/folds \
        --ref_cif_dir  $REF_CIF_DIR \
        --usalign_bin  $USALIGN_BIN \
        --mode         zeroshot
else
    echo "WARNING: $ZEROSHOT_CSV not found — skipping zero-shot TM-score"
fi

# ── no-MSA baseline TM-score ──────────────────────────────────────────────────
NOMSA_CSV=$NOMSA_DIR/benchmark_results.csv
if [ -f "$NOMSA_CSV" ]; then
    echo "=== Computing TM-scores: No-MSA baseline ==="
    python $REPO_DIR/scripts/compute_tmscore.py \
        --results_csv  $NOMSA_CSV \
        --fold_dir     $NOMSA_DIR/folds \
        --ref_cif_dir  $REF_CIF_DIR \
        --usalign_bin  $USALIGN_BIN \
        --mode         nomsa
else
    echo "WARNING: $NOMSA_CSV not found — skipping no-MSA TM-score"
fi

# ── 최종 비교 출력 ─────────────────────────────────────────────────────────────
python - << 'PYEOF'
import csv, os, math

def summarize(path, label):
    if not os.path.exists(path):
        print(f"[{label}] CSV not found: {path}")
        return
    rows = list(csv.DictReader(open(path)))
    ok = [r for r in rows if r.get("status") == "ok"]

    def avg(key):
        vals = []
        for r in ok:
            try:
                v = float(r.get(key, "nan"))
                if not math.isnan(v):
                    vals.append(v)
            except ValueError:
                pass
        return sum(vals)/len(vals) if vals else float("nan"), len(vals)

    plddt_mean, n_p = avg("plddt")
    tm_mean,    n_t = avg("tm_score")
    print(f"\n[{label}]  n={len(ok)}/{len(rows)}")
    print(f"  pLDDT   mean={plddt_mean:.2f}  (n={n_p})")
    print(f"  TM-score mean={tm_mean:.4f}  (n={n_t})")

zeroshot_csv = os.environ.get("ZEROSHOT_DIR", "runs/fold_benchmark") + "/benchmark_results.csv"
nomsa_csv    = os.environ.get("NOMSA_DIR",    "runs/fold_benchmark_nomsa") + "/benchmark_results.csv"

print("\n" + "="*60)
print("FOLD BENCHMARK COMPARISON")
print("="*60)
summarize(zeroshot_csv, "MSAFlow zero-shot")
summarize(nomsa_csv,    "No-MSA baseline")
print("="*60)
PYEOF

echo "All done: $(date)"
