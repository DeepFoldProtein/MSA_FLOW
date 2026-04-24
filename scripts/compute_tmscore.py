"""
Post-hoc TM-score computation for fold benchmark results.

Reads benchmark_results.csv, finds each protein's predicted CIF, aligns
it against the ground truth CIF, and writes updated CSV with tm_score/rmsd.

Usage:
    python scripts/compute_tmscore.py \\
        --results_csv    runs/fold_benchmark/benchmark_results.csv \\
        --fold_dir       runs/fold_benchmark/folds \\
        --ref_cif_dir    /gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520 \\
        --usalign_bin    USalign \\
        [--mode          zeroshot]   # zeroshot | nomsa

    python scripts/compute_tmscore.py \\
        --results_csv    runs/fold_benchmark_nomsa/benchmark_results.csv \\
        --fold_dir       runs/fold_benchmark_nomsa/folds \\
        --ref_cif_dir    /gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520 \\
        --mode           nomsa
"""

import argparse
import csv
import glob
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def find_predicted_cif(fold_dir: Path, prot_name: str, best_seed: int, mode: str) -> Path | None:
    """Locate the Protenix output CIF for a protein."""
    if mode == "zeroshot" and best_seed >= 0:
        # e.g. folds/7qp5-assembly1_seed3/protenix_out/.../*.cif
        run_name = f"{prot_name}_seed{best_seed}"
    else:
        # nomsa / fewshot: folds/7qp5-assembly1/protenix_out/.../*.cif
        run_name = prot_name

    search_dir = fold_dir / run_name / "protenix_out"
    matches = sorted(search_dir.rglob("*.cif")) if search_dir.exists() else []
    return matches[0] if matches else None


def find_ref_cif(ref_cif_dir: Path, prot_name: str) -> Path | None:
    """Find ground truth CIF by PDB ID (strip assembly suffix if needed)."""
    # FoldBench names are like "7qp5-assembly1" — try with and without suffix
    for name in [prot_name, prot_name.split("-")[0]]:
        for ext in [".cif", ".cif.gz"]:
            p = ref_cif_dir / f"{name}{ext}"
            if p.exists():
                return p
        # Case-insensitive glob fallback
        matches = list(ref_cif_dir.glob(f"{name}*.[Cc][Ii][Ff]*"))
        if matches:
            return matches[0]
    return None


def run_usalign(pred_cif: Path, ref_cif: Path, usalign_bin: str) -> tuple[float, float]:
    """Run USalign and parse TM-score (normalized by ref length) and RMSD."""
    try:
        result = subprocess.run(
            [usalign_bin, str(pred_cif), str(ref_cif), "-outfmt", "2"],
            capture_output=True, text=True, timeout=120,
        )
        tm_score, rmsd = float("nan"), float("nan")
        for line in result.stdout.splitlines():
            # -outfmt 2 tab-separated: Name1 Name2 TM1 TM2 RMSD Seq_ID Len1 Len2 ...
            parts = line.split()
            if len(parts) >= 5 and not line.startswith("#"):
                try:
                    tm_score = float(parts[3])   # TM-score normalized by ref (target) length
                    rmsd     = float(parts[4])
                    break
                except ValueError:
                    continue
        if result.returncode != 0 and float("nan") == tm_score:
            logger.warning("USalign failed (exit %d):\n%s", result.returncode, result.stderr[:500])
        return tm_score, rmsd
    except FileNotFoundError:
        logger.error("USalign binary not found: %s", usalign_bin)
        return float("nan"), float("nan")
    except subprocess.TimeoutExpired:
        logger.warning("USalign timed out for %s", pred_cif)
        return float("nan"), float("nan")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Post-hoc TM-score for fold benchmark results")
    parser.add_argument("--results_csv",  required=True, help="benchmark_results.csv to update")
    parser.add_argument("--fold_dir",     required=True, help="folds/ directory from the run")
    parser.add_argument("--ref_cif_dir",  required=True, help="Ground truth CIF directory")
    parser.add_argument("--usalign_bin",  default="USalign", help="Path to USalign binary")
    parser.add_argument("--mode",         default="zeroshot", choices=["zeroshot", "nomsa", "fewshot"])
    parser.add_argument("--output_csv",   default=None,
                        help="Output CSV path (default: overwrite input)")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    fold_dir    = Path(args.fold_dir)
    ref_cif_dir = Path(args.ref_cif_dir)
    output_csv  = Path(args.output_csv) if args.output_csv else results_csv

    with open(results_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        logger.error("No rows in %s", results_csv)
        return

    fieldnames = list(rows[0].keys())
    if "tm_score" not in fieldnames:
        fieldnames += ["tm_score", "rmsd"]

    ok = skipped = failed = 0

    for row in rows:
        prot_name = row["name"]
        best_seed = int(row.get("best_seed", -1))

        # Skip if already computed
        if row.get("tm_score") and row["tm_score"] != "nan":
            try:
                if not float("nan") == float(row["tm_score"]):
                    logger.info("%s  TM-score already present (%.4f), skipping",
                                prot_name, float(row["tm_score"]))
                    ok += 1
                    continue
            except ValueError:
                pass

        pred_cif = find_predicted_cif(fold_dir, prot_name, best_seed, args.mode)
        if pred_cif is None:
            logger.warning("%s  predicted CIF not found", prot_name)
            row["tm_score"] = "nan"
            row["rmsd"]     = "nan"
            skipped += 1
            continue

        ref_cif = find_ref_cif(ref_cif_dir, prot_name)
        if ref_cif is None:
            logger.warning("%s  reference CIF not found in %s", prot_name, ref_cif_dir)
            row["tm_score"] = "nan"
            row["rmsd"]     = "nan"
            skipped += 1
            continue

        logger.info("%s  pred=%s  ref=%s", prot_name, pred_cif.name, ref_cif.name)
        tm_score, rmsd = run_usalign(pred_cif, ref_cif, args.usalign_bin)
        row["tm_score"] = f"{tm_score:.6f}" if tm_score == tm_score else "nan"
        row["rmsd"]     = f"{rmsd:.4f}"     if rmsd == rmsd else "nan"

        if tm_score == tm_score:
            logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)
            ok += 1
        else:
            failed += 1

    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Done — ok=%d  skipped=%d  failed=%d  → %s", ok, skipped, failed, output_csv)

    # Print summary
    tm_scores = []
    for r in rows:
        try:
            v = float(r.get("tm_score", "nan"))
            if v == v:
                tm_scores.append(v)
        except ValueError:
            pass

    if tm_scores:
        import statistics
        print(f"\nTM-score  n={len(tm_scores)}")
        print(f"  mean  = {statistics.mean(tm_scores):.4f}")
        print(f"  median= {statistics.median(tm_scores):.4f}")
        print(f"  min   = {min(tm_scores):.4f}")
        print(f"  max   = {max(tm_scores):.4f}")


if __name__ == "__main__":
    main()
