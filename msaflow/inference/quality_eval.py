"""
MSAFlow MSA Quality Evaluation — no structure prediction required.

Samples N proteins directly from the LMDB (ground-truth msa_tokens and
esm_emb already stored), generates MSAs in zero-shot mode using the
trained Latent FM + SFM Decoder, then computes:

  Neff ratio          generated Neff / reference Neff   (diversity)
  gap_frac            fraction of gap tokens in generated seqs
  mean_diversity      mean pairwise Hamming distance between generated seqs
  seq_recovery        mean best-match identity to any reference sequence
  aa_kl_div           KL divergence of AA marginal distribution vs reference

Results are printed as a table and saved to a CSV.

Usage:
    python msaflow/inference/quality_eval.py \\
        --lmdb_path     /gpfs/deepfold/users/yjlee4/msaflow_merged.lmdb \\
        --decoder_ckpt  runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir    runs/quality_eval \\
        --n_proteins    50 \\
        [--device cuda]

Pro tip: if n_proteins is small (≤10) you can pass --verbose to print
the generated sequences for each protein.
"""

import argparse
import csv
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

GAP_IDX = 20      # from AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
VOCAB_SIZE = 22


def compute_neff(tokens: np.ndarray, threshold: float = 0.2) -> float:
    """Neff = sum of sequence weights (1 / cluster size at 80% identity)."""
    N = tokens.shape[0]
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    diff = (tokens[:, None, :] != tokens[None, :, :])
    hamming = diff.mean(axis=-1)
    counts = (hamming < threshold).sum(axis=1).astype(np.float32)
    return float((1.0 / counts.clip(min=1.0)).sum())


def gap_fraction(tokens: np.ndarray) -> float:
    return float((tokens == GAP_IDX).mean())


def mean_pairwise_diversity(tokens: np.ndarray) -> float:
    """Mean Hamming distance between all pairs."""
    N = tokens.shape[0]
    if N < 2:
        return 0.0
    diff = (tokens[:, None, :] != tokens[None, :, :])
    hamming = diff.mean(axis=-1)
    # upper-triangle only (exclude self)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float(hamming[mask].mean())


def seq_recovery(gen_tokens: np.ndarray, ref_tokens: np.ndarray) -> float:
    """
    For each generated sequence, find its best-matching reference sequence.
    Return the mean of those best-match identities.
    """
    if ref_tokens.shape[0] == 0 or gen_tokens.shape[0] == 0:
        return 0.0
    L = min(gen_tokens.shape[1], ref_tokens.shape[1])
    gen = gen_tokens[:, :L]
    ref = ref_tokens[:, :L]
    # (N_gen, N_ref) identity matrix
    identity = (gen[:, None, :] == ref[None, :, :]).mean(axis=-1)
    best_match = identity.max(axis=1)    # (N_gen,) best ref match per gen seq
    return float(best_match.mean())


def aa_kl_divergence(gen_tokens: np.ndarray, ref_tokens: np.ndarray,
                      vocab: int = VOCAB_SIZE, eps: float = 1e-8) -> float:
    """KL divergence of AA marginal distribution: KL(gen || ref)."""
    def marginal(toks):
        counts = np.bincount(toks.flatten(), minlength=vocab).astype(float)
        return (counts + eps) / (counts.sum() + eps * vocab)

    p = marginal(gen_tokens)
    q = marginal(ref_tokens)
    return float((p * np.log(p / q)).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def seqs_to_tokens(seqs: list[str]) -> np.ndarray:
    """Convert list of AA strings to (N, L) int array using our vocab."""
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from msaflow.data.preprocessing import AA_TO_IDX
    if not seqs:
        return np.zeros((0, 0), dtype=np.int32)
    L = max(len(s) for s in seqs)
    arr = np.full((len(seqs), L), AA_TO_IDX.get("X", 21), dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, aa in enumerate(s):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), AA_TO_IDX.get("X", 21))
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# LMDB sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_lmdb_entries(
    lmdb_path: str,
    n: int,
    seed: int = 42,
    require_esm: bool = True,
    require_msa_emb: bool = False,
) -> list[dict]:
    """
    Sample n entries from the LMDB and return them as dicts with tensors.

    Returned dict keys (subset of LMDB schema):
        esm_emb     (L, 1280) float16 → float32
        msa_tokens  (N, L) int32
        weights     (N,) float32
        query_seq   str
        seq_len     int
    """
    import lmdb as lmdb_lib

    env = lmdb_lib.open(lmdb_path, readonly=True, lock=False,
                        readahead=False, meminit=False, subdir=False)

    with env.begin() as txn:
        all_keys = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)]

    logger.info("LMDB has %d entries, sampling %d ...", len(all_keys), n)
    rng = random.Random(seed)
    sampled_keys = rng.sample(all_keys, min(n * 3, len(all_keys)))  # over-sample to allow filtering

    entries = []
    with env.begin() as txn:
        for key in sampled_keys:
            if len(entries) >= n:
                break
            raw = txn.get(key.encode())
            if raw is None:
                continue
            entry = pickle.loads(raw)

            if require_esm and entry.get("esm_emb") is None:
                continue
            if require_msa_emb and entry.get("msa_emb") is None:
                continue

            entry["_key"] = key
            entries.append(entry)

    env.close()
    logger.info("Loaded %d entries", len(entries))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_eval(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from msaflow.inference.generate import load_sfm_decoder, load_latent_fm
    from msaflow.models.latent_fm import sample_msa_embeddings
    from msaflow.inference.generate import decode_from_embedding

    logger.info("Loading SFM decoder from %s", args.decoder_ckpt)
    decoder = load_sfm_decoder(args.decoder_ckpt, device)
    logger.info("Loading Latent FM from %s", args.latent_fm_ckpt)
    latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
    logger.info("Models loaded.")

    # ── Sample LMDB entries ────────────────────────────────────────────────────
    entries = sample_lmdb_entries(
        lmdb_path=args.lmdb_path,
        n=args.n_proteins,
        seed=args.seed,
        require_esm=True,
        require_msa_emb=False,   # zero-shot: we don't need pre-computed msa_emb
    )

    results = []

    for idx, entry in enumerate(entries):
        key = entry["_key"]
        query_seq = entry.get("query_seq", "")
        L = entry.get("seq_len", len(query_seq))

        logger.info("─" * 50)
        logger.info("[%d/%d] %s  L=%d", idx + 1, len(entries), key, L)

        # ── Reference MSA ──────────────────────────────────────────────────────
        ref_tokens = entry.get("msa_tokens")   # (N, L) int32
        if ref_tokens is None:
            logger.warning("  No msa_tokens, skipping reference metrics")
            ref_tokens = np.zeros((0, L), dtype=np.int32)
        else:
            ref_tokens = ref_tokens[:, :L].astype(np.int32)

        ref_neff = compute_neff(ref_tokens) if ref_tokens.shape[0] > 0 else 0.0
        logger.info("  Reference: N=%d  Neff=%.1f  gap_frac=%.3f",
                    ref_tokens.shape[0], ref_neff, gap_fraction(ref_tokens))

        # ── ESM2 embedding (from LMDB — no model inference needed) ────────────
        esm_np = entry["esm_emb"].astype(np.float32)[:L]   # (L, 1280)
        esm_emb = torch.from_numpy(esm_np).unsqueeze(0).to(device)   # (1, L, 1280)

        # ── Generate MSA embeddings via Latent FM ──────────────────────────────
        with torch.no_grad():
            z_syn = sample_msa_embeddings(
                latent_fm, esm_emb,
                n_steps=args.n_steps,
                temperature=args.temperature,
            )  # (1, L, 128)

        # ── Decode sequences via SFM Decoder ───────────────────────────────────
        gen_seqs = decode_from_embedding(
            decoder, z_syn[0].cpu(),
            n_seqs=args.n_seqs,
            n_steps=args.n_steps,
            device=device,
        )
        gen_tokens = seqs_to_tokens(gen_seqs)   # (n_seqs, L)

        # ── Metrics ────────────────────────────────────────────────────────────
        gen_neff = compute_neff(gen_tokens)
        neff_ratio = gen_neff / max(ref_neff, 1e-6)
        g_frac = gap_fraction(gen_tokens)
        diversity = mean_pairwise_diversity(gen_tokens)
        recovery = seq_recovery(gen_tokens, ref_tokens) if ref_tokens.shape[0] > 0 else float("nan")
        kl = aa_kl_divergence(gen_tokens, ref_tokens) if ref_tokens.shape[0] > 0 else float("nan")

        logger.info("  Generated : N=%d  Neff=%.1f  Neff_ratio=%.3f",
                    len(gen_seqs), gen_neff, neff_ratio)
        logger.info("  gap_frac=%.3f  diversity=%.3f  recovery=%.3f  aa_kl=%.4f",
                    g_frac, diversity, recovery, kl)

        if args.verbose and len(gen_seqs) > 0:
            logger.info("  Sample generated seqs:")
            for i, s in enumerate(gen_seqs[:3]):
                logger.info("    [%d] %s", i, s[:80] + ("..." if len(s) > 80 else ""))

        results.append({
            "key": key,
            "seq_len": L,
            "ref_n_seqs": ref_tokens.shape[0],
            "ref_neff": ref_neff,
            "gen_neff": gen_neff,
            "neff_ratio": neff_ratio,
            "gap_frac": g_frac,
            "mean_diversity": diversity,
            "seq_recovery": recovery,
            "aa_kl_div": kl,
        })

    # ── CSV output ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "quality_eval.csv"
    fields = ["key", "seq_len", "ref_n_seqs", "ref_neff", "gen_neff",
              "neff_ratio", "gap_frac", "mean_diversity", "seq_recovery", "aa_kl_div"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results written to %s", csv_path)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    def _stats(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}"

    print("\n" + "=" * 65)
    print(f"MSA Quality Evaluation  ({len(results)} proteins)")
    print("=" * 65)
    print(f"  Neff ratio      {_stats([r['neff_ratio']      for r in results])}")
    print(f"  Gap fraction    {_stats([r['gap_frac']        for r in results])}")
    print(f"  Diversity       {_stats([r['mean_diversity']  for r in results])}")
    print(f"  Seq recovery    {_stats([r['seq_recovery']    for r in results])}")
    print(f"  AA KL div       {_stats([r['aa_kl_div']       for r in results])}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MSAFlow MSA quality evaluation")
    parser.add_argument("--lmdb_path",      required=True,
                        help="Path to merged LMDB")
    parser.add_argument("--decoder_ckpt",   required=True,
                        help="SFM decoder EMA checkpoint")
    parser.add_argument("--latent_fm_ckpt", required=True,
                        help="Latent FM EMA checkpoint")
    parser.add_argument("--output_dir",     default="runs/quality_eval")
    parser.add_argument("--n_proteins",     type=int, default=50,
                        help="Number of proteins to evaluate")
    parser.add_argument("--n_seqs",         type=int, default=32,
                        help="Generated sequences per protein")
    parser.add_argument("--n_steps",        type=int, default=100,
                        help="ODE integration steps")
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="SDE temperature (0 = deterministic ODE)")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--verbose",        action="store_true",
                        help="Print sample generated sequences per protein")
    args = parser.parse_args()

    run_quality_eval(args)


if __name__ == "__main__":
    main()
