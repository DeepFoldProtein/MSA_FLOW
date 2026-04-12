"""
MSAFlow data preprocessing pipeline.

Two preprocessing stages:

1. extract_msa_embeddings()
   Runs the Protenix MSAModule over OpenFold MSA files and dumps the
   compressed pair representation (L×128) to an LMDB database.
   This corresponds to the AF3-encoder step described in Section 3.1 and 6.8.1.

2. extract_esm_embeddings()
   Computes ESM2-650M representations (L×1280) for query sequences and stores
   them alongside the MSA embeddings.

Both functions write to the same LMDB, keyed by protein/MSA ID.

LMDB schema per entry (pickled dict):
  {
    "msa_emb":     torch.Tensor  [L, 128]   – compressed MSA embedding
    "esm_emb":     torch.Tensor  [L, 1280]  – ESM2 query embedding
    "msa_tokens":  torch.Tensor  [N, L]     – integer token ids (0..21)
    "weights":     torch.Tensor  [N]        – Neff reweighting
    "query_seq":   str                      – query sequence (no gaps)
    "seq_len":     int                      – L
  }
"""

import os
import sys
import pickle
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lmdb
from tqdm import tqdm

# ── Alphabet shared between MSA tokenisation and SFM decoder ─────────────────
# 20 canonical AAs (ACDEFGHIKLMNPQRSTVWY), gap '-', unknown 'X'
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
VOCAB_SIZE = len(AA_LIST)  # 22

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing statistics tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessStats:
    """Accumulates per-entry statistics for end-of-run summary."""
    n_total: int = 0
    n_written: int = 0
    skip_reasons: dict = field(default_factory=lambda: defaultdict(int))
    seq_lens: list = field(default_factory=list)
    msa_depths: list = field(default_factory=list)
    neff_values: list = field(default_factory=list)
    t_start: float = field(default_factory=time.time)

    def skip(self, reason: str) -> None:
        self.skip_reasons[reason] += 1

    def record(self, seq_len: int, msa_depth: int, weights: np.ndarray) -> None:
        self.n_written += 1
        self.seq_lens.append(seq_len)
        self.msa_depths.append(msa_depth)
        self.neff_values.append(float(weights.sum()))

    def eta_str(self, file_idx: int) -> str:
        elapsed = time.time() - self.t_start
        if file_idx == 0:
            return "N/A"
        rate = file_idx / elapsed          # files/sec
        remaining = (self.n_total - file_idx) / max(rate, 1e-9)
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log_progress(self, file_idx: int) -> None:
        elapsed = time.time() - self.t_start
        rate = file_idx / max(elapsed, 1e-9)
        gpu_mem = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem = f"  GPU {alloc:.1f}/{reserved:.1f}GB"
        logger.info(
            "Progress  [%d/%d]  written=%d  skip=%d  "
            "%.1f files/s  ETA=%s%s",
            file_idx, self.n_total,
            self.n_written,
            sum(self.skip_reasons.values()),
            rate,
            self.eta_str(file_idx),
            gpu_mem,
        )

    def log_summary(self) -> None:
        elapsed = time.time() - self.t_start
        n_skip = sum(self.skip_reasons.values())
        logger.info("=" * 60)
        logger.info("Preprocessing complete")
        logger.info("  Total files     : %d", self.n_total)
        logger.info("  Written         : %d  (%.1f%%)",
                    self.n_written, 100 * self.n_written / max(self.n_total, 1))
        logger.info("  Skipped         : %d", n_skip)
        for reason, count in sorted(self.skip_reasons.items(),
                                     key=lambda x: -x[1]):
            logger.info("    %-30s : %d", reason, count)
        logger.info("  Elapsed         : %.1f s", elapsed)
        if self.seq_lens:
            sl = np.array(self.seq_lens)
            md = np.array(self.msa_depths)
            neff = np.array(self.neff_values)
            logger.info("  seq_len   mean=%.0f  med=%.0f  min=%d  max=%d",
                        sl.mean(), np.median(sl), sl.min(), sl.max())
            logger.info("  msa_depth mean=%.0f  med=%.0f  min=%d  max=%d",
                        md.mean(), np.median(md), md.min(), md.max())
            logger.info("  Neff      mean=%.1f  med=%.1f  min=%.1f  max=%.1f",
                        neff.mean(), np.median(neff), neff.min(), neff.max())
        logger.info("=" * 60)


def _log_entry_shapes(key: str, entry: dict, verbose: bool) -> None:
    """Log shapes of a single LMDB entry when verbose=True."""
    if not verbose:
        return
    shapes = {
        k: (v.shape if isinstance(v, np.ndarray) else
            (v.shape if isinstance(v, torch.Tensor) else type(v).__name__))
        for k, v in entry.items()
    }
    logger.debug(
        "  [%s]  seq_len=%d  msa_depth=%d  "
        "esm_emb=%s  msa_emb=%s",
        key,
        entry["seq_len"],
        entry["msa_tokens"].shape[0],
        shapes.get("esm_emb", "None"),
        shapes.get("msa_emb", "None"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# MSA I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_a3m(path: str) -> tuple[list[str], list[str]]:
    """Parse A3M file; remove insertion columns (lowercase letters)."""
    names, seqs = [], []
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(buf))
                name = line[1:]
                names.append(name)
                buf = []
            else:
                buf.append("".join(c for c in line if c.isupper() or c == "-"))
        if name is not None:
            seqs.append("".join(buf))
    return names, seqs


def filter_msa(seqs: list[str], max_gap_frac: float = 0.1, min_seqs: int = 10) -> list[str]:
    """Keep sequences where at most max_gap_frac of columns are gaps."""
    if not seqs:
        return seqs
    L = len(seqs[0])
    filtered = [s for s in seqs if s.count("-") / max(L, 1) <= max_gap_frac]
    return filtered if len(filtered) >= min_seqs else seqs[:min_seqs]


def tokenise_msa(seqs: list[str]) -> np.ndarray:
    """Convert list of aligned sequences to integer array (N, L)."""
    N = len(seqs)
    L = len(seqs[0])
    arr = np.zeros((N, L), dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"])
    return arr


def compute_sequence_weights(tokens: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Neff reweighting: w_i = 1 / |{j : hamming(i,j) < threshold}|

    This is the scheme used in the paper (Section 6.8.2):
        w_i = (1 + Σ_{j≠i} 1{d_hamming(x_i, x_j) < 0.2})^{-1}

    Vectorised implementation: O(N²·L) via broadcasting — ~100× faster than
    the naive Python double loop.
    """
    N, _ = tokens.shape
    if N == 1:
        return np.ones(1, dtype=np.float32)
    # (N, N) pairwise fractional Hamming distances
    # tokens: (N, L) int → broadcast diff → mean over L axis
    diff = (tokens[:, None, :] != tokens[None, :, :])   # (N, N, L) bool
    hamming = diff.mean(axis=-1)                         # (N, N) float
    similar = hamming < threshold                        # (N, N) bool
    counts = similar.sum(axis=1).astype(np.float32)      # includes self (hamming=0)
    return 1.0 / counts.clip(min=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# MSA embedding extraction via Protenix
# ──────────────────────────────────────────────────────────────────────────────

def _build_protenix_msa_input(seqs: list[str], device: torch.device) -> dict:
    """
    Build minimal input_feature_dict for the Protenix MSAModule.

    The MSAModule (Algorithm 8 in AF3) expects:
      - msa:             (1, N_msa, L) int64  — amino acid indices (0..31 = AF3 vocab)
      - has_deletion:    (1, N_msa, L) float
      - deletion_value:  (1, N_msa, L) float

    We map our 22-token alphabet to AF3's 32-token residue encoding.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3] / "Protenix"))
    from protenix.data.constants import STD_RESIDUES_WITH_GAP

    # Build AF3 residue → index mapping
    af3_res_to_idx = {r: i for i, r in enumerate(STD_RESIDUES_WITH_GAP)}
    # Map our tokens to AF3 indices (unknown → last index 31)
    unk_idx = len(STD_RESIDUES_WITH_GAP) - 1

    N = len(seqs)
    L = len(seqs[0])
    msa_arr = np.zeros((N, L), dtype=np.int64)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            msa_arr[i, j] = af3_res_to_idx.get(aa.upper(), unk_idx)

    msa_t = torch.from_numpy(msa_arr).unsqueeze(0).to(device)   # (1, N, L)
    zeros = torch.zeros(1, N, L, device=device)

    return {
        "msa": msa_t,
        "has_deletion": zeros,
        "deletion_value": zeros,
    }


def extract_msa_embedding_protenix(
    seqs: list[str],
    protenix_model,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract compressed MSA embedding using Protenix's MSAModule.

    Steps (mirrors Section 3.1 & 6.8.1 of the paper):
      1. Build AF3-format input features from MSA sequences.
      2. Run the MSAModule to obtain pair representation P ∈ R^(L×L×128).
      3. Mean-pool along the second spatial dimension → m_seq ∈ R^(L×128).

    Args:
        seqs:            List of aligned sequences (all same length L).
        protenix_model:  Loaded Protenix model (nn.Module) with .msa_module attribute.
        device:          Compute device.

    Returns:
        m_seq: (L, 128) compressed MSA embedding.
    """
    L = len(seqs[0])
    logger.info("Extracting MSA embedding  ----- %d seqs, L=%d", len(seqs), L)

    # Build dummy pair embedding z and single embedding s_inputs
    # (MSAModule updates z; we start from zeros for embedding extraction)
    c_z = 128
    c_s = 449
    z = torch.zeros(1, L, L, c_z, device=device)
    s_inputs = torch.zeros(1, L, c_s, device=device)
    pair_mask = torch.ones(1, L, L, device=device, dtype=torch.bool)

    feat = _build_protenix_msa_input(seqs, device)

    with torch.no_grad():
        z_updated = protenix_model.msa_module(
            input_feature_dict=feat,
            z=z,
            s_inputs=s_inputs,
            pair_mask=pair_mask,
        )  # (1, L, L, 128)

    # Mean-pool along second spatial dimension (Eq. 4 in paper)
    m_seq = z_updated[0].mean(dim=1)   # (L, 128)
    return m_seq.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# ESM2 embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_esm_embedding(
    query_seq: str,
    esm_model,
    alphabet,
    device: torch.device,
    layer: int = 33,
) -> torch.Tensor:
    """
    Extract ESM2-650M representation for a single query sequence.

    Args:
        query_seq: Amino acid sequence (no gaps), length L.
        esm_model: Loaded ESM2 model.
        alphabet:  ESM2 Alphabet object.
        device:    Compute device.
        layer:     Representation layer (33 for ESM2-650M).

    Returns:
        emb: (L, 1280) ESM2 representation (BOS/EOS tokens removed).
    """
    logger.info("Extracting ESM2 embedding  ----- query length=%d", len(query_seq))
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("query", query_seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[layer], return_contacts=False)

    emb = out["representations"][layer][0, 1:-1, :]  # remove BOS/EOS → (L, 1280)
    return emb.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main LMDB building routine
# ──────────────────────────────────────────────────────────────────────────────

def build_lmdb(
    a3m_dir: str,
    output_path: str,
    protenix_checkpoint: Optional[str] = None,
    max_msa_seqs: int = 512,
    max_seq_len: int = 1024,
    device: str = "cuda",
    map_size_gb: int = 500,
    log_every: int = 100,
    verbose: bool = False,
):
    """
    Process all A3M files in a3m_dir and write LMDB database.

    Args:
        a3m_dir:              Directory containing *.a3m files.
        output_path:          Path for the output LMDB.
        protenix_checkpoint:  Path to Protenix checkpoint. If None, skip MSA emb.
        max_msa_seqs:         Maximum MSA depth to use.
        max_seq_len:          Maximum sequence length.
        device:               Compute device string.
        map_size_gb:          LMDB map size in GB.
        log_every:            Log progress every N files.
        verbose:              Log per-entry shapes at DEBUG level.
    """
    logger.info("build_lmdb started")
    logger.info("  a3m_dir    : %s", a3m_dir)
    logger.info("  output     : %s", output_path)
    logger.info("  max_seq_len: %d  max_msa_seqs: %d", max_seq_len, max_msa_seqs)
    logger.info("  device     : %s  map_size: %dGB", device, map_size_gb)
    logger.info("  protenix   : %s", protenix_checkpoint or "disabled")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load Protenix if checkpoint provided
    protenix_model = None
    if protenix_checkpoint is not None:
        logger.info("Loading Protenix from %s ...", protenix_checkpoint)
        sys.path.insert(0, str(Path(__file__).parents[3] / "Protenix"))
        from protenix.model.protenix import Protenix
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(Path(protenix_checkpoint).parent / "config.yaml")
        protenix_model = Protenix(cfg).eval().to(dev)
        state = torch.load(protenix_checkpoint, map_location=dev)
        missing, unexpected = protenix_model.load_state_dict(state["model"], strict=False)
        logger.info("Protenix loaded  (missing=%d  unexpected=%d)", len(missing), len(unexpected))
        if missing:
            logger.debug("Missing keys: %s", missing[:5])

    # Load ESM2-650M
    logger.info("Loading ESM2-650M ...")
    sys.path.insert(0, str(Path(__file__).parents[3] / "esm"))
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(dev)
    if torch.cuda.is_available():
        logger.info("ESM2 loaded  GPU mem: %.1f GB allocated",
                    torch.cuda.memory_allocated() / 1024**3)
    else:
        logger.info("ESM2 loaded  (CPU mode)")

    a3m_files = list(Path(a3m_dir).glob("**/*.a3m"))
    logger.info("Found %d A3M files in %s", len(a3m_files), a3m_dir)
    if not a3m_files:
        logger.error("No .a3m files found — check a3m_dir path")
        return

    env = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    stats = PreprocessStats(n_total=len(a3m_files))

    for file_idx, a3m_path in enumerate(tqdm(a3m_files, desc="Building LMDB")):
        key = a3m_path.stem
        try:
            # ── Parse ────────────────────────────────────────────────────────
            names, seqs = parse_a3m(str(a3m_path))
            if not seqs:
                stats.skip("empty_file")
                continue

            n_seqs_raw = len(seqs)
            seqs = filter_msa(seqs)
            seqs = [s[:max_seq_len] for s in seqs]
            L = len(seqs[0])

            if L == 0:
                stats.skip("zero_length")
                continue
            if L > max_seq_len:
                stats.skip("seq_too_long")
                continue

            query_seq = seqs[0].replace("-", "")
            L_query = len(query_seq)
            if L_query == 0:
                stats.skip("query_all_gaps")
                continue

            seqs = [s[:L_query] for s in seqs]
            tokens = tokenise_msa(seqs[:max_msa_seqs])   # (N, L_query)
            weights = compute_sequence_weights(tokens)    # (N,)
            N = tokens.shape[0]

            logger.debug(
                "[%s] raw_seqs=%d  after_filter=%d  used=%d  L_aligned=%d  L_query=%d  Neff=%.1f",
                key, n_seqs_raw, len(seqs), N, L, L_query, weights.sum(),
            )

            entry = {
                "msa_tokens": tokens,
                "weights": weights,
                "query_seq": query_seq,
                "seq_len": L_query,
                "msa_emb": None,
                "esm_emb": None,
            }

            # ── ESM2 embedding ───────────────────────────────────────────────
            entry["esm_emb"] = extract_esm_embedding(
                query_seq, esm_model, alphabet, dev
            ).half().numpy()
            esm_shape = entry["esm_emb"].shape
            if esm_shape[0] != L_query:
                stats.skip("esm_length_mismatch")
                logger.warning("[%s] ESM2 shape %s != L_query %d", key, esm_shape, L_query)
                continue

            # ── Protenix MSA embedding ───────────────────────────────────────
            if protenix_model is not None:
                entry["msa_emb"] = extract_msa_embedding_protenix(
                    seqs[:max_msa_seqs], protenix_model, dev
                ).half().numpy()
                msa_shape = entry["msa_emb"].shape
                if msa_shape[0] != L_query:
                    stats.skip("msa_emb_length_mismatch")
                    logger.warning("[%s] msa_emb shape %s != L_query %d", key, msa_shape, L_query)
                    continue

            # ── Write ────────────────────────────────────────────────────────
            _log_entry_shapes(key, entry, verbose)
            with env.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(entry))
            stats.record(L_query, N, weights)

        except Exception as exc:
            stats.skip(f"exception:{type(exc).__name__}")
            logger.warning("Skipped %s  [%s] %s", a3m_path.name, type(exc).__name__, exc)
            if verbose:
                logger.debug("Traceback:", exc_info=True)

        if (file_idx + 1) % log_every == 0:
            stats.log_progress(file_idx + 1)

    env.close()
    stats.log_summary()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    parser = argparse.ArgumentParser(description="Build MSAFlow LMDB dataset")
    parser.add_argument("--a3m_dir",              required=True)
    parser.add_argument("--output",               required=True)
    parser.add_argument("--protenix_checkpoint",  default=None)
    parser.add_argument("--max_msa_seqs",         type=int, default=512)
    parser.add_argument("--max_seq_len",          type=int, default=1024)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--map_size_gb",          type=int, default=500)
    parser.add_argument("--log_every",            type=int, default=100,
                        help="Log progress every N files")
    parser.add_argument("--verbose",              action="store_true",
                        help="Log per-entry shapes (sets log level to DEBUG)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    build_lmdb(
        a3m_dir=args.a3m_dir,
        output_path=args.output,
        protenix_checkpoint=args.protenix_checkpoint,
        max_msa_seqs=args.max_msa_seqs,
        max_seq_len=args.max_seq_len,
        device=args.device,
        map_size_gb=args.map_size_gb,
        log_every=args.log_every,
        verbose=args.verbose,
    )
