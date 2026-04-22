"""
Extract sequences from FoldBench ground truth CIF files and write a FASTA
for the monomer_protein benchmark targets.

Usage:
    python scripts/extract_foldbench_fasta.py \
        --targets   FoldBench/targets/monomer_protein.csv \
        --cif_dir   /gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520 \
        --output    data/foldbench_monomer.fasta
"""

import argparse
import gzip
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "MSE": "M", "UNK": "X",
}


def read_cif(path: Path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, "rt", errors="replace").read()


def extract_chain_sequence(cif_text: str, chain_id: str) -> str | None:
    """
    Extract one-letter sequence for the given auth_asym_id / label_asym_id chain.

    Strategy:
      1. _struct_asym  → map chain_id (id) to entity_id
      2. _entity_poly  → get pdbx_seq_one_letter_code_can for that entity
      Falls back to scanning _atom_site CA atoms if step 1/2 fail.
    """
    # ── Build entity_id → sequence map ───────────────────────────────────────
    entity_seqs: dict[str, str] = {}

    # Look for multi-line or inline _entity_poly loop
    ep_match = re.search(
        r"loop_\s*\n(?:_entity_poly\.\S+\s*\n)+", cif_text
    )
    if ep_match:
        block_start = ep_match.start()
        # Collect column names
        col_names = re.findall(r"_entity_poly\.(\S+)", ep_match.group(0))
        # Collect data lines after the loop header
        data_section = cif_text[ep_match.end():]
        # Stop at next loop_ or top-level # or _
        data_section = re.split(r"\n(?:loop_|#\s*$|_\w)", data_section)[0]
        data_lines = [
            l.strip() for l in data_section.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if "entity_id" in col_names:
            eid_idx = col_names.index("entity_id")
            for seq_col in ("pdbx_seq_one_letter_code_can", "pdbx_seq_one_letter_code"):
                if seq_col in col_names:
                    seq_idx = col_names.index(seq_col)
                    # Join continued lines (semicolon-delimited multi-line values)
                    joined = " ".join(data_lines)
                    # Split on whitespace respecting that sequences are long tokens
                    tokens = joined.split()
                    stride = len(col_names)
                    for i in range(0, len(tokens) - stride + 1, stride):
                        eid = tokens[eid_idx + i] if eid_idx + i < len(tokens) else None
                        seq = tokens[seq_idx + i] if seq_idx + i < len(tokens) else None
                        if eid and seq and seq != "?":
                            entity_seqs[eid] = re.sub(r"[^A-Za-z]", "", seq).upper()
                    break

    # ── Map chain → entity via _struct_asym ──────────────────────────────────
    sa_match = re.search(
        r"loop_\s*\n(?:_struct_asym\.\S+\s*\n)+", cif_text
    )
    if sa_match and entity_seqs:
        sa_col_names = re.findall(r"_struct_asym\.(\S+)", sa_match.group(0))
        sa_data = cif_text[sa_match.end():]
        sa_data = re.split(r"\n(?:loop_|#\s*$|_\w)", sa_data)[0]
        sa_lines = [
            l.strip() for l in sa_data.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if "id" in sa_col_names and "entity_id" in sa_col_names:
            id_idx  = sa_col_names.index("id")
            eid_idx = sa_col_names.index("entity_id")
            for line in sa_lines:
                parts = line.split()
                if len(parts) > max(id_idx, eid_idx):
                    cid = parts[id_idx]
                    eid = parts[eid_idx]
                    if cid == chain_id and eid in entity_seqs:
                        return entity_seqs[eid]

        # If the requested chain not found, try returning first protein entity
        for eid, seq in entity_seqs.items():
            if len(seq) > 10:  # skip short non-protein entities
                return seq

    # ── Fallback: scan _atom_site for CA atoms of requested chain ─────────────
    logger.debug("Falling back to atom_site scan for chain %s", chain_id)
    residues: list[tuple[int, str]] = []
    atom_header = re.search(
        r"loop_\s*\n(?:_atom_site\.\S+\s*\n)+", cif_text
    )
    if atom_header:
        col_names = re.findall(r"_atom_site\.(\S+)", atom_header.group(0))
        data = cif_text[atom_header.end():]
        data = re.split(r"\n(?:loop_|#\s*$)", data)[0]

        needed = ["label_atom_id", "label_comp_id", "label_seq_id"]
        chain_cols = ["label_asym_id", "auth_asym_id"]
        if not all(c in col_names for c in needed):
            return None

        atom_idx  = col_names.index("label_atom_id")
        comp_idx  = col_names.index("label_comp_id")
        seq_idx   = col_names.index("label_seq_id")
        chain_idx = next(
            (col_names.index(c) for c in chain_cols if c in col_names), -1
        )
        group_idx = col_names.index("group_PDB") if "group_PDB" in col_names else -1

        seen_seq_ids: set[int] = set()
        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= max(atom_idx, comp_idx, seq_idx):
                continue
            if group_idx >= 0 and parts[group_idx] != "ATOM":
                continue
            if parts[atom_idx] != "CA":
                continue
            if chain_idx >= 0 and parts[chain_idx] != chain_id:
                continue
            try:
                sid = int(parts[seq_idx])
            except ValueError:
                continue
            if sid not in seen_seq_ids:
                seen_seq_ids.add(sid)
                aa3 = parts[comp_idx]
                residues.append((sid, AA_3TO1.get(aa3, "X")))

    if residues:
        residues.sort()
        return "".join(r[1] for r in residues)

    return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="monomer_protein.csv")
    parser.add_argument("--cif_dir", required=True, help="Ground truth CIF directory")
    parser.add_argument("--output",  required=True, help="Output FASTA path")
    parser.add_argument("--min_len", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()

    df = pd.read_csv(args.targets)
    logger.info("Targets: %d", len(df))

    cif_dir = Path(args.cif_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    fasta_entries: list[str] = []

    for _, row in df.iterrows():
        pdb_id   = str(row["pdb_id"])
        chain_id = str(row["chain_id"])

        cif_path = cif_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            cif_path = cif_dir / f"{pdb_id}.cif.gz"
        if not cif_path.exists():
            logger.warning("CIF not found: %s", pdb_id)
            skipped += 1
            continue

        try:
            cif_text = read_cif(cif_path)
            seq = extract_chain_sequence(cif_text, chain_id)
        except Exception as exc:
            logger.warning("Parse error %s: %s", pdb_id, exc)
            skipped += 1
            continue

        if not seq:
            logger.warning("No sequence extracted: %s chain %s", pdb_id, chain_id)
            skipped += 1
            continue

        if not (args.min_len <= len(seq) <= args.max_len):
            logger.info("Skip %s: len=%d out of [%d, %d]", pdb_id, len(seq), args.min_len, args.max_len)
            skipped += 1
            continue

        fasta_entries.append(f">{pdb_id}\n{seq}")
        written += 1

    with open(out_path, "w") as fh:
        fh.write("\n".join(fasta_entries) + "\n")

    logger.info("Written: %d  Skipped: %d  →  %s", written, skipped, out_path)


if __name__ == "__main__":
    main()
