"""Per-file molecular metrics for generated QM9 structures.

Two sources are combined into one ``<name>.vun.json`` per ``<name>.mol_dict``.

cG-SchNet (Gebauer et al. 2022), read from the ``<name>.res`` that
``scripts/compute_metrics_dir.sh`` writes. Bonds are perceived with Open Babel;
a molecule is valid if every atom has its exact valence and the bond graph is
connected.

* stability  -- (unique valid + duplicates) / generated. The quantity plotted in
  the paper.
* uniqueness -- unique valid / (unique valid + duplicates). Two valid molecules
  are duplicates if their FP2 fingerprints have Tanimoto similarity 1 and their
  canonical SMILES agree, where a mirror image counts as the same molecule.
* novelty    -- new / unique valid, where "new" is cG-SchNet's definition: the
  molecule matches no training and no validation structure (a match with the
  test split counts as new). The number of matches with each split is stored
  as well, so other definitions can be derived.

RDKit, with the validity rule of GPFF (``analyze_with_rdkit`` in
``gpff/analysis/structure.py``, github.com/stefaanhessmann/gpff). It is
reimplemented here to avoid GPFF's PoseBusters and torch imports; on 12,580
molecules (generated and QM9) it agreed with the original on every molecule.

* validity   -- ``rdDetermineBonds.DetermineBonds`` assigns connectivity and bond
  orders from the coordinates for a neutral molecule without formal charges. The
  molecule is valid if this succeeds within 5 s, no atom carries a radical
  electron, and the canonical SMILES is a single fragment. Unlike the EDM
  validity, disconnected structures are invalid. RDKit's bond perception is more
  permissive than Open Babel's, so this rejects about 5.5 % of real QM9 test
  molecules.

If a sample set has no valid molecule, cG-SchNet crashes; the shell script
detects that and writes a ``.res`` with zero counts, so stability, uniqueness
and novelty are 0. In general, a rate with an empty denominator is reported as 0.
Molecules with non-finite coordinates are invalid and unstable.

Wilson 95% confidence intervals are reported for every rate.

Usage
-----
    python3 scripts/compute_vun.py <file-or-directory> [...] \
        [--out-dir DIR] [--overwrite]
"""

import argparse
import json
import math
import os
import pickle
import signal
import sys
from io import StringIO

import numpy as np

RES_FIELDS = {
    "generated": "Number of generated molecules:",
    "duplicates": "Number of duplicate molecules:",
    "unique_valid": "Number of unique and valid molecules:",
    "new": "Number of new molecules:",
    "match_train": "Number of molecules matching training data:",
    "match_val": "Number of molecules matching validation data:",
    "match_test": "Number of molecules matching test data:",
}


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (95% by default)."""
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k, n):
    """Rate with a Wilson interval; an empty denominator gives 0."""
    if n == 0:
        return {
            "count": int(k),
            "total": 0,
            "rate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    lo, hi = wilson_interval(k, n)
    return {
        "count": int(k),
        "total": int(n),
        "rate": k / n,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def read_res(path):
    """Counts from a .res written by compute_metrics_dir.sh."""
    counts = {}
    with open(path) as f:
        for line in f:
            for key, label in RES_FIELDS.items():
                if key not in counts and line.startswith(label):
                    counts[key] = int(line[len(label) :].split()[0])
    missing = [label for key, label in RES_FIELDS.items() if key not in counts]
    if missing:
        raise ValueError(
            f"{path} lacks {missing}; it predates the novelty check. "
            "Delete it and re-run scripts/compute_metrics_dir.sh."
        )
    return counts


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def gpff_validity(positions, numbers, timeout=5):
    """RDKit validity as defined by GPFF's analyze_with_rdkit."""
    from ase import Atoms
    from ase.io import write
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.rdBase import BlockLogs

    if not np.isfinite(positions).all():
        return False

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout)
    try:
        with BlockLogs():
            xyz = StringIO()
            write(xyz, Atoms(numbers=numbers, positions=positions), format="xyz")
            mol = Chem.MolFromXYZBlock(xyz.getvalue())
            rdDetermineBonds.DetermineBonds(
                mol, charge=0, allowChargedFragments=False, embedChiral=True
            )
            if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
                return False
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
            return smiles != "" and "." not in smiles
    except Exception:  # includes the timeout: GPFF counts both as invalid
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def load_molecules(path):
    """All molecules of a .mol_dict as (positions, atomic numbers)."""
    with open(path, "rb") as f:
        mol_dict = pickle.load(f)
    molecules = []
    for n_atoms in sorted(mol_dict):
        entry = mol_dict[n_atoms]
        molecules += list(zip(entry["_positions"], entry["_atomic_numbers"]))
    return molecules


def evaluate(molecules, counts):
    n_total = len(molecules)
    n_valid = 0
    for positions, numbers in molecules:
        positions = np.asarray(positions, dtype=float)
        numbers = np.asarray(numbers, dtype=int)
        n_valid += gpff_validity(positions, numbers)

    unique_valid = counts["unique_valid"]
    valid_with_duplicates = unique_valid + counts["duplicates"]
    return {
        "n_generated": n_total,
        "stability": rate(valid_with_duplicates, counts["generated"]),
        "uniqueness": rate(unique_valid, valid_with_duplicates),
        "novelty": rate(counts["new"], unique_valid),
        "validity": rate(n_valid, n_total),
        "cgschnet_counts": counts,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "inputs", nargs="+", help=".mol_dict files or directories containing them"
    )
    ap.add_argument("--out-dir", default=None, help="default: next to each input file")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            files += [
                os.path.join(item, f)
                for f in sorted(os.listdir(item))
                if f.endswith(".mol_dict")
            ]
        else:
            files.append(item)
    if not files:
        sys.exit("No .mol_dict inputs found.")

    n_missing = 0
    for path in files:
        stem = os.path.basename(path)[: -len(".mol_dict")]
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        out_path = os.path.join(out_dir, stem + ".vun.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip (exists): {out_path}")
            continue
        res_path = os.path.join(os.path.dirname(path), stem + ".res")
        if not os.path.exists(res_path):
            print(f"skip (no .res yet): {stem}")
            n_missing += 1
            continue

        counts = read_res(res_path)
        result = evaluate(load_molecules(path), counts)
        if counts["generated"] != result["n_generated"]:
            print(
                f"WARNING {stem}: .res reports {counts['generated']} generated, "
                f".mol_dict holds {result['n_generated']}"
            )
        result["source"] = os.path.basename(path)

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"{stem}: stability={result['stability']['rate']:.4f} "
            f"validity={result['validity']['rate']:.4f} "
            f"uniqueness={result['uniqueness']['rate']:.4f} "
            f"novelty={result['novelty']['rate']:.4f}"
        )

    if n_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
