"""Run cG-SchNet's ``filter_generated.py`` under Open Babel 3.x.

cG-SchNet (Gebauer et al. 2022), which provides the molecular stability
metric used in the paper, was written against Open Babel 2.x and does
``import openbabel as ob`` / ``import pybel``. Open Babel 3.x moved both
under the ``openbabel`` package. This wrapper pre-binds the 3.x modules
under their old top-level names and then executes the upstream script, so
the cG-SchNet checkout on disk needs no changes.

Usage (identical to the original script, plus ``--split_file``):

    python3 scripts/run_filter_generated.py <mol_dict> --train_data_path <qm9.db> \
        --split_file <split.npz> --threads 0

``--split_file`` enables cG-SchNet's novelty check. Upstream only accepts
``--model_path``, a directory holding ``split.npz``; the wrapper creates such a
directory in a temporary location. Novelty has only been tested with
``--threads 0``.

Two bugs in the upstream source are fixed in memory before it is executed, and
the training fingerprints needed for novelty are computed once and cached (see
``_PATCHES``); the checkout on disk is left untouched. The cache lives in
CGSCHNET_FP_CACHE (default ~/.cache/cgschnet_train_fps) and is keyed on the
database and the split indices.

Set CGSCHNET_DIR to point at the cG-SchNet checkout (default:
molecules/cG-SchNet).
"""

import os
import sys
import tempfile

from openbabel import openbabel as _openbabel
from openbabel import pybel as _pybel

# Bind the 2.x names. Both submodules are already imported at this point,
# so replacing the package entry in sys.modules is safe.
sys.modules["openbabel"] = _openbabel
sys.modules["pybel"] = _pybel

# cG-SchNet also expects SchNetPack 1.x, which exposed a ``Properties``
# namespace of string constants. This code uses SchNetPack 2.x, where the same
# strings live in ``schnetpack.properties``. cG-SchNet only uses the constants
# below (as plain dict keys), so a small namespace is enough.
import tvsnr_mol  # noqa: F401
import schnetpack as _spk


class _Properties:
    R = "_positions"
    Z = "_atomic_numbers"
    atom_mask = "_atom_mask"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell = "_cell"
    cell_offset = "_cell_offset"


assert _spk.properties.R == _Properties.R and _spk.properties.Z == _Properties.Z, (
    "SchNetPack 2.x position/atomic-number keys no longer match the 1.x names "
    "that cG-SchNet expects; the .mol_dict keys would not line up."
)
_spk.Properties = _Properties

# Open Babel 3.x renamed several 2.x methods that cG-SchNet still calls.
# Re-add the old names as aliases rather than patching the upstream checkout.
_OB_ALIASES = {
    "OBBond": {"GetBO": "GetBondOrder", "SetBO": "SetBondOrder"},
    "OBAtom": {
        "GetValence": "GetExplicitDegree",
        "GetImplicitValence": "GetImplicitHCount",
        "ImplicitHydrogenCount": "GetImplicitHCount",
        "BOSum": "GetExplicitValence",
    },
}
for _cls_name, _aliases in _OB_ALIASES.items():
    _cls = getattr(_openbabel, _cls_name, None)
    if _cls is None:
        continue
    for _old, _new in _aliases.items():
        if not hasattr(_cls, _old) and hasattr(_cls, _new):
            # Wrap in a plain Python function: the SWIG-generated attribute is a
            # builtin that does not bind ``self`` when set as a class attribute.
            def _alias(self, *args, __target=getattr(_cls, _new), **kwargs):
                return __target(self, *args, **kwargs)

            setattr(_cls, _old, _alias)

CGSCHNET_DIR = os.environ.get(
    "CGSCHNET_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cG-SchNet"),
)
target = os.path.join(CGSCHNET_DIR, "filter_generated.py")

if not os.path.isfile(target):
    sys.exit(f"cG-SchNet not found at {target}. Set CGSCHNET_DIR to the checkout path.")

# Upstream bugs, fixed on the source text before execution. Each patch must match
# exactly once, so a changed checkout fails loudly instead of running unpatched.
_PATCHES = [
    # Molecule.type_infos includes S (16) and Cl (17), but the atomic-number
    # histogram was truncated at Z = 9. Only training molecules take this branch,
    # so every novelty run crashed with "IndexError: index 16 is out of bounds".
    (
        "np.bincount(mol, minlength=10)",
        "np.bincount(mol, minlength=max(Molecule.type_infos) + 1)",
    ),
    # train_idx is [train, val, test], so the first test molecule sits at index
    # len(train_idx) - n_test_mols and was counted as a validation match.
    (
        "if j > len(train_idx) - n_test_mols:",
        "if j >= len(train_idx) - n_test_mols:",
    ),
    # Speed, not correctness: the training fingerprints are the same for every
    # sample file and took ~50 s of a ~60 s run. They are cached as bit sets
    # (pybel Fingerprint objects cannot be pickled), and the comparison uses the
    # bit sets, which gives the same Tanimoto similarity.
    (
        "train_fps = _get_training_fingerprints(dbpath, train_idx, print_file,",
        "train_fps = _cached_training_fingerprints(dbpath, train_idx, print_file,",
    ),
    (
        "stats.T, stat_heads, print_file)",
        "stats.T, stat_heads, print_file, use_bits=True)",
    ),
]

with open(target) as f:
    source = f.read()
for _old, _new in _PATCHES:
    _n = source.count(_old)
    if _n != 1:
        sys.exit(
            f"Cannot patch {target}: expected 1 occurrence of {_old!r}, found {_n}."
        )
    source = source.replace(_old, _new)

# --split_file <split.npz>  ->  --model_path <tmpdir containing split.npz>
argv = sys.argv[1:]
_split_dir = None
if "--split_file" in argv:
    if "--model_path" in argv:
        sys.exit("Pass either --split_file or --model_path, not both.")
    i = argv.index("--split_file")
    split_file = os.path.abspath(argv[i + 1])
    del argv[i : i + 2]
    if not os.path.isfile(split_file):
        sys.exit(f"Split file not found: {split_file}")
    _split_dir = tempfile.TemporaryDirectory(prefix="cgschnet_split_")
    os.symlink(split_file, os.path.join(_split_dir.name, "split.npz"))
    argv += ["--model_path", _split_dir.name]

FP_CACHE_DIR = os.environ.get(
    "CGSCHNET_FP_CACHE", os.path.expanduser("~/.cache/cgschnet_train_fps")
)
main_globals = {"__name__": "__main__", "__file__": target}


def _cached_training_fingerprints(dbpath, train_idx, print_file=True, **kwargs):
    """Training fingerprints as bit sets, cached per (database, split indices)."""
    import hashlib
    import pickle

    import numpy as np

    key = hashlib.md5()
    key.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    key.update(os.path.abspath(dbpath).encode())
    key.update(str(os.path.getmtime(dbpath)).encode())
    path = os.path.join(FP_CACHE_DIR, f"train_fps_{key.hexdigest()[:16]}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    fps = main_globals["_get_training_fingerprints"](
        dbpath, train_idx, print_file, use_bits=True, use_con_mat=False
    )
    os.makedirs(FP_CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(fps, f)
    os.replace(tmp, path)  # atomic: parallel runs never read a partial file
    return fps


main_globals["_cached_training_fingerprints"] = _cached_training_fingerprints

# filter_generated.py imports its own sibling modules (utility_classes, ...)
sys.path.insert(0, CGSCHNET_DIR)
sys.argv = [target] + argv
exec(compile(source, target, "exec"), main_globals)
