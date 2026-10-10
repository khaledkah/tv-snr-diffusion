import os
import pickle
import sys

import numpy as np
import ase.db
from tqdm import tqdm

import tvsnr_mol  # noqa: F401
from schnetpack import properties


def convert_to_mol_dict(path):

    # only files ending with .db
    files = [file for file in os.listdir(path) if file.endswith(".db")]

    # only non-existing files
    files = [
        file
        for file in files
        if not os.path.exists(os.path.join(path, file.replace(".db", ".mol_dict")))
    ]

    for file in tqdm(files):
        source = ase.db.connect(os.path.join(path, file))
        mol_dict = {}

        for row in source.select():
            mol = row.toatoms()
            n_atoms = len(mol)
            if n_atoms in mol_dict:
                mol_dict[n_atoms] = {
                    properties.R: np.concatenate(
                        [mol_dict[n_atoms][properties.R], mol.positions[None]], axis=0
                    ),
                    properties.Z: np.concatenate(
                        [mol_dict[n_atoms][properties.Z], mol.numbers[None]], axis=0
                    ),
                }
            else:
                mol_dict[n_atoms] = {
                    properties.R: mol.positions[None],
                    properties.Z: mol.numbers[None],
                }

        # save dictionary
        save_path = os.path.join(path, file.replace(".db", ".mol_dict"))
        with open(save_path, "wb") as f:
            pickle.dump(mol_dict, f)


if __name__ == "__main__":
    path = sys.argv[1]
    convert_to_mol_dict(path)
