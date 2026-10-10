import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from tqdm import tqdm

import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from tvsnr_mol.bonds import allowed_bonds_dict, bonds1, bonds2, bonds3


def compute_neighbors(
    old_batch,
    neighbor_list_trn: Optional[trn.Transform] = None,
    cutoff=5.0,
    fully_connected=False,
    additional_keys=[],
    device=None,
):
    """
    function to compute the neighbors for a batch of systems

    Args:
        old_batch: batch of systems to compute the neighbors for
        neighbor_list_trn: transform to compute the neighbors
        cutoff: cutoff radius for the neighbor list
        fully_connected: if True, all atoms are connected to each other.
                            Ignores the cutoff.
        additional_keys: additional keys to be included in the new batch
        device: Pytorch device
    """
    if device is None:
        device = old_batch[properties.R].device

    # get the float precision
    f_dtype = old_batch[properties.R].dtype

    # initialize the neighbor list transform
    if fully_connected:
        from tvsnr_mol.transforms import AllToAllNeighborList

        neighbors_calculator = AllToAllNeighborList()
    else:
        neighbors_calculator = neighbor_list_trn or trn.MatScipyNeighborList(
            cutoff=cutoff
        )

    batch = []

    # compute the neighbors for each molecule in the batch
    for j, i in enumerate(torch.unique(old_batch[properties.idx_m])):
        mask = old_batch[properties.idx_m] == i
        inp = {
            properties.idx: old_batch[properties.idx][[j]].detach().cpu(),
            properties.n_atoms: old_batch[properties.n_atoms][[j]].detach().cpu(),
            properties.Z: old_batch[properties.Z][mask].detach().cpu(),
            properties.R: old_batch[properties.R][mask].detach().cpu(),
            properties.cell: old_batch[properties.cell][[j]].detach().to(f_dtype).cpu(),
            properties.pbc: old_batch[properties.pbc].view(-1, 3)[j].detach().cpu(),
        }

        inp = neighbors_calculator(inp)

        batch.append(inp)

    # create the new batch
    batch = _atoms_collate_fn(batch)

    # add additional keys
    batch.update({k: old_batch[k] for k in additional_keys})

    batch = {p: batch[p].to(device) for p in batch}

    return batch


def squared_euclidean_distance(a, b):
    """
    Efficiently compute the squared Euclidean distance between two sets of points.

    Args:
        a: first set of points
        b: second set of points
    """
    distance = (
        (a**2).sum(axis=1)[:, None] - 2 * np.dot(a, b.T) + (b**2).sum(axis=1)[None]
    )

    return np.where(distance < 0, np.zeros(distance.shape), distance)


def check_validity(
    inputs,
    m_bonds_1,
    m_bonds_2,
    m_bonds_3,
    allowed_bonds,
    bonds_relaxation=None,
    progress_bar=True,
):
    """
    Fast check for the validity of molecules in a batch, including mol connectivity.

    Args:
        inputs: batch of molecules
        m_bonds_1: matrix of covalent radii for single bonds
        m_bonds_2: matrix of covalent radii for double bonds
        m_bonds_3: matrix of covalent radii for triple bonds
        allowed_bonds: number of allowed bonds per atom
        bonds_relaxation: relaxation coefficients for the covalent radii
        progress_bar: show tqsm progress bar
    """
    # set default covalent radii relaxation coefficients
    bonds_relaxation = bonds_relaxation or [0.1, 0.05, 0.03]

    bonds = []
    stable_atoms = []
    stable_molecules = []
    stable_atoms_wo_h = []
    stable_molecules_wo_h = []
    connected = []
    connected_wo_h = []

    # create idx_m if one system is given
    if properties.idx_m not in inputs:
        inputs[properties.idx_m] = torch.zeros(
            len(inputs[properties.Z]), dtype=torch.int32
        )
        progress_bar = False

    # loop over molecules in the batch
    for m in tqdm(torch.unique(inputs[properties.idx_m]), disable=not progress_bar):
        # get the atomic numbers and positions for the current molecule
        mask = inputs[properties.idx_m] == m
        R = inputs[properties.R][mask]
        Z = inputs[properties.Z][mask]
        if torch.is_tensor(R):
            R = R.detach().cpu().numpy()
        if torch.is_tensor(Z):
            Z = Z.detach().cpu().numpy()

        # get covalent radii for the atoms in the current molecule
        ex_bonds_1 = m_bonds_1[Z[None], Z[:, None]]
        ex_bonds_2 = m_bonds_2[Z[None], Z[:, None]]
        ex_bonds_3 = m_bonds_3[Z[None], Z[:, None]]

        # compute distance matrix
        dist = squared_euclidean_distance(R, R) ** 0.5
        np.fill_diagonal(dist, np.inf)

        # get bond types per atom
        bonds_ = np.where(dist < ex_bonds_1 + bonds_relaxation[0], 1, 0)
        bonds_ = np.where(dist < ex_bonds_2 + bonds_relaxation[1], 2, bonds_)
        bonds_ = np.where(dist < ex_bonds_3 + bonds_relaxation[2], 3, bonds_)

        bonds.append(bonds_)

        # check if molecule is stable
        total_bonds = bonds_.sum(1)
        stable_at = allowed_bonds[Z] == total_bonds
        stable_atoms.append(stable_at)
        stable_molecules.append(stable_at.all())

        # check if molecule is stable without hydrogen
        stable_at_wo_h = stable_at.copy()
        stable_at_wo_h[Z == 1] = True
        stable_atoms_wo_h.append(stable_at_wo_h)
        stable_molecules_wo_h.append(stable_at_wo_h.all())

        # check if ALL the molecule is connected
        # using the exponent of the adjacency matrix trick
        bonds_t = (bonds[-1]) + np.eye(bonds[-1].shape[0])
        bonds_t = bonds_t > 0
        for i in range(bonds_t.shape[0]):
            bonds_t = bonds_t.dot(bonds_t)
        connected.append(bonds_t.all(1).any())

        # check if molecule is connected without hydrogen
        bonds_t[:, Z == 1] = True
        connected_wo_h.append(bonds_t.all(1).any())

    results = {
        "bonds": bonds,
        "stable_atoms": stable_atoms,
        "stable_molecules": stable_molecules,
        "connected": connected,
        "stable_atoms_wo_h": stable_atoms_wo_h,
        "stable_molecules_wo_h": stable_molecules_wo_h,
        "connected_wo_h": connected_wo_h,
    }

    return results


def generate_bonds_data(save_path: Optional[str] = None, overwrite: bool = False):
    """
    generate the bonds data as connectivity matrix between possible atoms.

    Args:
        save_path: path to save the data
        overwrite: overwrite existing data
    """
    save_path = save_path or "./bonds.pkl"

    if os.path.exists(save_path) and not overwrite:
        logging.info("Bonds data already exists, skipping generation and reloading...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    atoms = np.array(chemical_symbols)
    indices = np.arange(len(atoms))
    m_bonds_1 = np.ones((len(atoms), len(atoms))) * -np.inf
    m_bonds_2 = m_bonds_1.copy()
    m_bonds_3 = m_bonds_1.copy()
    allowed_bonds = np.zeros((len(atoms)), dtype=np.int32)

    # define the bonds types and allowed bonds per atom
    for at in atoms:
        for at2 in atoms:
            if at in bonds1 and at2 in bonds1[at]:
                m_bonds_1[indices[atoms == at], indices[atoms == at2]] = (
                    bonds1[at][at2] / 100.0
                )
            if at in bonds2 and at2 in bonds2[at]:
                m_bonds_2[indices[atoms == at], indices[atoms == at2]] = (
                    bonds2[at][at2] / 100.0
                )
            if at in bonds3 and at2 in bonds3[at]:
                m_bonds_3[indices[atoms == at], indices[atoms == at2]] = (
                    bonds3[at][at2] / 100.0
                )
        if at in allowed_bonds_dict:
            allowed_bonds[indices[atoms == at]] = allowed_bonds_dict[at]

    data = {
        "bonds_1": m_bonds_1,
        "bonds_2": m_bonds_2,
        "bonds_3": m_bonds_3,
        "allowed_bonds": allowed_bonds,
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return data


def _infer_inputs(system: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Checks the input data and creates the first inputs to the sampler.
    Fills missing data with random values, for instance, if no positions are given
    to sample from p(R|Z).

    Args:
        system: one input system.
    """
    if not isinstance(system, dict):
        raise ValueError("Inputs must be dicts.")

    if properties.Z not in system:
        raise NotImplementedError(
            "Atomic numbers must be provided. "
            "Generation of Z is not implemented yet."
        )
    else:
        numbers = system[properties.Z]

    if properties.cell in system or properties.pbc in system:
        raise NotImplementedError("Cell and PBC generation are not supported yet.")

    if (
        properties.R not in system
        and properties.Z not in system
        and properties.n_atoms not in system
    ):
        raise ValueError("at least one of R, Z or n_atoms must be provided")

    # get number of atoms
    if properties.n_atoms not in system:
        n_atoms = (
            len(system[properties.R])
            if properties.R in system
            else len(system[properties.Z])
        )
    else:
        n_atoms = system[properties.n_atoms].item()

    # get or initialize positions
    if properties.R not in system:
        positions = torch.randn(n_atoms, 3)  # type: ignore
    else:
        positions = system[properties.R]

    if not (len(numbers) == len(positions) == n_atoms):
        raise ValueError("len of R and Z must be equal to n_atoms.")

    return numbers, positions


def create_inputs(
    inputs: List[Union[torch.Tensor, Dict[str, torch.Tensor], Atoms]],
    additional_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    transforms: Optional[List[trn.Transform]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Prepares and converts the inputs for the sampler.

    Args:
        inputs: the inputs to be converted to the sampler.
        additional_inputs: Optional additional inputs to append to each molecule.
    """
    if device is None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cpu")
        )

    # set default transforms
    if transforms is None:
        transforms = [
            trn.CastTo64(),
            trn.SubtractCenterOfGeometry(),
        ]

    # check inputs format
    if (
        isinstance(inputs, torch.Tensor)
        or isinstance(inputs, dict)
        or isinstance(inputs, Atoms)
    ):
        inputs = [inputs]
    elif not isinstance(inputs, list):
        raise ValueError(
            "Inputs must be:"
            "one element or list of tensors with the atomic numbers Z "
            "one element or list of Dict of tensors including R and Z "
            "one element or list of ase.Atoms."
        )

    if isinstance(inputs[0], torch.Tensor):
        inputs = [{properties.Z: inp} for inp in inputs]  # type: ignore

    if isinstance(additional_inputs, dict):
        additional_inputs = [additional_inputs]

    if additional_inputs is not None and len(inputs) != len(additional_inputs):
        raise ValueError(
            "len of inputs and additional_inputs must be equal."
            f"Got {len(inputs)} and {len(additional_inputs)}."
        )

    batch = []
    for idx, system in enumerate(inputs):
        if isinstance(system, dict):
            # sanity checks
            numbers, positions = _infer_inputs(system)

            # convert to ase.Atoms
            mol = Atoms(numbers=numbers, positions=positions)
        elif isinstance(system, Atoms):
            mol = system
            system = {}
        else:
            raise ValueError("system must be a dict or ase.Atoms object.")

        # convert to dict of tensors
        system.update(
            {
                properties.n_atoms: torch.tensor([mol.get_global_number_of_atoms()]),
                properties.Z: torch.from_numpy(mol.get_atomic_numbers()),
                properties.R: torch.from_numpy(mol.get_positions()),
                properties.cell: torch.from_numpy(mol.get_cell().array).view(-1, 3, 3),
                properties.pbc: torch.from_numpy(mol.get_pbc()).view(-1, 3),
                properties.idx: torch.tensor([idx]),
            }
        )

        # apped additional inputs
        if additional_inputs is not None:
            system.update(additional_inputs[idx])

        # apply transforms
        for transform in transforms:
            system = transform(system)

        batch.append(system)

    # collate batch in a dict of tensors
    batch = _atoms_collate_fn(batch)

    # Move input batch to device
    batch = {p: batch[p].to(device) for p in batch}

    return batch
