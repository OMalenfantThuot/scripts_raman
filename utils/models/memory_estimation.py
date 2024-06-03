import torch
import numpy as np
from schnetpack.utils import load_model


def get_schnet_hyperparams(model):
    temp_model = load_model(model, map_location="cpu")
    n_neurons = temp_model.representation.embedding.weight.shape[1]
    n_interactions = len(temp_model.representation.interactions)
    cutoff = float(temp_model.representation.interactions[0].cutoff_network.cutoff)
    del temp_model
    return n_neurons, n_interactions, cutoff


def schnet_memory_estimation_for_graphene(
    interactions, cutoff, n_neurons, prefactor=0.000031, bias=0.03
):
    return prefactor * interactions * cutoff * n_neurons + bias


def graphene_schnet_memory_estimation_per_atom(
    n_interactions: int, cutoff: float, n_neurons: int
):
    if not (cutoff == np.array([5, 6, 7])).any():
        raise NotImplementedError(f"Not implemented for cutoff = {cutoff} A")

    if (cutoff == np.array([5, 6])).any():
        return (
            0.022 + 3.4e-4 * n_neurons + 2.8e-5 * n_neurons * n_interactions * cutoff**2
        )
    elif cutoff == 7:
        return (
            0.022 + 7.4e-4 * n_neurons + 2.8e-5 * n_neurons * n_interactions * cutoff**2
        )


def graphene_max_atoms_per_patch_forces(
    n_interactions: int, cutoff: float, n_neurons: int
):
    device_name = torch.cuda.get_device_name()

    if device_name == "Tesla V100-SXM2-16GB":
        if n_interactions == 2:
            if cutoff == 7 and n_neurons == 256:
                return 11250
    raise NotImplementedError(
        f"The maximum number of atoms is not know for the device {device_name}"
        f"with {n_interactions} interaction blocks, {cutoff} cutoff and {n_neurons} neurons."
    )


def graphene_max_atoms_per_patch_hessian(
    n_interactions: int, cutoff: float, n_neurons: int
):
    device_name = torch.cuda.get_device_name()

    if device_name == "Tesla V100-SXM2-16GB":
        if n_interactions == 2:
            if cutoff == 6 and n_neurons == 128:
                return 12800
            elif cutoff == 7 and n_neurons == 256:
                return 6728
        elif n_interactions == 3:
            if cutoff == 6:
                if n_neurons == 128:
                    return 9800
                elif n_neurons == 64:
                    return 8712  # relance
                elif n_neurons == 256:
                    return 5202
            if cutoff == 5 and n_neurons == 128:
                return 13122
            if cutoff == 7 and n_neurons == 128:
                return 7442
        elif n_interactions == 4:
            if cutoff == 6 and n_neurons == 128:
                return 8192
        elif n_interactions == 6:
            if cutoff == 6 and n_neurons == 128:
                return 5408

    raise NotImplementedError(
        f"The maximum number of atoms is not know for the device {device_name}"
        f"with {n_interactions} interaction blocks, {cutoff} cutoff and {n_neurons} neurons."
    )


def get_graphene_patches_grid(
    pred: str,
    n_interactions: int,
    cutoff: float,
    n_neurons: int,
    supercell_size_x: int,
    supercell_size_y: int,
):
    n_atoms_supercell = supercell_size_x * supercell_size_y * 2
    if pred == "hessian":
        max_atoms_per_patch = graphene_max_atoms_per_patch_hessian(
            n_interactions, cutoff, n_neurons
        )
    elif pred == "forces":
        max_atoms_per_patch = graphene_max_atoms_per_patch_forces(
            n_interactions, cutoff, n_neurons
        )
    else:
        raise RuntimeError("Prediction not recognized.")

    if n_atoms_supercell < max_atoms_per_patch:
        return np.array([1, 1, 1])
    else:
        target_num_atoms = int(0.95 * max_atoms_per_patch)
        prim_len = 2.46738
        buffer_width = (n_interactions * cutoff) / (prim_len * np.sin(np.radians(60)))

        new_x, new_y = 1, 1
        full_num_atoms = target_num_atoms * 2  # Random initialisation

        while full_num_atoms > target_num_atoms:
            if supercell_size_x / new_x >= supercell_size_y / new_y:
                new_x += 1
            else:
                new_y += 1

            patches_area = (
                (supercell_size_x / new_x)
                * (supercell_size_y / new_y)
                * np.sin(np.radians(60))
            )
            patches_num_atoms = (
                (supercell_size_x / new_x) * (supercell_size_y / new_y) * 2
            )
            full_area = (
                (supercell_size_x / new_x + 2 * buffer_width)
                * (supercell_size_y / new_y + 2 * buffer_width)
                * np.sin(np.radians(60))
            )
            full_num_atoms = patches_num_atoms * full_area / patches_area

        grid = np.array([new_x, new_y, 1])
        if pred == "hessian":
            grid[grid == 2] += 1
        return grid
