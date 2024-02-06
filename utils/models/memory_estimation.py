import torch
import numpy as np


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
            0.022
            + 3.4e-4 * n_neurons
            + 2.8e-5 * n_neurons * n_interactions * cutoff**2
        )
    elif cutoff == 7:
        return (
            0.022
            + 7.4e-4 * n_neurons
            + 2.8e-5 * n_neurons * n_interactions * cutoff**2
        )


def graphene_max_atoms_per_patch_hessian(
    n_interactions: int, cutoff: float, n_neurons: int
):
    device_name = torch.cuda.get_device_name()

    if device_name == "Tesla V100-SXM2-16GB":
        if n_interactions == 2:
            if cutoff == 6 and n_neurons == 128:
                return 12800
        elif n_interactions == 3:
            if cutoff == 6:
                if n_neurons == 128:
                    return 9800
                elif n_neurons == 64:
                    return 8712 # relance
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
    mem_per_atom, interactions, cutoff, supercell_size_x, supercell_size_y
):
    # total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    total_gpu_mem = 30000
    if supercell_size_x * supercell_size_y * 2 * mem_per_atom < total_gpu_mem:
        return np.array([1, 1, 1])
    else:
        target_num_atoms = int(0.95 * total_gpu_mem / mem_per_atom)
        print(target_num_atoms)
        prim_len = 2.46738

        buffer = (interactions * cutoff) / np.sin(np.radians(60)) / prim_len

        new_x, new_y = 1, 1
        full_num_atoms = target_num_atoms * 2

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
                (supercell_size_x / new_x + 2 * buffer)
                * (supercell_size_y / new_y + 2 * buffer)
                * np.sin(np.radians(60))
            )
            full_num_atoms = patches_num_atoms * full_area / patches_area
        return np.array([new_x, new_y, 1])
