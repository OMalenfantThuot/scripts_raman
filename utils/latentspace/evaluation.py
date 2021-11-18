import torch
from schnetpack import AtomsLoader
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack.environment import AseEnvironmentProvider


def get_latent_space_representations(model, atoms):
    r"""
    Function that returns an (N_atoms x N_neurons) Tensor containing the latent
    space representation of the configurations of in a dataset for a given 
    trained model.
    """

    cutoff = float(model.representation.interactions[0].cutoff_network.cutoff)
    n_neurons = model.representation.n_atom_basis

    data = SchnetPackData(
        data=atoms,
        environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        collect_triples=False,
    )
    data_loader = AtomsLoader(data, batch_size=1)

    for batch in data_loader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        with torch.no_grad():
            rep = model.representation(batch)
        batch["representation"] = rep
        return batch


def get_scaling_factors(inputs, train_representations, metric="euclidian", model=None):

    if metric == "euclidian":
        factors = torch.ones(train_representations.shape[1])

    elif metric == "scaled_max":
        factors = (
            torch.max(train_representations, dim=0).values
            - torch.min(train_representations, dim=0).values
        )
        factors /= torch.max(factors)

    elif metric == "scaled_std":
        factors = torch.std(train_representations, dim=0)
        factors /= torch.max(factors)

    elif metric == "gradient":
        inputs["representation"].requires_grad_()
        outputmod = model.output_modules[0]
        outputmod.derivative = None
        out = outputmod(inputs)

        (factors,) = torch.abs(
            torch.autograd.grad(
                out["energy"], inputs["representation"], torch.ones_like(out["energy"])
            )[0]
        )

    return factors


def get_latent_space_distances(
    representations, train_representations, factors, metric=None
):
    r"""
    Function that returns the distances between every atoms in a test set (representations)
    and every atoms in a training set (train_representations). The way those distances are 
    calculated is dependent on the chosen metric (Default is euclidian distance).

    Choices of the metric keyword values:

    euclidian: Uses euclidian distances where every dimension is weighted equally.

    scaled_max: Every dimension is scaled proportionaly to the maximum width in that same
        dimension between any atoms in the training set.

    scaled_std: Similar to scaled_max, but scaled proportionaly to the standard deviation
        instead of the maximum.
    """

    if metric == "gradient":
        factors = factors.reshape(factors.shape[0], 1, factors.shape[1])

    distances = (representations - train_representations) * factors
    distances = torch.linalg.norm(distances, dim=2).cpu().detach().numpy()
    return distances
