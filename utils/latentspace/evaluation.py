import torch
from schnetpack import AtomsLoader
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack.environment import AseEnvironmentProvider


def get_latent_space_representations(model, atoms, batch_size=1):
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
    data_loader = AtomsLoader(data, batch_size=batch_size)

    individual_reps = []
    for batch in data_loader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        with torch.no_grad():
            rep = model.representation(batch)
        individual_reps.append(rep.detach())

    representations = torch.cat(individual_reps).reshape(-1, n_neurons)
    return representations


def get_latent_space_distances(
    representations, train_representations, metric="euclidian"
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

    if metric == "euclidian":
        distances = (
            torch.linalg.norm(representations - train_representations, dim=2)
            .cpu()
            .detach()
            .numpy()
        )

    elif metric == "scaled_max":
        factors = (
            torch.max(train_representations, dim=0).values
            - torch.min(train_representations, dim=0).values
        )
        factors /= torch.max(factors)

        distances = (representations - train_representations) * factors
        distances = torch.linalg.norm(distances, dim=2).cpu().detach().numpy()

    elif metric == "scaled_std":
        factors = torch.std(train_representations, dim=0)
        factors /= torch.max(factors)

        distances = (representations - train_representations) * factors
        distances = torch.linalg.norm(distances, dim=2).cpu().detach().numpy()

    return distances
