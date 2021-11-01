import torch
from schnetpack import AtomsLoader
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack.environment import AseEnvironmentProvider


def get_latent_space_representations(model, atoms, batch_size=1):

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
