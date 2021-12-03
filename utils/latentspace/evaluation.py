import torch
from schnetpack import AtomsLoader
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack.environment import AseEnvironmentProvider
from utils.models import LatentAtomwise

# from schnetpack.nn import shifted_softplus


def get_latent_space_representations(model, atoms, output_rep=False, latent_out=None):
    r"""
    Function that returns an (N_atoms x N_neurons) Tensor containing the latent
    space representation of the configurations in a dataset for a given 
    trained model.
    """

    if not isinstance(atoms, list):
        atoms = [atoms]
    run_mode = "dataset" if len(atoms) > 1 else "single"

    cutoff = float(model.representation.interactions[0].cutoff_network.cutoff)
    n_neurons = model.representation.n_atom_basis

    data = SchnetPackData(
        data=atoms,
        environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        collect_triples=False,
    )
    data_loader = AtomsLoader(data, batch_size=1)

    if output_rep and latent_out is None:
        latent_out = LatentAtomwise.from_Atomwise(model.output_modules[0])
        latent_out.to(next(model.parameters()).device)

    if run_mode == "dataset":
        individual_reps, individual_outs = [], []
        for batch in data_loader:
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            with torch.no_grad():
                rep = model.representation(batch)
            individual_reps.append(rep)

            if output_rep:
                batch["representation"] = rep
                output = latent_out(batch)[-1]
                individual_outs.append(output)

        representations = torch.cat(individual_reps).reshape(-1, n_neurons)
        if output_rep:
            output_reps = torch.cat(individual_outs).reshape(-1, int(n_neurons / 2))
            return {"representation": representations, "output": output_reps}
        else:
            return {"representation": representations}

    elif run_mode == "single":
        for batch in data_loader:
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            with torch.no_grad():
                rep = model.representation(batch)
            batch["representation"] = rep

            if output_rep:
                output = latent_out(batch)[-1]
                batch["output"] = output[0]
            return batch


def get_scaling_factors(inputs, train_representations, scaling="isotropic", model=None):

    if scaling in ["isotropic"]:
        factors = torch.ones(train_representations.shape[1])

    elif scaling == "scaled_max":
        factors = (
            torch.max(train_representations, dim=0).values
            - torch.min(train_representations, dim=0).values
        )
        factors /= torch.max(factors)

    elif scaling == "scaled_std":
        factors = torch.std(train_representations, dim=0)
        factors /= torch.max(factors)

    elif scaling == "gradient":
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
    representations, train_representations, factors, metric=None, grad=False
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

    if grad:
        factors = factors.reshape(factors.shape[0], 1, factors.shape[1])

    distances = (representations - train_representations) * factors
    if metric == "euclidian":
        distances = torch.linalg.norm(distances, dim=2).cpu().detach().numpy()
    elif metric == "linear":
        distances = torch.sum(torch.abs(distances), dim=2).cpu().detach().numpy()
    else:
        raise NotImplementedError("Metric {} not implemented.".format(metric))
    return distances
