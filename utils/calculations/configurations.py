import torch
from mlcalcdriver.interfaces import posinp_to_ase_atoms, SchnetPackData
from schnetpack import AtomsLoader
from schnetpack.representation import SchNet
from schnetpack.nn import CosineCutoff
from schnetpack.environment import AseEnvironmentProvider

def compare_reps(rep1, rep2):
    at_list1 = [at for at in rep1]
    at_list2 = [at for at in rep2]
    for at1 in at_list1:
        for i, at2 in enumerate(at_list2):
            if torch.allclose(at1, at2, atol=1e-04):
                del at_list2[i]
                break
        else:
            return(False)
    else:
        return(True)

def determine_unique_configurations(configurations):
    cutoff = 5.0
    unique_reps, unique_config, reps = [], [], []
    schnet = SchNet(n_atom_basis=32, n_filters=32, cutoff=cutoff, cutoff_network=CosineCutoff)
    env = AseEnvironmentProvider(cutoff=cutoff)

    data = [posinp_to_ase_atoms(pos) for pos in configurations]
    
    data = SchnetPackData(data=data, environment_provider=env, collect_triples=False)
    data_loader = AtomsLoader(data, batch_size=1)

    for batch in data_loader:
        reps.append(torch.squeeze(schnet(batch)))

    for i, rep in enumerate(reps):
        for uni in unique_reps:
            if compare_reps(rep, uni):
                break
        else:
            unique_reps.append(rep)
            unique_config.append(configurations[i])
    return unique_config
