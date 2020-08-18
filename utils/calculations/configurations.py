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
    cutoff = 6.0
    print('Cutoff:', cutoff)
    unique_reps, unique_config, reps, count_configs, unique_idx  = [], [], [], [], []
    schnet = SchNet(n_atom_basis=32, n_filters=32, cutoff=cutoff, cutoff_network=CosineCutoff)
    print('SchNet ok')
    env = AseEnvironmentProvider(cutoff=cutoff)
    print('env ok')

    data = [posinp_to_ase_atoms(pos) for pos in configurations]
    print('data ok')
    
    data = SchnetPackData(data=data, environment_provider=env, collect_triples=False)
    print('SchnetPackData ok')
    data_loader = AtomsLoader(data, batch_size=1)
    print('data_loader ok')

    aa=0

    for batch in data_loader:
        print('batch', aa)
        reps.append(torch.squeeze(schnet(batch)))
        aa+=1

    for i, rep in enumerate(reps):
        for j, uni in enumerate(unique_reps):
            #print('comparing {} and {}'.format(i, j))
            if compare_reps(rep, uni):
                count_configs[j]+=1
                break
        else:
            unique_reps.append(rep)
            unique_config.append(configurations[i])
            count_configs.append(1)
            unique_idx.append(i)
            print('Config {} is unique'.format(i))
    print('Unique indices:', unique_idx)        
    return unique_config, count_configs
