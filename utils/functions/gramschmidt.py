import numpy as np

def gramschmidt(basis, normalize=True):
    
    basis = [np.array(vec, dtype=np.float64) for vec in basis]
    nvectors = len(basis)
    ndim = len(basis[0])
    for vec in basis:
        assert len(vec) == ndim

    new_basis = []
    for i in range(nvectors):
        new_vec = basis[i].copy()
        for vec in new_basis:
            new_vec -= proj(basis[i], vec)
        if normalize:
            new_vec /= np.linalg.norm(new_vec)
        new_basis.append(new_vec)
    return np.array(new_basis)

def proj(v, u):
    return np.dot(v, u) / np.dot(u,u) * u
