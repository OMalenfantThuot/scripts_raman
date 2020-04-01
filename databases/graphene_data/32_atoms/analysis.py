from mlcalcdriver import Posinp
import numpy as np
import os
from copy import copy
import matplotlib.pyplot as plt

pos_folder = "positions/"

positions = [
    Posinp.from_file(pos_folder + file)
    for file in os.listdir(pos_folder)
    if file.endswith(".xyz")
]

distances, angles = [], []
for pos in positions:
    az_idx = [i for i, at in enumerate(pos) if at.type == "N"]
    if len(az_idx) == 2:
        distances.append(pos.distance(az_idx[0], az_idx[1]))
        angles.append(np.pi)
    elif len(az_idx) == 3:
        mid = np.array(pos.cell) / 2
        d = np.array([np.linalg.norm(pos[idx].position - mid) for idx in az_idx])
        mid_idx = az_idx[np.argmin(d)]
        other_idx = copy(az_idx)
        del other_idx[np.argmin(d)]

        distances.append(pos.distance(mid_idx, other_idx[0]))
        distances.append(pos.distance(mid_idx, other_idx[1]))
        angles.append(pos.angle(other_idx[0], mid_idx, other_idx[1]))
    else:
        raise NotImplementedError

# Uncomment for visual

# fig, ax = plt.subplots(ncols = 2, figsize=[15, 11])
# ax1, ax2 = ax.flatten()
#
# ax1.set_xlim(0, 7)
# ax1.hist(distances, bins=20)
# ax1.set_title("Distance distribution")
#
# ax2.set_xlim(0, np.pi)
# ax2.hist(angles, bins=20)
# ax2.set_title("Angle distribution")
# plt.show()

hist_d, edges_d = np.histogram(distances, bins=20)
hist_an, edges_an = np.histogram(angles, bins=20)
nzd, nza = np.where(hist_d > 0)[0], np.where(hist_an > 0)[0]
prob_d, prob_an = np.zeros_like(hist_d), np.zeros_like(hist_an)
prob_d[nzd] = 1.0 / (hist_d[nzd] / np.sum(hist_d))
prob_an[nza] = 1.0 / (hist_an[nza] / np.sum(hist_an))

probability = []

if len(distances) == len(positions):
    raise NotImplementedError
elif len(distances) == 2 * len(positions):
    for i in range(len(positions)):
        p_d1 = prob_d[np.where(edges_d[:-1] <= distances[2 * i])[0][-1]]
        p_d2 = prob_d[np.where(edges_d[:-1] <= distances[2 * i + 1])[0][-1]]
        p_an = prob_an[np.where(edges_an[:-1] <= angles[i])[0][-1]]
        probability.append(p_d1 * p_d2 * p_an)
else:
    raise ValueError

probability = np.array(probability)
probability = probability / np.sum(probability)
np.savez("prob_32at_3d.npz", probability=probability)
