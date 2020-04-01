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

n = 200000
npzfile = np.load("prob_32at_3d.npz")
probability = npzfile["probability"]
assert len(positions) == len(probability)

distances, angles = [], []

for _ in range(n):
    i = int(np.random.choice(len(positions), 1, p=probability))
    posinp = positions[i]
    az_idx = [i for i, at in enumerate(posinp) if at.type == "N"]
    if len(az_idx) == 2:
        distances.append(posinp.distance(az_idx[0], az_idx[1]))
        angles.append(np.pi)
    elif len(az_idx) == 3:
        mid = np.array(posinp.cell) / 2
        d = np.array([np.linalg.norm(posinp[idx].position - mid) for idx in az_idx])
        mid_idx = az_idx[np.argmin(d)]
        other_idx = copy(az_idx)
        del other_idx[np.argmin(d)]

        distances.append(posinp.distance(mid_idx, other_idx[0]))
        distances.append(posinp.distance(mid_idx, other_idx[1]))
        angles.append(posinp.angle(other_idx[0], mid_idx, other_idx[1]))
    else:
        raise NotImplementedError

# Uncomment for visual

fig, ax = plt.subplots(ncols=2, figsize=[15, 11])
ax1, ax2 = ax.flatten()

ax1.set_xlim(0, 7)
ax1.hist(distances, bins=20)
ax1.set_title("Distance distribution")

ax2.set_xlim(0, np.pi)
ax2.hist(angles, bins=20)
ax2.set_title("Angle distribution")
plt.show()
