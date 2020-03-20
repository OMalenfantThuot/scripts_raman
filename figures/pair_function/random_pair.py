import numpy as np
from mlcalcdriver import Posinp
import matplotlib.pyplot as plt


pos = Posinp.from_file("data/posinp.xyz")

translations = (
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 0, 1],
            [1, 0, -1],
            [-1, 0, 1],
            [-1, 0, -1],
        ]
    )
    * pos.cell
)

results = []
n = 50000
n_at = len(pos)

for _ in range(n):
    i,j = np.random.choice(n_at, 2, replace=False)
    atom1, atom2 = pos[i], pos[j]
    all_pos = atom2.position + translations
    distances = np.sqrt(np.sum((atom1.position - all_pos) ** 2, axis=1)).tolist()
    results.append(distances)

results = np.array(results).flatten()
results = results[np.where(results<15)[0]]

fig, ax = plt.subplots()

ax.set_xlim(0, 15)
ax.hist(results, bins=38)
#plt.show()
plt.savefig("pair_correlation.jpg")
