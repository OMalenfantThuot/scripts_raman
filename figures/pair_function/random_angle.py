import numpy as np
from mlcalcdriver import Posinp
import matplotlib.pyplot as plt

#pos = Posinp.from_file("data/surface.xyz")
pos = Posinp.from_file("data/posinp.xyz")

results = []

n = 200000
n_at = len(pos)

for z in range(n):
    i,j,k = np.random.choice(n_at, 3, replace=False)
    atom1, atom2, atom3 = pos[i], pos[j], pos[k]
    results.append(pos.angle(i,j,k))
    results.append(pos.angle(j,k,i))
    results.append(pos.angle(k,i,j))
    if z%10000 == 0:
        print(z)

results = np.array(results)
np.savetxt("random_angles.data", results)
#results=np.loadtxt("results_angles.data")
fig, ax = plt.subplots(figsize=[15,11])

ax.set_xlim(0, np.pi)
ax.hist(results, bins=60)
#plt.show()
plt.savefig("random_angle.jpg")
