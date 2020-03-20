import numpy as np
from mlcalcdriver import Posinp
import matplotlib.pyplot as plt

##pos = Posinp.from_file("data/surface.xyz")
#pos = Posinp.from_file("data/posinp.xyz")
#
#results = []
#
#for i, atom1 in enumerate(pos):
#    print(i)
#    for j, atom2 in enumerate(pos):
#        if atom1 is atom2:
#            pass
#        else:
#            for k, atom3 in enumerate(pos):
#                if atom3 is atom1 or atom3 is atom2:
#                    pass
#                else:
#                    results.append(pos.angle(i,j,k))
#                    results.append(pos.angle(j,k,i))
#                    results.append(pos.angle(k,i,j))
#
#results = np.array(results)
#np.savetxt("results_angles.data", results)
#results = results[np.where(results<15)[0]]
results=np.loadtxt("results_angles.data")
fig, ax = plt.subplots(figsize=[15,11])

ax.set_xlim(0, np.pi)
ax.hist(results, bins=60)
#plt.show()
plt.savefig("pair_angle.jpg")
