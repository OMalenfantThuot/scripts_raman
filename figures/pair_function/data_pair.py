import numpy as np
import matplotlib.pyplot as plt

results = np.loadtxt("distances.data")

fig, ax = plt.subplots()

ax.set_xlim(0, 6)
ax.hist(results, bins=40)
#plt.show()
plt.savefig("data_pairs.jpg")
