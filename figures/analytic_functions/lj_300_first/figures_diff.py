import numpy as np
import matplotlib.pyplot as plt

def dljdx(x):
    return 4 * (-12 * (1./x) ** 13 + 6 * (1./x) ** 7)

forces_def = np.loadtxt("data/forces_def.data")
forces_up = np.loadtxt("data/forces_up.data")
forces_down = np.loadtxt("data/forces_down.data")
distances = np.linspace(0.85, 1.95, 10**4)

# Theoric Values
x = np.linspace(0.85, 1.95, 10000)
theory = dljdx(x)

# Figure
fig, ax = plt.subplots(figsize=[20, 11], ncols=3)
ax1, ax2, ax3 = ax.flatten()
fig.subplots_adjust(hspace=0, wspace=0)

ax1.set_ylabel("Error", fontsize=30)

for a in ax:
    a.grid(True, linestyle="-.")
    a.set_xlabel("Distance", fontsize=30)
    a.set_ylim(-0.5, 1.2)

ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20, right=True, labelright=True, left=False, labelleft=False)
ax3.tick_params(labelsize=20, right=True, labelright=True, left=False, labelleft=False)

ax1.plot(x, -1.0 * forces_def - theory, "k", linewidth=2, label="Default loss")
ax2.plot(x, -1.0 * forces_down - theory, "r", label="Down loss")
ax3.plot(x, -1.0 * forces_up - theory, "b", label="Up loss")

fig.legend(loc="upper center", fontsize=25)

plt.savefig("errors.png", dpi=100)
#plt.show()
