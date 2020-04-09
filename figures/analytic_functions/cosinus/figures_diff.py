import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_init = 0.05
x_final = 6.2

x = np.linspace(x_init, x_final, 10 ** 6)
energy = np.loadtxt("data/ener_100.data")
forces = np.loadtxt("data/forces_100.data")

# Theoric Values
ener_th = np.cos(x)
forces_th = np.sin(x)

# Figure
fig, ax = plt.subplots(figsize=[15, 11], ncols=2)
ax1, ax2 = ax.flatten()
fig.subplots_adjust(hspace=0, wspace=0)

ax1.set_xlabel("Distance", fontsize=30)
ax1.set_ylabel("Error", fontsize=30)
ax2.set_xlabel("Distance", fontsize=30)

ax1.grid(True, linestyle="-.")
ax2.grid(True, linestyle="-.")

ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20, right=True, labelright=True, left=False, labelleft=False)

ax1.plot(x, energy - ener_th, "-k", linewidth=2, label="Value Error")
ax2.plot(x, forces - forces_th, "r", label="Derivative Error")

ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(-0.1, 0.1)
ax2.set_xlim(0, 2 * np.pi)
ax2.set_ylim(-0.1, 0.1)

fig.legend(loc="upper center", fontsize=25)

plt.savefig("errors.png", dpi=100)
# plt.show()
