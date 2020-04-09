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
fig, ax = plt.subplots(ncols=3,figsize=[20, 11])
ax1, ax2, ax3 = ax.flatten()

ax1.set_xlabel("Distance", fontsize=30)
ax1.set_ylabel("Function value", fontsize=30)

ax1.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax1.plot(distances, -1.0 * forces_def, "r", label="Model Results")

ax2.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax2.plot(distances, -1.0 * forces_up, "r", label="Model Results")

ax3.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax3.plot(distances, -1.0 * forces_down, "r", label="Model Results")

# Pour figure complete
#for a in ax:
#    a.set_xlim(0.8, 2)
#    a.set_ylim(-100, 5)
#    a.set_xlabel("Distance", fontsize=20)
#plt.show()
#plt.savefig("lj_forces.png", dpi=100)

# Pour figure zoom
for a in ax:
    a.set_xlim(1.03, 1.3)
    a.set_ylim(-10, 3)
    a.set_xlabel("Distance", fontsize=20)

#plt.show()
plt.savefig("lj_forces_zoom.png", dpi=100)
