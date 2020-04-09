import numpy as np
import matplotlib.pyplot as plt

def lj(x):
    return 4 * ((1./x) ** 12 - (1./x) ** 6)

energy_def = np.loadtxt("data/energy_def.data")
energy_up = np.loadtxt("data/energy_up.data")
energy_down = np.loadtxt("data/energy_down.data")
distances = np.linspace(0.85, 1.95, 10**4)

# Theoric Values
x = np.linspace(0.85, 1.95, 10000)
theory = lj(x)

# Figure
fig, ax = plt.subplots(ncols=3,figsize=[20, 11])
ax1, ax2, ax3 = ax.flatten()

ax1.set_xlabel("Distance", fontsize=30)
ax1.set_ylabel("Function value", fontsize=30)

ax1.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax1.plot(distances, energy_def, "r", label="Model Results")

ax2.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax2.plot(distances, energy_up, "r", label="Model Results")

ax3.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax3.plot(distances, energy_down, "r", label="Model Results")

# Pour figure complete
#for a in ax:
#    a.set_xlim(0.8, 2)
#    a.set_ylim(-1.05, 1.3)
#    a.set_xlabel("Distance", fontsize=20)
#plt.show()
#plt.savefig("lj_energy.png", dpi=100)

# Pour figure zoom
for a in ax:
    a.set_xlim(1.03, 1.3)
    a.set_ylim(-1.05, -0.4)
    a.set_xlabel("Distance", fontsize=20)

#plt.show()
plt.savefig("lj_energy_zoom.png", dpi=100)
