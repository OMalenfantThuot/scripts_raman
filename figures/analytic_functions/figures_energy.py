import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_init = 0.05
x_final = 6.2
n_points = 100

x = np.linspace(x_init, x_final, 10 ** 6)
training_points = np.linspace(x_init, x_final, n_points)
energy = np.loadtxt("data/ener_100.data")

# Theoric Values
theory = np.cos(x)
training_values = np.cos(training_points)

# Figure
fig, ax = plt.subplots(figsize=[15, 11])
ax.set_xlabel("Distance", fontsize=30)
ax.set_ylabel("Function value", fontsize=30)
ax.tick_params(labelsize=20)

ax.plot(x, theory, "--k", linewidth=2, label="Analytic Function")
ax.plot(x, energy, "r", label="Model Results")

# Pour figure complete
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.1, 1.1)
ax.legend(loc="upper center", fontsize=25)
ax.plot([2.6, 3.1, 3.1, 2.6, 2.6],[-0.85, -0.85, -1.03, -1.03, -0.85], "-k")
plt.savefig("value_rec.png", dpi=100)

# Pour figure zoom
# ax.set_xlim(2.6, 3.1)
# ax.set_ylim(-1.03, -0.85)
# ax.plot(
#     training_points,
#     training_values,
#     "+b",
#     markersize=20,
#     linewidth=2,
#     label="Training Points",
# )
# ax.legend(loc="upper right", fontsize=25)
# plt.savefig("zoom_value.png", dpi=100)

# plt.show()
