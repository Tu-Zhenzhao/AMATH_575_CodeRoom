import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# Parameters
K = 1.0
n_iter = 200  # fewer iterations for clarity
n_seeds = 5   # fewer seeds for visible trajectories

# Inverse map definition
def inverse_standard_map(theta, I):
    theta_prev = (theta - I) % 1.0
    I_prev = I + (K / (2 * math.pi)) * math.sin(2 * math.pi * theta_prev)
    return theta_prev, I_prev

# Fixed point and stable eigenvector
theta_star, I_star = 0.5, 1.0
v_s = np.array([-0.618, 1.0]) / np.linalg.norm([-0.618, 1.0])

# Small bundle of seeds along stable direction
epsilons = np.linspace(-2e-4, 2e-4, n_seeds)


# Settings
snapshots = [20, 40, 60, 80, 100, 120]  # Iteration steps to plot
cmap = get_cmap("plasma")  # Blue to orange colormap
n_colors = max(snapshots)

# Plot setup
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
axs = axs.flatten()

# Iterate over each snapshot step
for ax_idx, max_iter in enumerate(snapshots):
    ax = axs[ax_idx]
    for eps in epsilons:
        theta_0 = theta_star + eps * v_s[0]
        I_0 = I_star + eps * v_s[1]
        theta, I = theta_0, I_0
        traj = [(theta, I)]
        for _ in range(max_iter):
            theta, I = inverse_standard_map(theta, I)
            traj.append((theta, I))
        traj = np.array(traj)

        # Plot each segment with color gradient
        for i in range(len(traj)-1):
            c = cmap(i / n_colors)
            ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=c, linewidth=1)

    ax.set_title(f'{max_iter} Iterations')
    ax.set_xlabel(r'$\theta$ (mod 1)')
    ax.set_ylabel(r'$I$')
    ax.grid(True)

plt.tight_layout()
plt.suptitle("Evolution of Stable Manifold Trajectories Over Time", fontsize=16, y=1.02)
plt.savefig("hw4_VI_e.pdf")
plt.show()
