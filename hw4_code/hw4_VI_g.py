import numpy as np
import matplotlib.pyplot as plt
import math

# Standard map parameters
K = 1.0

def standard_map(theta, I):
    theta_next = (theta + I - (K/(2*math.pi))*math.sin(2*math.pi*theta)) % 1.0
    I_next = I - (K/(2*math.pi))*math.sin(2*math.pi*theta)
    return theta_next, I_next

def inverse_standard_map(theta, I):
    theta_prev = (theta - I) % 1.0
    I_prev = I + (K/(2*math.pi))*math.sin(2*math.pi*theta_prev)
    return theta_prev, I_prev

# Fixed point and eigenvectors
theta_star, I_star = 0.5, 1.0
v_u = np.array([1.618, 1.0]); v_u /= np.linalg.norm(v_u)
v_s = np.array([-0.618, 1.0]); v_s /= np.linalg.norm(v_s)

# Manifold parameters
n_seeds = 40
eps_range = 2e-4
epsilons_u = np.linspace(-eps_range, eps_range, n_seeds)
epsilons_s = np.linspace(-eps_range, eps_range, n_seeds)

n_iter = 600

# Unstable manifold (forward iterations)
unstable_points = []
for eps in epsilons_u:
    th, I = theta_star + eps*v_u[0], I_star + eps*v_u[1]
    for _ in range(n_iter):
        th, I = standard_map(th, I)
        unstable_points.append((th, I))
unstable_points = np.array(unstable_points)

# Stable manifold (inverse iterations)
stable_points = []
for eps in epsilons_s:
    th, I = theta_star + eps*v_s[0], I_star + eps*v_s[1]
    for _ in range(n_iter):
        th, I = inverse_standard_map(th, I)
        stable_points.append((th, I))
stable_points = np.array(stable_points)

# Orbit of a single interior point
theta0, I0 = theta_star + 0.015, I_star - 0.015  # inside tangle but not on manifolds
orbit_steps = 400
orbit = [(theta0, I0)]
th, I = theta0, I0
for _ in range(orbit_steps):
    th, I = standard_map(th, I)
    orbit.append((th, I))
orbit = np.array(orbit)


# Create side-by-side plots (1 row, 2 columns)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: orbit only
axs[0].plot(orbit[:, 0], orbit[:, 1], color='mediumseagreen', linewidth=1.5, label='Orbit inside tangle')
axs[0].scatter([theta_star], [I_star], color='lime', edgecolor='k', s=60, label='Fixed point')
axs[0].set_title('Trajectory of a Point Inside the Tangle')
axs[0].set_xlabel(r'$\theta$ (mod 1)')
axs[0].set_ylabel(r'$I$')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Right plot: orbit + light stable and unstable manifolds
axs[1].scatter(stable_points[:, 0], stable_points[:, 1], s=1, c='lightblue', label='Stable manifold (light)')
axs[1].scatter(unstable_points[:, 0], unstable_points[:, 1], s=1, c='lightcoral', label='Unstable manifold (light)')
axs[1].plot(orbit[:, 0], orbit[:, 1], color='navajowhite', linewidth=1.5, label='Orbit inside tangle')
axs[1].scatter([theta_star], [I_star], color='lime', edgecolor='k', s=60, label='Fixed point')
axs[1].set_title('Orbit with Stable/Unstable Manifolds Overlay')
axs[1].set_xlabel(r'$\theta$ (mod 1)')
axs[1].set_ylabel(r'$I$')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.savefig('hw4_VI_g.pdf')
plt.show()
