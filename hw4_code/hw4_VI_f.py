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
orbit_steps = 600
orbit = [(theta0, I0)]
th, I = theta0, I0
for _ in range(orbit_steps):
    th, I = standard_map(th, I)
    orbit.append((th, I))
orbit = np.array(orbit)

# Stable + Unstable Manifolds
plt.figure(figsize=(8, 6))
plt.scatter(stable_points[:, 0], stable_points[:, 1], s=2, c='dodgerblue', label='Stable manifold')
plt.scatter(unstable_points[:, 0], unstable_points[:, 1], s=2, c='coral', label='Unstable manifold')
plt.scatter([theta_star], [I_star], color='lime', edgecolor='k', s=60, label='Fixed point')
plt.xlabel(r'$\theta$ (mod 1)')
plt.ylabel(r'$I$')
plt.title('Stable and Unstable Manifolds Forming a Homoclinic Tangle')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('hw4_VI_f.pdf')
plt.show()
