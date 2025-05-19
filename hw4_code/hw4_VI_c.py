import numpy as np
import matplotlib.pyplot as plt
import math

K = 1.0

def standard_map(theta, I):
    """One iteration of the standard map with K = 1."""
    theta_next = theta + I - (K / (2 * math.pi)) * math.sin(2 * math.pi * theta)
    theta_next = theta_next % 1.0           # keep θ on the unit circle
    I_next = I - (K / (2 * math.pi)) * math.sin(2 * math.pi * theta)
    return theta_next, I_next

theta_star, I_star = 0.5, 1.0

# Jacobian at the fixed point (θ*, I*)
A = 2          # 1 - K cos(2πθ*) with cos π = -1, K=1
B = 1
C = 1
D = 1
J = np.array([[A, B], [C, D]])

# Eigen‑values & vectors (stable first, unstable second)
lam, vec = np.linalg.eig(J)
idx = np.argsort(lam)          # sort so lam[0]<1<lam[1]
lam_s, lam_u = lam[idx]
v_s, v_u = vec[:, idx[0]], vec[:, idx[1]]

# normalise eigenvectors
v_s = v_s / np.linalg.norm(v_s)
v_u = v_u / np.linalg.norm(v_u)

epsilons = np.linspace(-1e-4, 1e-4, 25)     # tiny displacements along v_u
n_iter = 600

theta_vals, I_vals = [], []
for ε in epsilons:
    θ0 = theta_star + ε * v_u[0]
    I0 = I_star   + ε * v_u[1]
    θ, I = θ0, I0
    for _ in range(n_iter):
        θ, I = standard_map(θ, I)
        theta_vals.append(θ)
        I_vals.append(I)

# plot the manifold
plt.figure(figsize=(7, 7))
plt.scatter(theta_vals, I_vals, s=2)
plt.xlabel(r'$\theta$  (mod 1)')
plt.ylabel(r'$I$')
plt.title('Global unstable manifold of the fixed point $(\\frac{1}{2},\,1)$')
plt.savefig('hw4_VI_c.pdf')
plt.show()


# save in pdf

