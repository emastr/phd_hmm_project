import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *

eps = 0.00
epsbound = 0.00

L = 0.3
M = 5

tau = lambda t: (1 + L * np.cos(M * t)) * np.exp(1j * t)
dtaudt = lambda t: (- L * M * np.sin(M * t) \
                    + 1j * (1 + L * np.cos(M * t))) * np.exp(1j * t)
ddtaudt2 = lambda t: (-L * M * M * np.cos(M * t) \
                      - 2 * L * M * np.sin(M * t) * 1j \
                      - 1 - L * np.cos(M * t)) * np.exp(1j * t)
zp = 1 + 1j
F = lambda z: np.imag(z ** 2 / (z - zp))
f = lambda t: F(tau(t))


def solve(n_pts):
    t = np.linspace(0, 2 * np.pi, n_pts + 1)
    dt = t[1:] - t[:-1]
    t = t[:-1]
    tau_t = tau(t)
    dtaudt_t = dtaudt(t)
    ddtaudt2_t = ddtaudt2(t)
    f_t = f(t)

    K = np.zeros((n_pts, n_pts))

    for n in range(n_pts):
        for m in range(n_pts):
            if m != n:
                K[n, m] = np.imag(dtaudt_t[m] / (tau_t[m] - tau_t[n]) / np.pi * dt[m])
            else:
                K[n, m] = np.imag(ddtaudt2_t[m] / dtaudt_t[m] / np.pi * dt[m])

    mu = np.linalg.solve(np.eye(n_pts) + K, f_t)
    # eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
    # plt.scatter(np.real(eigvals), np.imag(eigvals))

    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * dt / (tau_t - z) / np.pi), mu)
    return u


# Plot radial error 3d
m = 2
t_fix = 26 / 64 * 2 * np.pi
N = 1000

z_fix = tau(t_fix)
radii = np.linspace(0.1, 0.9, N)
z = z_fix * radii

# t = np.linspace(0, 2*pi + 64)
# plt.scatter(np.real(z[-1]), np.imag(z[-1]))
# plt.scatter(np.real(zb), np.imag(zb))

N_pts = 7
n_pts_grid = 64 * 2 ** np.arange(0, N_pts, 1)
ers = np.zeros_like(n_pts_grid).astype(float)
u_tru = F(z)

for idx, n_pts in enumerate(n_pts_grid):
    # for n_pts in n_pts_grid:
    u = solve(n_pts)
    u_apx = u(z[:, None])

    # Find the distance from the boundary such that the relative error is lower than 0.1.
    rel_err = np.abs(u_apx - u_tru) / u_tru
    ers[idx] = np.sum(rel_err)
    # plt.plot(rel_err)

plt.loglog(n_pts_grid, ers)


for n in 2 ** np.arange(2,6,1):
    A = -2*np.eye(n)
    for m in range(n-1):
        A[m, m+1] = 1
        A[m+1, m] = 1
    A = A
    eigs, _ = np.linalg.eig(A)
    abseigs = np.abs(eigs)
    kond = abseigs.max() / abseigs.min()
    plt.scatter(n, kond)

#plt.xscale('log')
#plt.yscale('log')