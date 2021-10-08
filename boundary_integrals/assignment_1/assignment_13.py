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
dtaudt = lambda t: (- L * M * np.sin(M * t)\
                    + 1j * (1 + L * np.cos(M * t))) * np.exp(1j * t)
ddtaudt2 = lambda t: (-L * M * M * np.cos(M * t) \
                      - 2 * L * M * np.sin(M * t) * 1j \
                      - 1 - L * np.cos(M * t)) * np.exp(1j * t)
zp = 1 + 1j
F = lambda z: np.imag(z ** 2 / (z - zp))
f = lambda t: F(tau(t))

def solve(n_pts):
    t = np.linspace(0, 2*np.pi, n_pts+1)
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
                K[n, m] = np.imag(ddtaudt2_t[m] / 2 / dtaudt_t[m] / np.pi * dt[m])


    mu = np.linalg.solve(np.eye(n_pts) + K, f_t)
    #eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
    #plt.scatter(np.real(eigvals), np.imag(eigvals))

    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * dt/(tau_t - z) / np.pi), mu)
    return u, np.eye(n_pts) + K

m = 2
t_fix = 26/64 * 2 * np.pi
N = 1000

z_fix = tau(t_fix)
radii = 1 - np.logspace(-1, -10, N)
z = z_fix * radii


N_pts = 7
n_pts_grid = 64 * 2 ** np.arange(0, N_pts, 1)
u_tru = F(z)

condvals = np.zeros(n_pts_grid.shape)
for idx, n_pts in enumerate(n_pts_grid):
    #for n_pts in n_pts_grid:
    u, sysmat = solve(n_pts)
    eigvals, eigvecs = np.linalg.eig(sysmat)
    abseig = np.abs(eigvals)
    condvals[idx] = abseig.max() / abseig.min()
    print(condvals[idx], n_pts)
    plt.scatter(np.ones_like(eigvals)*idx, np.real(eigvals))

fig = plt.figure()
plt.plot(n_pts_grid, condvals)
plt.xlabel("Number of grid points")
plt.ylabel("System condition number")
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.savefig("boundary_integrals/figures/condition_no.pdf", bbox_inches='tight')

