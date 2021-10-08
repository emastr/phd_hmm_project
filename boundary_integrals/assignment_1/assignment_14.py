import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres


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

def solve(n_pts, **kwargs):
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
                K[n, m] = np.imag(ddtaudt2_t[m] / 2/dtaudt_t[m] / np.pi * dt[m])


    #mu = np.linalg.solve(np.eye(n_pts) + K, f_t)
    mu, info = gmres(np.eye(n_pts) + K, f_t, **kwargs)
    #eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
    #plt.scatter(np.real(eigvals), np.imag(eigvals))

    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * dt/(tau_t - z) / np.pi), mu)
    return u, np.eye(n_pts) + K, info

m = 2
t_fix = 26/64 * 2 * np.pi
N = 1000

z_fix = tau(t_fix)
radii = 1 - np.logspace(-1, -10, N)
z = z_fix * radii


N_pts = 6
n_pts_grid = 64 * 2 ** np.arange(0, N_pts, 1)
u_tru = F(z)

class CallbackCounter:
    def __init__(self):
        self.counts = 0

    def __call__(self, *args):
        self.counts += 1

    def reset(self):
        self.counts = 0

    def count(self):
        return self.counts

counter = CallbackCounter()

condvals = np.zeros(n_pts_grid.shape)
means = []
stds = []
eigvallist=[]
stepslist=[]
timelist = []
import  time
for idx, n_pts in enumerate(n_pts_grid):
    #for n_pts in n_pts_grid:
    counter.reset()
    assert counter.count() == 0
    dt =  time.time()
    u, sysmat, info = solve(n_pts, callback=counter, tol=1e-16)
    dt = time.time() - dt
    timelist.append(dt)
    eigvals, eigvecs = np.linalg.eig(sysmat)
    abseig = np.abs(eigvals)
    condvals[idx] = abseig.max() / abseig.min()
    print(condvals[idx], n_pts, counter.count())
    plt.scatter(np.ones_like(eigvals)*idx, np.real(eigvals))
    means.append(np.mean(eigvals))
    stds.append(np.std(eigvals))
    eigvallist.append(eigvals)
    stepslist.append(counter.count())


plt.plot(n_pts_grid, stepslist)
plt.plot(n_pts_grid, timelist)
fig = plt.figure()
plt.plot(n_pts_grid, condvals)
plt.xlabel("Number of grid points")
plt.ylabel("System condition number")
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.savefig("boundary_integrals/figures/condition_no.pdf", bbox_inches='tight')


x = np.linspace(0,2,500)
for i in [0,2,4]:
    yi = 1/(2*np.pi*stds[i])**2 * np.exp(-(x-means[i])**2 / 2 / stds[i]**2)
    plt.plot(x,yi, label=f"n={n_pts_grid[i]}")
plt.legend(edgecolor='none')
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.yticks([])


boxpos = list(range(len(n_pts_grid)))
plt.boxplot(x=eigvallist, positions=boxpos, showfliers=False)
plt.xticks(boxpos, n_pts_grid)


plt.plot(n_pts_grid,1.85/n_pts_grid**0.5, 'red',label="$y = 1.85/\sqrt{x}$")
plt.scatter(n_pts_grid, stds,color='black',label='Data')
plt.yscale("log")
plt.xscale("log")
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylabel("standard deviation")
plt.xlabel("Number of gridpoints")
plt.title("Standard Deviation of System Eigenvalues")
plt.legend(edgecolor='none')
plt.tight_layout()
plt.figure(1).savefig("boundary_integrals/figures/grouping_of_eigvals.pdf", bbox_inches="tight")