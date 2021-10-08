import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

L = 0.5
eps = 0.5

# Types: 1 - square, 2 - circle

boundary = 3
plot_grid = False

if boundary == 1:
    tau = lambda t: np.where(t <= 2, np.where(t <= 1, -np.ones_like(t)  + 1j * (1 - t * 2), ((t - 1) * 2 - 1) - 1j*np.ones_like(t)),\
                                      np.where(t <= 3, np.ones_like(t) + 1j * ((t - 2) * 2 - 1), ((4 - t)*2 - 1) + 1j*np.ones_like(t)))
    dtaudt = lambda t: np.where(t <= 2, np.where(t <= 1, -1j * 2 * np.ones_like(t), 2 * np.ones_like(t)),\
                                         np.where(t <= 3, 1j * 2 * np.ones_like(t), - 2 * np.ones_like(t)))
    ddtauddt = lambda t: np.zeros_like(t)
elif boundary == 2:
    tau = lambda t: np.cos(np.pi * t / 2) + 1j * np.sin(np.pi * t / 2)
    dtaudt = lambda t: np.pi / 2 * (-np.sin(np.pi * t / 2) + 1j * np.cos(np.pi * t / 2))
    ddtauddt = lambda t: (np.pi / 2)**2 * (-np.cos(np.pi * t / 2) - 1j * np.sin(np.pi * t / 2))
elif boundary == 3:
    tau = lambda t: np.sin(2*np.pi*t/4) + 2 * 1j * np.sin(np.pi * t / 4) - 1j
    dtaudt = lambda t: np.cos(2*np.pi*t/4)*2*np.pi/4 + 1j * 2* np.pi/4 * np.cos(np.pi * t / 4)
    ddtauddt = lambda t: -np.sin(2*np.pi*t/4)*(2*np.pi/4)**2 - 2*1j * (np.pi/4)**2*np.sin(np.pi*t/4)
else:
    p = boundary
    curve = lambda t, p: (1 - np.abs(t) ** p) ** (1/p)
    curvep = lambda t, p: -t * np.abs(t) ** (p-2) * (1 - np.abs(t) ** p) ** (1/p - 1)
    curvepp = lambda t, p: (1-p) * np.abs(t) ** (p-2) * (1 - np.abs(t) ** p) ** (1/p - 1)

    tau = lambda t: np.where(t <= 2, (1-t) + 1j * curve(1 - t, p), (t-3) - 1j * curve(t - 3, p))
    dtaudt = lambda t: np.where(t <=2, -1 - 1j * curvep(1 - t, p), 1 - 1j * curve(t-3, p))
    ddtauddt = lambda t: np.where(t <= 2, 1j * curvep(1 - t, p),  -1j * curve(t - 3, p))

def mask_fun(X, Y):
    if boundary == 1:
        mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
    elif boundary == 2:
        mask = (X ** 2 + Y ** 2) <= 1 -eps
    elif boundary == 3:
        #T = np.arctan2(Y.flatten(), X.flatten()).reshape((n_grid, n_grid))
        #mask = ((X ** 2 + Y ** 2) <= (np.sin(T) * (1 + np.cos(T)**2)**0.5 - eps)**2)
        mask = (X < (1-((Y+1-eps)/2/(1-eps))**2)**0.5 * (Y+1-eps)) * (X > -(1-((Y+1-eps)/2/(1-eps))**2)**0.5 * (Y+1-eps))
    else:
        p = boundary
        #curve = lambda t, p: (1 - np.abs(t) ** p) ** (1 / p)
        #curvep = lambda t, p: -t * np.abs(t) ** (p - 2) * (1 - np.abs(t) ** p) ** (1 / p - 1)
        #curvepp = lambda t, p: (1 - p) * np.abs(t) ** (p - 2) * (1 - np.abs(t) ** p) ** (1 / p - 1)
        mask = (np.abs(X) ** p + np.abs(Y) ** p) ** (1/p) <= 1 - eps
    return mask


zp = 0 + 1.3j
F = lambda z: np.imag(z ** 2 / (z - zp))
f = lambda t: F(tau(t))


def solve(t, weights, with_gmres=True, **kwargs):
    n_pts = len(t)
    tau_t = tau(t)
    dtaudt_t = dtaudt(t)
    ddtauddt_t = ddtauddt(t)
    f_t = f(t)

    K = np.zeros((n_pts, n_pts))

    for n in range(n_pts):
        for m in range(n_pts):
            if m != n:
                K[n, m] = np.imag(dtaudt_t[m] / (tau_t[m] - tau_t[n]) / np.pi * weights[m])
            else:
                K[n, m] = np.imag(ddtauddt_t[m] / 2 / dtaudt_t[m] / np.pi * weights[m])

    if with_gmres:
        mu, info = gmres(np.eye(n_pts) + K, f_t, **kwargs)
    else:
        mu = np.linalg.solve(np.eye(n_pts) + K, f_t)
        info = None
    #eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
    #plt.scatter(np.real(eigvals), np.imag(eigvals))
    #mu = f_t
    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * weights/(tau_t - z) / np.pi), mu)
    return u, mu, np.eye(n_pts) + K, info

segments = np.linspace(0, 4, 17)
if boundary == 1:
    sharp_corners = np.isin(segments, np.array([0.,1.,2.,3.,4.])).astype(int)
if boundary == 2:
    #sharp_corners = np.isin(segments, np.array([0., 1., 2., 3., 4.])).astype(int)
    sharp_corners = np.isin(segments, np.array([])).astype(int)
if boundary == 3:
    sharp_corners = np.isin(segments, np.array([0., 4.])).astype(int)


errors = []
eig_mean = []
eig_std = []
npts = []
ns = list(range(0,20,2))
for n in ns:
    gridObject = GaussLegGrid(segments, sharp_corners)
    gridObject.refine_corners_nply(n)

    print(len(gridObject.segments)*16)
    gridpts, weights = gridObject.get_grid_and_weights()
    u, mu, syst, _ = solve(gridpts, weights, tol=1e-16)
    print("System solved")


    def f_slow(Z, fun):
        # Memory conserving u.
        U = np.zeros_like(Z)
        for i in range(U.shape[1]):
            print(f"F Eval: {i}/{U.shape[1]}", end="\r")
            U[:, i] = fun(Z[:, i, None]).flatten()
        return U

    # Prepare solution
    n_grid = 200
    x_mesh = np.linspace(-(1 + L), (1 + L), n_grid)
    y_mesh = np.linspace(-(1 + L), (1 + L), n_grid)

    X, Y = np.meshgrid(y_mesh, x_mesh)
    Z = X + 1j * Y

    U = f_slow(Z, u)
    Utru = f_slow(Z, F)
    er = np.abs(U - Utru)

    # Get mask
    mask = mask_fun(X, Y)
    er_mask = np.where(mask, er, np.nan)
    errors.append(np.nanmean(er_mask))

    grid, weights = gridObject.get_grid_and_weights()
    npts.append(len(grid))

    eigvals, eigvecs = np.linalg.eig(syst)
    #plt.scatter(np.real(eigvals), np.imag(eigvals))
    eig_mean.append(np.mean(np.real(eigvals)))
    eig_std.append(np.std(np.real(eigvals)))

x = np.linspace(0,2,200)

#for i in range(len(eig_mean)):
plt.plot(eig_mean)


plt.scatter(ns, eig_std, color="black")
plt.plot(ns, 0.0902/np.array(ns)**0.14, color="red")
plt.yscale("log")
plt.xscale("log")

plt.scatter(ns, errors, color='black', label="data")
plt.plot(ns, 1/(10**9.5)*np.exp(-1.35*np.array(ns)), color="red", label="$\\frac{3}{10^{10}} e^{-1.35 n}$")
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("Error")
plt.legend()

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.figure(1).savefig("boundary_integrals/figures/nply_error.pdf", bbox_inches="tight")

plt.scatter(ns, npts)
#plt.imshow(er_mask)