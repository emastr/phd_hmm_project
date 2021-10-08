import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

L = 0.5
eps = 0.3

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
gridObject = GaussLegGrid(segments, sharp_corners)

#grid.refine()
#gridObject.refine_all_nply(0)
gridObject.refine_corners_nply(0)
#grid.segments

gridpts, weights = gridObject.get_grid_and_weights()
print(len(gridpts))

u, mu, _, _ = solve(gridpts, weights, tol=1e-16)

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
#U = u(Z.flatten()[:,None]).reshape((n_grid, n_grid))
#Utru = F(Z.flatten()[:,None]).reshape((n_grid, n_grid))
U = f_slow(Z, u)
Utru = f_slow(Z, F)
er = np.abs(Utru - U)

U = np.log10(er)
U = np.where(U==-np.inf, -17*np.ones_like(U), U)
# Get mask
mask = mask_fun(X, Y)
U_mask = np.where(mask, U, np.nan)
print(np.nanmean(10 ** U_mask), np.nanmax(10 ** U_mask), np.nanmin(10 ** U_mask))

Nb = 2000 # Resolution of boundary
tb = np.linspace(0, 4, Nb)
zb = tau(tb)
fb = f(tb)


# magma, cividis, gnuplot, coolwarm,inferno,seismic,summer,turbo
cmap = cm.get_cmap('gist_yarg_r', 6)
plt.figure(figsize=(7,7))
plt.pcolormesh(X, Y, U_mask, shading='nearest', vmax=0, vmin=-16, cmap=cmap, label="Measurement")
plt.plot(np.real(zb), np.imag(zb), 'black', label="Boundary", linewidth=4)
#plt.scatter(x_,y_,c=f_, label="Basis functions")
plt.xlim([-(1+L+0.01), (1+L+0.01)])
plt.ylim([-(1+L+0.01), (1+L+0.01)])
remove_axis(plt.gca())
#plt.colorbar()#location="bottom")
plt.tight_layout()

xynorm = dtaudt(gridpts)/1j
xynorm = xynorm/np.abs(xynorm)

#plt.plot(np.real(xy), np.imag(xy), label="Boundary")
corners = segments[np.argwhere(sharp_corners==1)]
#set it up so that one corner is always at the ends

plt.legend(edgecolor='none')
#plt.figure(1).savefig("boundary_integrals/figures/sample_domain.png")
#plt.plot(mu)
#plt.plot(mu*0)
#plt.stem(grid.segments, fun(grid.segments))
#plt.stem(gridpts, weights)
#plt.stem(gridpts, fun(gridpts))
#grid.integrate_func(fun, gridpts, weights)

# N = 200
# t = np.linspace(0, np.pi, N)
# z = np.sin(2*t)/2 + 1j * np.sin(t)
# plt.plot(np.real(z), np.imag(z))
# a = np.array([1,0,0,1,0,1,0,1,1])
# a = np.array([])
# i_nt = np.array([0,2,3,5,7,9,11,13,15])
# n = len(a)
# N = n + sum(a * 2 - a * np.roll(a, 1))
# i = np.arange(0,n,1)
# d_i = i + np.roll(np.cumsum(a),1) + np.roll(np.cumsum(a),0) - 1 - np.cumsum(np.roll(a,1) * a) + a[-1]*a[0]
# d_i[0] = 0
# A_test = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1,0, 0,0, 1, 0, 1, 0])
# A = np.zeros(N)
# A[d_i] = a
# np.abs(A-A_test).sum()

# img = np.where(np.isnan(U_mask), np.zeros_like(U_mask), U_mask)
# img = (-img)/11
# plt.imshow(img, cmap="Greys_r")

# def A(X):
#     return X
#
# def Aapx(X):
#     new = np.zeros_like(X)
#     for i in range(0,n_grid,2):
#         new[i, :] = X[i, :]
#     return new
#
# def D(X):
#     new = np.zeros_like(X)
#     for i in range(n_grid-1):
#         new[i,:] = X[i, :] - X[i+1, :]
#     return new
#
# def Dadj(X):
#     new = np.zeros_like(X)
#     for i in range(1,n_grid):
#         new[i,:] = X[i,:] - X[i-1,:]
#     return new
# <x | Ay> = <A'x |y>  -> summa(y(j)(x(j)-x(j+1))) = x(j)*(y(j)-y(j-1))

#plt.imshow(Aapx(img))


# y = A(img)
#
# plt.figure(figsize=(15,6))
# plt.subplot(1,3,1)
# plt.title("Accurate op $\mathcal{F}(x)$")
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(1,3,2)
# plt.title("Approximate op $\widetilde{\mathcal{F}}(x)$")
# plt.imshow(Aapx(img))
# plt.xticks([])
# plt.yticks([])
#
# x = np.zeros_like(img)
# for i in range(101):
#     #x = x - 0.5*(Aapx(Aapx(x))+0.1*Dadj(D(x)) - Aapx(y))
#     x = x - 0.5 * (Aapx(A(x)) - Aapx(y))
# plt.subplot(1,3,3)
# plt.title("Best possible learned recon")
# plt.imshow(x, vmin=0, vmax=1)
# plt.xticks([])
# plt.yticks([])

#min( (Ax - y)'(Ax - y) + x'D'Dx) -> min(x'(A'A+D'D)x - 2x'A'y) -> ((A'A+D'D)x - A'y) = grad(F) -> x - a * grad(F) = x - a*A'(Ax-y)
# min( (Ax - y)'(Ax - y)) -> min(x'A'Ax - 2x'A'y) -> A'(Ax - y) = grad(F) -> x - a * grad(F) = x - a*A'(Ax-y)
