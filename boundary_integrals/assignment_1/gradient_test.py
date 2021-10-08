import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

L = 0.5
eps = 0.01

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
gradF = lambda z: 1j * np.conjugate(z*(z-2*zp)/(z-zp)**2)

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

    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * weights/(tau_t - z) / np.pi), mu)
    gradu = lambda z: np.dot(np.conjugate(dtaudt_t * weights / (tau_t - z)**2) * 1j/ np.pi, mu)
    return u, gradu, mu, np.eye(n_pts) + K, info

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
gridObject.refine_corners_nply(16)
#grid.segments

gridpts, weights = gridObject.get_grid_and_weights()
print(len(gridpts))

u, gradu, mu, _, _ = solve(gridpts, weights, tol=1e-16)

print("System solved")
def f_slow(Z, fun):
    # Memory conserving u.
    U = np.zeros(Z.shape)
    for i in range(U.shape[1]):
        print(f"F Eval: {i}/{U.shape[1]}", end="\r")
        U[:, i] = fun(Z[:, i, None]).flatten()
    return U

u = lambda z: np.imag(gradu(z))
F = lambda z: np.imag(gradF(z))



# Prepare solution
n_grid = 2000
x_mesh = np.linspace(-(1 + L), (1 + L), n_grid)
y_mesh = np.linspace(-(1 + L), (1 + L), n_grid)

X, Y = np.meshgrid(y_mesh, x_mesh)
Z = X + 1j * Y
#U = u(Z.flatten()[:,None]).reshape((n_grid, n_grid))
#Utru = F(Z.flatten()[:,None]).reshape((n_grid, n_grid))
U = f_slow(Z, u)
Utru = f_slow(Z, F)
er = np.abs(Utru - U)
#U = er
U = np.log10(er)
#U = np.where(U==-np.inf, -17*np.ones_like(U), U)
# Get mask
mask = mask_fun(X, Y)
U_mask = np.where(mask, U, np.nan)


Nb = 2000 # Resolution of boundary
tb = np.linspace(0, 4, Nb)
zb = tau(tb)
fb = f(tb)

# 2D plot
# Plot domain with boundary conditions

# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
# 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
# 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn',
# 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
# 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
# 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr',
# 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r',
# 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
# 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
# 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
# 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r',
# 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
# 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring',
# 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
# 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
# 'viridis_r', 'winter', 'winter_r'

# magma, cividis, gnuplot, coolwarm,inferno,seismic,summer,turbo
cmap = cm.get_cmap('gist_yarg_r', 6)
plt.figure(figsize=(8,7))
plt.pcolormesh(X, Y, U_mask, shading='nearest', label="Solution", cmap=cmap)#, vmax=2, vmin=-16)
plt.plot(np.real(zb), np.imag(zb), 'black', label="Boundary", linewidth=4)
#plt.scatter(x_,y_,c=f_, label="Basis functions")
plt.xlim([-(1+L+0.01), (1+L+0.01)])
plt.ylim([-(1+L+0.01), (1+L+0.01)])
remove_axis(plt.gca())
plt.colorbar()#location="bottom")
plt.tight_layout()

xy = tau(gridpts)
if plot_grid:
    plt.scatter(np.real(xy), np.imag(xy), color='black')#, c=10*gridpts ** 2)

xynorm = dtaudt(gridpts)/1j
xynorm = xynorm/np.abs(xynorm)

#plt.figure()
scale = 0.6
minmu = min(mu)
deltamu = max(mu) - minmu
muscaled = (mu-minmu)/deltamu*scale
mubound = xy + xynorm * muscaled
zerobound = xy - xynorm*minmu/deltamu *scale


fvals = F(xy)
minf = min(fvals)
deltaf = max(fvals) - minf
fscaled = (fvals - minmu)/deltamu*scale
fbound = xy + xynorm * fscaled



plt.scatter(np.real(zp), np.imag(zp), color="black", label="$z_p$", marker="x", s=50)
plt.legend(edgecolor='none')


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
