import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

# Grid refinement
n_corner_refine = 0
n_all_refine = 0
# Plotting options
L = 0.5
eps = 0.05
plot_grid = False
# Settings for boundary + conditions

# 1 - pipe flow, set inlet and outlet radii
# 2 - stirring boundary condition, set u_tangential
# 3 - Flow across surface with stream velocity at top boundary. Boundary is box with rough boundary
# 4 - load cos-sine series from folder
in_rad = 0.2      # Radius at inlet
out_rad = 1.0          # Radius at outlet
vol_in = 1.0        # Volume flowing through pipe per time unit. For setting 1 only
u_tangental = 1.0   # Max stirring velocity, for setting 2,3 only
roughness = 21      # Roughness of bottom boundary in setting 3
amplitude = 0.02     # Amplitude of bottom boundary in setting 3

w = 1j * np.pi
w2 = np.pi
def tau(t):
    tau_t = np.where(t <= 2, \
                     out_rad * np.exp(w*t), \
                     0.5 + in_rad * np.exp(-w*t) * (1 + 0.5 * np.sin(2 * w2 * t)))
    return tau_t

def dtaudt(t):
    dtaudt_t = np.where(t <= 2, \
                        w * out_rad * np.exp(w*t), \
                        -w * in_rad * np.exp(-w*t)*(1 + 0.5 * (np.sin(2 * w2 * t) + 2*w2*np.cos(2 * w2 * t)/(-w))))
    return dtaudt_t

def ddtauddt(t):
    ddtauddt_t = np.where(t <= 2, \
                          w**2 * out_rad * np.exp(w * t), \
                          w**2 * in_rad * np.exp(-w * t)*(1 + 0.5 * (np.sin(2 * w2 * t) + 4*w2*np.cos(2 * w2 * t)/(-w) - (2*w2)**2*np.sin(2 * w2 * t)/w**2)))
    return ddtauddt_t

def mask_fun(X, Y):
    R = X ** 2 + Y ** 2
    mask = (R < (out_rad - eps)**2) * (R > (in_rad + eps)**2)
    return mask


def limit_coef(t):
    return -1 + 2 * (t < 2)
    #return np.ones_like(t)


# Tangental BC
f = lambda t: np.where(t<2, dtaudt(t), np.zeros_like(t)) * 1.0
F = None



#F = lambda z: -u_inout * (np.imag(z)-1)*(np.imag(z)+1)
def solve(t, weights, with_gmres=True, **kwargs):
    n_pts = len(t)
    tau_t = tau(t)
    dtaudt_t = dtaudt(t)
    dtaudt_t_conj = np.conjugate(dtaudt_t)
    ddtauddt_t = ddtauddt(t)
    f_t = f(t)

    K = np.zeros((n_pts, n_pts)).astype(np.complex128)
    Kconj = np.zeros((n_pts, n_pts)).astype(np.complex128)
    sign = limit_coef(t)
    for n in range(n_pts):
        for m in range(n_pts):
            if m != n:
                K[n, m] = np.imag(dtaudt_t[m] / (tau_t[m] - tau_t[n]) / np.pi * weights[m])
                Kconj[n, m] = -np.imag(dtaudt_t[m] * np.conjugate(tau_t[m] - tau_t[n]))/\
                              np.conjugate(tau_t[m] - tau_t[n])**2 / np.pi * weights[m]
            else:
                K[n, m] = sign[n]*np.imag(ddtauddt_t[m] / (2 * dtaudt_t[m])) /np.pi * weights[m]
                Kconj[n, m] = sign[n]*np.imag(ddtauddt_t[m] * dtaudt_t_conj[m]) / (2 * dtaudt_t_conj[m]**2) /np.pi * weights[m]

    A = np.diag(limit_coef(t)) + K
    Aconj = Kconj
    sysMat = np.vstack([np.hstack([np.real(A) + np.real(Aconj),-np.imag(A) + np.imag(Aconj)]),
                        np.hstack([np.imag(A) + np.imag(Aconj), np.real(A) - np.real(Aconj)])])

    # Assert boundary integral is zero (or sufficiently close to it)
    assert np.abs(np.imag(np.sum(np.conjugate(f_t) * weights * dtaudt_t))) < 1e-12, "Boundary integral not zero"
    b = 1j*f_t
    sysVec = np.vstack([np.real(b)[:,None], np.imag(b)[:,None]])
    # SYSTEM:
    # Ax + Bxc = b -> (Ar + jAi)(xr+jxi) + (Br + jBi)(xr - jxi) = br + jbi
    # (Arxr - Aixi + Brxr + Bixi) + j(Arxi + Aixr + Bixr - Brxi) = br + jbi

    if with_gmres:
        omega, info = gmres(sysMat, sysVec, **kwargs)
    else:
        omega = np.linalg.solve(sysMat, sysVec)
        info = None

    omega = omega[0:n_pts] + 1j * omega[n_pts:2*n_pts]
    #mu = f_t
    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t / (tau_t - z)) / 1j / np.pi * weights , omega) - \
                  np.dot(np.imag(dtaudt_t * np.conjugate((tau_t - z))) / np.conjugate(tau_t - z)**2 / 1j / np.pi * weights, np.conjugate(omega))

    #plt.imshow(sysMat)
    return u, omega, sysMat, sysVec, info

segments = np.linspace(0, 4, 17)
sharp_corners = np.isin(segments, np.array([0.,1.,2.,3.,4.])).astype(int)
gridObject = GaussLegGrid(segments, sharp_corners)
gridObject.refine_all_nply(n_all_refine)
gridObject.refine_corners_nply(n_corner_refine)

gridpts, weights = gridObject.get_grid_and_weights()
#np.abs(np.imag(np.sum(np.conjugate(f(gridpts)) * weights * dtaudt(gridpts))))
tau_t = tau(gridpts)
dtaudt_t = dtaudt(gridpts)
ddtauddt_t = ddtauddt(gridpts)

def mask_fun(X,Y):
    Z = X + 1j*Y

    # Outer boundary
    dtz = tau_t[:, None, None] - Z[None, :, :]
    discriminator = np.sum(weights[:, None, None] * np.imag(dtaudt_t[:,None,None] / dtz), axis=0) / (2 * np.pi)
    return (discriminator > 0.5)

u,omega, sysMat, sysVec, info = solve(gridpts, weights, tol=1e-15)
print("System solved")


def f_slow(Z, fun):
    # Memory conserving u.
    U = np.zeros_like(Z)
    for i in range(U.shape[1]):
        print(f"F Eval: {i}/{U.shape[1]}", end="\r")
        U[:, i] = fun(Z[:, i, None]).flatten()
    return U

# Prepare solution
n_grid = 500
#n_grid = 1000
x_mesh = np.linspace(-(1 + L), (1 + L), n_grid)
y_mesh = np.linspace(-(1 + L), (1 + L), n_grid)

X, Y = np.meshgrid(y_mesh, x_mesh)
Z = X + 1j * Y
#U = u(Z.flatten()[:,None]).reshape((n_grid, n_grid))
#Utru = F(Z.flatten()[:,None]).reshape((n_grid, n_grid))
U = f_slow(Z, u)

if F is not None:
    Utru = f_slow(Z, F)
else:
    Utru = U

Uer = np.log10(np.abs(U-Utru))


#U_mask = Utru_mask

Nb = 2000 # Resolution of boundary
tb = np.linspace(0, 4, Nb)
zb = tau(tb)
fb = f(tb)

# 2D plot
# Plot domain with boundary conditions
# Some plots
# plt.plot(gridpts, np.real(omega))
# plt.plot(gridpts, np.imag(omega))
# plt.plot(gridpts, np.real(f(gridpts)))
# plt.plot(gridpts, np.imag(f(gridpts)))
taus = tau(gridpts)
dtaus = dtaudt(gridpts)

# Get mask
mask = mask_fun(X, Y)
U_mask = np.where(mask, U, np.nan + 1j*np.nan)
Utru_mask = np.where(mask, Utru, np.nan + 1j*np.nan)
Uer_mask = np.where(mask, Uer, np.nan)
# Boundary values
#plt.scatter(np.real(taus), np.imag(taus), c=f(gridpts))
#plt.quiver(np.real(taus), np.imag(taus), -np.imag(dtaus)/np.abs(dtaus), np.real(dtaus)/np.abs(dtaus))
#plt.scatter(np.real(taus)-0.03*np.imag(dtaus), np.imag(taus)+0.03*np.real(dtaus))
#plt.colorbar()
cmap = cm.get_cmap("coolwarm")
fig = plt.figure()
plt.plot(np.real(zb), np.imag(zb), 'black', linewidth=2)
plt.pcolormesh(X, Y, np.abs(U_mask), cmap=cmap)#, vmax=1, vmin=0)
plt.colorbar()#location="bottom")
#plt.streamplot(X, Y, np.real(U_mask), np.imag(U_mask), linewidth=1, color="white", density=2)#, color=np.log(np.abs(U_mask)), cmap="inferno")
plt.streamplot(X, Y, np.real(U), np.imag(U), linewidth=1, color="white", density=3)#, color=np.log(np.abs(U_mask)), cmap="inferno")
idx = list(range(0,len(zb), 50))
#plt.quiver(np.real(zb[idx]), np.imag(zb[idx]), np.real(fb[idx]), np.imag(fb[idx]), scale=10)
plt.xlim([-(1+0.01), (1+0.01)])
plt.ylim([-(1+0.01), (1+0.01)])
remove_axis(plt.gca())
plt.tight_layout()

#fig.savefig("boundary_integrals/figures/curve_flow.png", bbox_inches="tight", dpi=200)


# Plot Velocity field
# # magma, cividis, gnuplot, coolwarm,inferno,seismic,summer,turbo
cmap = cm.get_cmap('gist_yarg_r', 6)
#for velfield in [np.real(U_mask), np.imag(U_mask), np.real(Utru_mask), np.imag(Utru_mask), Uer_mask]:
#for velfield in [np.abs(np.real(Utru_mask-U_mask)), np.abs(np.imag(U_mask-Utru_mask))]:
for velfield in []:
    plt.figure(figsize=(8,7))
    plt.pcolormesh(X, Y, velfield, shading='nearest', label="Solution", cmap=cmap, vmax=0, vmin=-16)
    #plt.pcolormesh(X, Y, np.log10(np.abs(U-f_slow(Z,F))), shading='nearest', label="Solution", vmax=0, vmin=-16, cmap=cmap)
    plt.plot(np.real(zb), np.imag(zb), 'black', label="Boundary", linewidth=4)
    #plt.scatter(x_,y_,c=f_, label="Basis functions")
    #plt.xlim([-(1+L+0.01), (1+L+0.01)])
    #plt.ylim([-(1+L+0.01), (1+L+0.01)])
    plt.xlim([-(1 + 0.01), (1 + 0.01)])
    plt.ylim([-(1 + 0.01), (1 + 0.01)])
    remove_axis(plt.gca())
    plt.colorbar()#location="bottom")
    plt.tight_layout()

