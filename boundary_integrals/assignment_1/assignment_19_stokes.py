import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

# Grid refinement
n_corner_refine = 1
n_all_refine = 1
# Plotting options
L = 0.5
eps = 0.05
plot_grid = False
# Settings for boundary + conditions
setting = 3
# 1 - pipe flow, set inlet and outlet radii
# 2 - stirring boundary condition, set u_tangential
# 3 - Flow across surface with stream velocity at top boundary. Boundary is box with rough boundary
# 4 - load cos-sine series from folder
in_rad = 1.0      # Radius at inlet
out_rad = 0.5          # Radius at outlet
vol_in = 1.0        # Volume flowing through pipe per time unit. For setting 1 only
u_tangental = 1.0   # Max stirring velocity, for setting 2,3 only
roughness = 21      # Roughness of bottom boundary in setting 3
amplitude = 0.02     # Amplitude of bottom boundary in setting 3

# Force equal radii if chose fre flow
if setting == 3:
    in_rad = 1
    out_rad = 1

if setting in [1,2]:
    def tau(t):
        ones_t = np.ones_like(t)
        left_wall = -ones_t  + 1j * in_rad * (1 - t * 2)
        bot_wall = ((t - 1) * 2 - 1) - 1j*(in_rad * (1-(t-1)) + out_rad * (t-1))
        right_wall = ones_t + 1j * out_rad * ((t - 2) * 2 - 1)
        top_wall = ((4 - t)*2 - 1) + 1j*(out_rad * (1-(t-3)) + in_rad*(t-3))
        tau_t = np.where(t <= 2, np.where(t <= 1, left_wall, bot_wall), np.where(t <= 3, right_wall, top_wall))
        return tau_t

    def dtaudt(t):
        ones_t = np.ones_like(t)
        left_wall = -1j * in_rad * 2 * ones_t
        bot_wall = 2*ones_t- 1j * (out_rad - in_rad) * ones_t
        right_wall = 1j * out_rad * 2*ones_t
        top_wall = -2 * ones_t + 1j * (in_rad - out_rad) * ones_t
        dtaudt_t = np.where(t <= 2, np.where(t <= 1, left_wall, bot_wall), np.where(t <= 3, right_wall, top_wall))
        return dtaudt_t

    def ddtauddt(t):
        return np.zeros_like(t)

    def mask_fun(X, Y):
        y_x = (out_rad - in_rad) * X / 2 + (out_rad + in_rad) / 2
        mask = (X <= 1 - eps) * (-1 + eps <= X) * (Y <= y_x - eps) * (-y_x + eps <= Y)
        # mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
        return mask

elif setting==3:
    def tau(t):
        ones_t = np.ones_like(t)
        left_wall = -ones_t + 1j * in_rad * (1 - t * 2)
        bot_wall = ((t - 1) * 2 - 1) + (-1 - amplitude * np.sin(roughness * np.pi*(t-1)))*1j
        right_wall = ones_t + 1j * out_rad * ((t - 2) * 2 - 1)
        top_wall = ((4 - t) * 2 - 1) + 1j
        tau_t = np.where(t <= 2, np.where(t <= 1, left_wall, bot_wall), np.where(t <= 3, right_wall, top_wall))
        return tau_t


    def dtaudt(t):
        ones_t = np.ones_like(t)
        left_wall = -1j * in_rad * 2 * ones_t
        bot_wall = 2 - 1j * (np.pi * roughness) * amplitude * np.cos(roughness * np.pi*(t-1))
        right_wall = 1j * out_rad * 2 * ones_t
        top_wall = -2 * ones_t
        dtaudt_t = np.where(t <= 2, np.where(t <= 1, left_wall, bot_wall), np.where(t <= 3, right_wall, top_wall))
        return dtaudt_t


    def ddtauddt(t):
        zero_t = np.zeros_like(t)
        dd_bot = 1j * (np.pi*roughness)**2 * amplitude * np.sin(roughness * np.pi * (t-1))
        ddtauddt_t = np.where(t <= 2, np.where(t <= 1, zero_t, dd_bot), np.where(t <= 3, zero_t, zero_t))
        return ddtauddt_t


    def mask_fun(X, Y):
        y_x = -1-amplitude * np.sin(roughness * np.pi * (X + 1)/2)
        mask = (X <= 1 - eps) * (-1 + eps <= X) * (Y <= 1 - eps) * (y_x + eps <= Y)
        # mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
        return mask
else:
    func_dict = np.load("boundary_integrals/saved_cossin_bdries/curve.npy", allow_pickle=True).flatten()[0]
    freqs = func_dict["weights"] * np.pi/2
    sq_freqs = freqs**2
    coefs = func_dict["coefs"]
    coefs = (coefs[:,0] + 1j*coefs[:,1])[:,None]
    n_freqs = len(freqs)


    tau = lambda t: (np.hstack([np.cos(t[:, None] @ freqs.T), np.sin(t[:, None] @ freqs[1:].T)]) @ coefs).flatten()
    dtaudt = lambda t: (np.hstack([-np.sin(t[:,None] @ freqs.T)*freqs.T, np.cos(t[:,None] @ freqs[1:].T)*freqs[1:].T]) @ coefs).flatten()
    ddtauddt = lambda t: (np.hstack([-np.cos(t[:,None] @ freqs.T)*sq_freqs.T, np.cos(t[:, None] @ freqs[1:].T)*sq_freqs[1:].T]) @ coefs).flatten()

    t = np.linspace(0, 4, 200)
    bdry_pts = tau(t)
    normal = dtaudt(t) / 1j
    normal = normal / np.abs(normal)
    inner_bdry = (bdry_pts - eps * normal)
    curvature = np.abs(ddtauddt(t))

    plt.scatter(np.real(bdry_pts), np.imag(bdry_pts))
    plt.scatter(np.real(inner_bdry), np.imag(inner_bdry))


    discriminator = lambda z: np.sum(curvature[:,None,None] * (1 / np.abs(inner_bdry[:,None,None] - z) - 1 / np.abs(bdry_pts[:,None,None] - z)), axis=0)


    def mask_fun(X, Y): # Just take square
        Z = X + 1j * Y
        mask = discriminator(Z[None,:]) > 0
        # mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
        return mask


# Boundary Conditions (Dirichlet)s
if setting == 1:
    def f(t):
        align=True
        y = np.imag(tau(t))

        u_inlet = (in_rad - y) * (in_rad + y) * 6 * vol_in / (2*in_rad)**3
        v_inlet = u_inlet * (out_rad - in_rad) / 2 * y / in_rad
        if align:
            inlet = u_inlet + 1j*v_inlet
        else:
            inlet = u_inlet

        u_outlet = (out_rad - y) * (out_rad + y) * 6 * vol_in / (2*out_rad)**3
        v_outlet = u_outlet * (out_rad - in_rad) / 2 * y / out_rad
        if align:
            outlet = u_outlet + 1j*v_outlet
        else:
            outlet = u_outlet

        noslip = np.zeros_like(t)
        f_t = np.where(t <= 2, np.where(t <= 1, inlet, noslip), np.where(t <= 3, outlet, noslip))
        return f_t


    def F(z):
        x = np.real(z)
        y = np.imag(z)

        rad = (out_rad - in_rad) / 2 * x + (in_rad + out_rad) / 2
        u = (rad - y) * (rad + y) * 6 * vol_in / (2 * rad) ** 3
        v = u * (out_rad - in_rad) / 2 * y / rad
        uv = u + 1j * v
        return uv

elif setting == 2:
    def f(t):
        y = np.imag(tau(t))
        tangental = (y - in_rad) * (y + in_rad) / in_rad**2 * u_tangental * (-1j)
        #tangental = np.ones_like(t) * u_tangental
        noslip = np.zeros_like(t)
        f_t = np.where(t <= 2, np.where(t <= 1, tangental, noslip), np.where(t <= 3, noslip, noslip))
        return f_t

    # No access to true solution
    F = None
elif setting==3:
    # velocity decay towards boundary
    #decay = lambda s: np.log10(1 + 5*s)
    decay = lambda s: s
    def f(t):
        y = np.imag(tau(t))
        tangental = u_tangental * decay(y+1)/decay(2)
        #tangental = np.ones_like(t) * u_tangental
        noslip = np.zeros_like(t)
        f_t = np.where(t <= 2, np.where(t <= 1, tangental, noslip), np.where(t <= 3, tangental, tangental))
        return f_t


    def F(z):
        # Approximate as inlet
        y = np.imag(z)
        u = u_tangental * decay(y+1)/decay(2)
        v = np.zeros_like(u)
        uv = u + 1j * v
        return uv
else:

    f = lambda t: dtaudt(t) * 1.0
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

    for n in range(n_pts):
        for m in range(n_pts):
            if m != n:
                K[n, m] = np.imag(dtaudt_t[m] / (tau_t[m] - tau_t[n]) / np.pi * weights[m])
                Kconj[n, m] = -np.imag(dtaudt_t[m] * np.conjugate(tau_t[m] - tau_t[n]))/\
                              np.conjugate(tau_t[m] - tau_t[n])**2 / np.pi * weights[m]
            else:
                K[n, m] = np.imag(ddtauddt_t[m] / (2 * dtaudt_t[m])) /np.pi * weights[m]
                Kconj[n, m] = np.imag(ddtauddt_t[m] * dtaudt_t_conj[m]) / (2 * dtaudt_t_conj[m]**2) /np.pi * weights[m]

    A = np.eye(n_pts) + K
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

# Special mask based on rbfs, have to create down here since the grid needs to be defined
if setting == 4:
    def mask_fun(X, Y):
        #Just take square
        Z = X + 1j * Y
        p = 3
        dtz = taus[:,None,None] - Z[None,:,:]
        discriminator = np.sum(weights[:, None, None] * np.abs(dtaus)[:,None,None] * np.imag(dtz * np.conjugate(dtaus[:, None, None])) / np.abs(dtz) ** p, axis=0)
        mask = (discriminator < 0) * (discriminator > -(3000)**(4/p))
        # mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
        return mask

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
plt.pcolormesh(X, Y, np.abs(U_mask), cmap=cmap, vmax=1, vmin=0)
plt.colorbar()#location="bottom")
#plt.streamplot(X, Y, np.real(U_mask), np.imag(U_mask), linewidth=1, color="white", density=2)#, color=np.log(np.abs(U_mask)), cmap="inferno")
plt.streamplot(X, Y, np.real(U), np.imag(U), linewidth=1, color="white", density=3)#, color=np.log(np.abs(U_mask)), cmap="inferno")
idx = list(range(0,len(zb), 50))
#plt.quiver(np.real(zb[idx]), np.imag(zb[idx]), np.real(fb[idx]), np.imag(fb[idx]), scale=10)
plt.xlim([-(1+0.01), (1+0.01)])
plt.ylim([-(1+0.01), (1+0.01)])
remove_axis(plt.gca())
plt.tight_layout()

fig.savefig("boundary_integrals/figures/curve_flow.png", bbox_inches="tight", dpi=200)


# Plot Velocity field
# # magma, cividis, gnuplot, coolwarm,inferno,seismic,summer,turbo
cmap = cm.get_cmap('gist_yarg_r', 6)
#for velfield in [np.real(U_mask), np.imag(U_mask), np.real(Utru_mask), np.imag(Utru_mask), Uer_mask]:
#for velfield in [np.abs(np.real(Utru_mask-U_mask)), np.abs(np.imag(U_mask-Utru_mask))]:
for velfield in [np.where(Uer_mask<-16, -16, Uer_mask)]:
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

plt.figure(1).savefig("boundary_integrals/figures/stokes_pipe_error.png", dpi=200)

plt.figure()
x = np.linspace(-1, 1, 100)
flows = np.zeros_like(x)
flows2 = np.zeros_like(x)
flowGrid = GaussLegGrid(np.array([-1, 1]), np.array([1, 1]))
flowpts, flowweights = flowGrid.get_grid_and_weights()
for i, x_p in enumerate(x):
    ymax = (out_rad - in_rad)*x_p/2 + (out_rad + in_rad)/2
    z = x_p + 1j * flowpts * ymax
    flows[i] = np.sum(np.real(u(z[:,None]))*flowweights * ymax)
    if F is not None:
        flows2[i] = np.sum(np.real(F(z))*flowweights * ymax)
    else:
        flows2[i] = flows[i]

plt.plot(flows)
plt.plot(flows2)
plt.plot(vol_in * np.ones_like(flows), '--')
# xy = tau(gridpts)
