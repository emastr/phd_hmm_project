import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *

eps = 0.02
epsbound = 0.00

L = 0.3
M = 5

tau = lambda t: (1 + L * np.cos(M * t)) * np.exp(1j * t)
dtaudt = lambda t: (- L * M * np.sin(M * t)\
                    + 1j * (1 + L * np.cos(M * t))) * np.exp(1j * t)
ddtaudt2 = lambda t: (-L * M * M * np.cos(M * t) \
                      - 2 * L * M * np.sin(M * t) * 1j \
                      - 1 - L * np.cos(M * t)) * np.exp(1j * t)
zp = 1.3 + 1j * 1.3
F = lambda z: np.imag(z ** 2 / (z - zp))
f = lambda t: F(tau(t))

n_pts = 1000
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

# PLOT SOLUTION
# Boundary points
N = 1000
t = np.linspace(0, 2*np.pi, N)
zb = tau(t)
fb = f(t)

# Prepare solution
n_grid = 200
x_grid = np.linspace(-(1+L), (1+L), n_grid)
y_grid = np.linspace(-(1+L), (1+L), n_grid)

X, Y = np.meshgrid(y_grid, x_grid)
Z = X + 1j * Y
U = u(Z.flatten()[:,None]).reshape((n_grid, n_grid))
Utru = F(Z.flatten()[:,None]).reshape((n_grid, n_grid))
U = np.log10(np.abs(U-Utru))
# Get mask
T = np.arctan2(Y.flatten(), X.flatten()).reshape((n_grid, n_grid))
mask = (X**2 + Y**2) <= (1 + L*np.cos(M*T) - eps)**2
U_mask = np.where(mask, U, np.nan)

# 2D plot
# Plot domain with boundary conditions
plt.figure(figsize=(7.5,7))
plt.pcolormesh(X, Y, U_mask, shading='nearest', label="Solution")
plt.plot(np.real(zb)*(1-epsbound), np.imag(zb)*(1-epsbound), 'black', label="Boundary", linewidth=2)
#plt.scatter(x_,y_,c=f_, label="Basis functions")
plt.xlim([-(1+L+0.1), (1+L+0.1)])
plt.ylim([-(1+L+0.1), (1+L+0.1)])
remove_axis(plt.gca())
plt.colorbar()

# cmap
vir = cm.get_cmap("viridis",100)
col = vir(mask*(U-fb.min())/(fb.max()-fb.min()))
#
# # Plot in 3d
# fig = plt.figure(figsize=(9,7))
# ax = fig.add_subplot(projection='3d')
# ax.set_box_aspect((6,6,1))
# # Shadow
# ax.plot(np.real(zb), np.imag(zb), min(fb)*np.ones_like(zb), 'black', alpha=0.5)
# ax.plot_surface(X, Y, U_mask*0+min(fb), color='black', alpha=0.5)
# # Drum
# ax.plot(np.real(zb), np.imag(zb), fb, 'black')
# ax.plot_surface(X, Y, U_mask, rcount=100, ccount=100, facecolors=col)
# remove_axis_3d(ax)
# ax.axis('off')
# ax.view_init(azim=94, elev=20)
# ax.set_xlim([-(1+L*0), (1+L*0)])
# ax.set_ylim([-(1+L*0), (1+L*0)])
# plt.tight_layout()


# # Plot radial error 3d
# m = 2.04
# t_fix = m/M * 2 * np.pi
# N = 500
#
# z_fix = tau(t_fix)
# radii = np.linspace(0, 1, N)
# z = z_fix * radii
#
# u_apx = u(z[:, None])
# u_tru = F(z)
#
# fig = plt.figure(figsize=(9,7))
# ax = fig.add_subplot(projection='3d')
# ax.set_box_aspect((2,2,1))
# # Shadow
# ax.plot(np.real(zb), np.imag(zb), min(fb)*np.ones_like(zb), 'black', alpha=0.3)
# ax.plot_surface(X, Y, U_mask*0+min(fb), rcount=200, ccount=200, color='black', alpha=0.2)
# # Drum
# #ax.plot(np.real(zb), np.imag(zb), fb, 'black')
# ax.plot(np.real(z), np.imag(z), np.zeros_like(z)+min(fb), color=[0.9,0.9,0.9], linewidth=3)
# ax.plot(np.real(z), np.imag(z), u_tru+min(fb), 'g', linewidth=3, label='True')
# ax.plot(np.real(z), np.imag(z), u_apx+min(fb), 'r--', linewidth=3, label='Approximation')
# ax.plot_surface(np.vstack([np.real(z),np.real(z)]),
#                 np.vstack([np.imag(z),np.imag(z)]),
#                 np.vstack([u_tru+min(fb),np.zeros_like(u_tru)+min(fb)]),
#                 color=[0.8,0.8,0.8], alpha=0.5)
# remove_axis_3d(ax)
# ax.axis('off')
# ax.view_init(azim=37, elev=20)
# ax.set_xlim([-(1+L*0), (1+L*0)])
# ax.set_ylim([-(1+L*0), (1+L*0)])
# plt.legend(bbox_to_anchor=(0.25, 0, 0.5, 0), loc="lower left", mode="expand", ncol=2, edgecolor='none')
# plt.tight_layout()
#
# fig.savefig("boundary_integrals/figures/displacement_slice.pdf", bbox_inches='tight')