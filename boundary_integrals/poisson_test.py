import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *


L = 0.3
M = 7
eps = 0.05
epsbound = 0.04

x = lambda t: np.cos(t)*(1 + L*np.sin(M*t))
dxdt = lambda t: L*M*np.cos(t)*np.cos(M*t) - np.sin(t)*(1 + L*np.sin(M*t))
ddxdt2 = lambda t: -np.cos(t)*(L*(M**2 +1)*np.sin(M*t) + 1) - 2*L*M*np.sin(t)*np.cos(M*t)

y = lambda t: np.sin(t)*(1 + L*np.sin(M*t))
dydt = lambda t: L*M*np.sin(t)*np.cos(M*t) + np.cos(t)*(1 + L*np.sin(M*t))
ddydt2 = lambda t: -np.sin(t)*(L*(M**2 + 1)*np.sin(M*t)+1) + 2*L*M*np.cos(t)*np.cos(M*t)

f = lambda t: (1+np.sin(2*t))

n_pts = 400
t_ = np.linspace(0, 2*np.pi, n_pts+1)
dt_ = t_[1:] - t_[:-1]
t_ = t_[:-1]
x_ = x(t_)
y_ = y(t_)
f_ = f(t_)
dxdt_ = dxdt(t_)
dydt_ = dydt(t_)
ddxdt2_ = ddxdt2(t_)
ddydt2_ = ddydt2(t_)

K = np.zeros((n_pts, n_pts))

for n in range(n_pts):
    for m in range(n_pts):
        if m != n:
            K[n, m] = -((x_[n] - x_[m]) * dydt_[m] - (y_[n] - y_[m]) * dxdt_[m])\
                      /((x_[n] - x_[m]) ** 2 + (y_[n] - y_[m]) ** 2) * dt_[m] / np.pi
        else:
            K[n, m] = -(ddxdt2_[m] * dydt_[m] - ddydt2_[m] * dxdt_[m])\
                      /(dxdt_[m] ** 2 + dydt_[m] ** 2) * dt_[m] / np.pi


mu = np.linalg.solve(np.eye(n_pts) + K, f_)
#eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
#plt.scatter(np.real(eigvals), np.imag(eigvals))

# U function
u = lambda X, Y: np.dot(-((X - x_)*dydt_ - (Y - y_)*dxdt_)/((X - x_)**2 + (Y - y_)**2)*dt_, mu) / np.pi

# PLOT SOLUTION
# Boundary points
N = 1000
t = np.linspace(0, 2*np.pi, N)
xb = x(t)
yb = y(t)
fb = f(t)

# Prepare solution
n_grid = 200
x_grid = np.linspace(-(1+L), (1+L), n_grid)
y_grid = np.linspace(-(1+L), (1+L), n_grid)

X, Y = np.meshgrid(y_grid, x_grid)
U = u(X.flatten()[:,None], Y.flatten()[:,None]).reshape((n_grid, n_grid))

# Get mask
T = np.arctan2(Y.flatten(), X.flatten()).reshape((n_grid, n_grid))
mask = (X**2 + Y**2) <= (1 + L*np.sin(M*T) -eps)**2
U_mask = np.where(mask, U, np.nan)

# 2D plot
# Plot domain with boundary conditions
plt.figure(figsize=(7.5,7))
plt.pcolormesh(X, Y, U_mask, shading='nearest', label="Solution")
plt.plot(xb*(1-epsbound), yb*(1-epsbound), 'black', label="Boundary", linewidth=6)
#plt.scatter(x_,y_,c=f_, label="Basis functions")
plt.xlim([-(1+L+0.1), (1+L+0.1)])
plt.ylim([-(1+L+0.1), (1+L+0.1)])
remove_axis(plt.gca())

# cmap
vir = cm.get_cmap("viridis",100)
col = vir(mask*(U-fb.min())/(fb.max()-fb.min()))

# Plot in 3d
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((6,6,1))
# Shadow
ax.plot(xb, yb, min(fb)*np.ones_like(xb), 'black', alpha=0.5)
ax.plot_surface(X, Y, U_mask*0+min(fb), color='black', alpha=0.5)
# Drum
ax.plot(xb, yb, fb, 'black')
ax.plot_surface(X, Y, U_mask, rcount=100, ccount=100, facecolors=col)
remove_axis_3d(ax)
ax.axis('off')
ax.view_init(azim=94, elev=20)
ax.set_xlim([-(1+L*0), (1+L*0)])
ax.set_ylim([-(1+L*0), (1+L*0)])
plt.tight_layout()