import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *

# BOUNDARY 1
L = 0.0
M = 7

x = lambda t: np.cos(t)*(1 + L*np.sin(M*t))
dxdt = lambda t: L*M*np.cos(t)*np.cos(M*t) - np.sin(t)*(1 + L*np.sin(M*t))
ddxdt2 = lambda t: -np.cos(t)*(L*(M**2 +1)*np.sin(M*t) + 1) - 2*L*M*np.sin(t)*np.cos(M*t)

y = lambda t: np.sin(t)*(1 + L*np.sin(M*t))
dydt = lambda t: L*M*np.sin(t)*np.cos(M*t) + np.cos(t)*(1 + L*np.sin(M*t))
ddydt2 = lambda t: -np.sin(t)*(L*(M**2 + 1)*np.sin(M*t)+1) + 2*L*M*np.cos(t)*np.cos(M*t)

f = lambda t: (1.-np.sin(1*t)) #np.ones_like(t)#

# BOUNDARY 2
c = 0.5

x2 = lambda t: np.cos(t) * c
dxdt2 = lambda t: -np.sin(t) * c
ddxdt22 = lambda t: -np.cos(t) * c

y2 = lambda t: np.sin(t) * c
dydt2 = lambda t: np.cos(t) * c
ddydt22 = lambda t: -np.sin(t) * c

f2 = lambda t: -c*np.sin(1*t)*0.2 + 2.0

# BOUNDARY 1 pts
n_pts1 = 200
t1_ = np.linspace(0, 2*np.pi, n_pts1+1)
dt1_ = t1_[1:] - t1_[:-1]
t1_ = t1_[:-1]
x1_ = x(t1_)
y1_ = y(t1_)
f1_ = f(t1_)
dxdt1_ = dxdt(t1_)
dydt1_ = dydt(t1_)
ddxdt21_ = ddxdt2(t1_)
ddydt21_ = ddydt2(t1_)
sign1 = np.ones_like(t1_)

# BOUNDARY 2 pts
n_pts2 = 200
t2_ = np.linspace(0, 2*np.pi, n_pts2+1)
dt2_ = t2_[1:] - t2_[:-1]
t2_ = t2_[:-1]
x2_ = x2(t2_)
y2_ = y2(t2_)
f2_ = f2(t2_)
dxdt2_ = dxdt2(t2_)
dydt2_ = dydt2(t2_)
ddxdt22_ = ddxdt22(t2_)
ddydt22_ = ddydt22(t2_)
sign2 = -np.ones_like(t2_)

# Concating
n_pts = n_pts1 + n_pts2
t_ = np.hstack([t1_, t2_])
dt_ = np.hstack([dt1_, dt2_])
x_ = np.hstack([x1_, x2_])
y_ = np.hstack([y1_, y2_])
f_ = np.hstack([f1_, f2_])
dxdt_ = np.hstack([dxdt1_, dxdt2_])
dydt_ = np.hstack([dydt1_, dydt2_])
ddxdt2_ = np.hstack([ddxdt21_, ddxdt22_])
ddydt2_ = np.hstack([ddydt21_, ddydt22_])
sign = np.hstack([sign1, sign2])

K = np.zeros((n_pts, n_pts))

for n in range(n_pts):
    for m in range(n_pts):
        if m != n:
            K[n, m] = -((x_[n] - x_[m]) * dydt_[m] - (y_[n] - y_[m]) * dxdt_[m])\
                      /((x_[n] - x_[m]) ** 2 + (y_[n] - y_[m]) ** 2) * dt_[m] / np.pi
        else:
            K[n, m] = -(ddxdt2_[m] * dydt_[m] - ddydt2_[m] * dxdt_[m]) / (dxdt_[m] ** 2 + dydt_[m] ** 2) * dt_[m] / np.pi


mu = np.linalg.solve(np.diag(sign) + K, f_)
eigvals, eigvecs = np.linalg.eig(np.diag(sign) + K)
magn_eigs = (np.real(eigvals)**2 + np.imag(eigvals)**2)**0.5
magn_eigs.max() / magn_eigs.min()

plt.scatter(np.real(eigvals), np.imag(eigvals))

# U function
u = lambda X, Y: np.dot(-((X - x_)*dydt_ - (Y - y_)*dxdt_)/((X - x_)**2 + (Y - y_)**2)*dt_, mu) / np.pi

# PLOT SOLUTION
# Boundary points
N = 1000
t = np.linspace(0, 2*np.pi, N)
xb = x(t)
yb = y(t)
fb = f(t)
xb2 = x2(t)
yb2 = y2(t)
fb2 = f2(t)

# Prepare solution
n_grid = 400
x_grid = np.linspace(-(1+L), (1+L), n_grid)
y_grid = np.linspace(-(1+L), (1+L), n_grid)

X, Y = np.meshgrid(y_grid, x_grid)
U = u(X.flatten()[:,None], Y.flatten()[:,None]).reshape((n_grid, n_grid))

# Get mask
T = np.arctan2(Y.flatten(), X.flatten()).reshape((n_grid, n_grid))
mask = (X**2 + Y**2) <= (1 + L*np.sin(M*T) -0.03)**2
mask2 = (X**2 + Y**2) >= (c + 0.03)**2
mask = mask * mask2
U_mask = np.where(mask, U, np.nan)

# 2D plot
# Plot domain with boundary conditions
plt.pcolormesh(X, Y, U_mask, shading='nearest', label="Solution")
plt.plot(xb*0.98, yb*0.98, 'black', label="Boundary", linewidth=5)
plt.plot(xb2*1.03, yb2*1.03, 'black', label="Boundary", linewidth=5)
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
# Shadow
ax.plot(xb, yb, min(fb)*np.ones_like(xb), 'black', alpha=0.5)
ax.plot(xb2, yb2, min(fb)*np.ones_like(xb2), 'black', alpha=0.5)
ax.plot_surface(X, Y, U_mask*0+min(fb), color='black', alpha=0.5)
# Drum
ax.plot(xb, yb, fb, 'black')
ax.plot(xb2, yb2, fb2, 'black')
ax.plot_surface(X, Y, U_mask, facecolors=col, rcount=100, ccount=100)
remove_axis_3d(ax)
ax.axis('off')
ax.view_init(elev=10, azim=2)
