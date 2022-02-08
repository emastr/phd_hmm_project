import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

R_trunk = 1
R_knot = 0.2
dist_knot = 0.7
knot_strength = -0.13#-0.001

# Parametrisation of boundary
ma, na = 0.01, 0.03
m,n=10,6
tau_trunk = lambda t: R_trunk * np.exp(1j*t) * (1 + ma*np.sin(m * t) + na*np.sin(n * t))
tau_trunk_der = lambda t: 1j * R_trunk * np.exp(1j*t) * (1 + ma*(np.sin(m * t) + np.cos(m*t)*m/1j) + na*(np.sin(n * t) + np.cos(n*t)*n/1j))

# Parametrisation of knot
tau_knot = lambda t: dist_knot + R_knot * np.exp(1j*t)
tau_knot_der = lambda t: 1j * R_knot * np.exp(1j*t)

# Grid segments for numerical integration
N_trunk = 10
t_trunk = np.linspace(0, np.pi*2,N_trunk)
grid_trunk = GaussLegGrid(t_trunk, np.zeros_like(t_trunk).astype(int))
gridpts_trunk, weights_trunk = grid_trunk.get_grid_and_weights()
tau_trunk_t = tau_trunk(gridpts_trunk)
tau_trunk_der_t = tau_trunk_der(gridpts_trunk)


# Grid for knot
N_knot = 10
t_knot = np.linspace(0, np.pi*2,N_knot)
grid_knot = GaussLegGrid(t_knot, np.zeros_like(t_knot).astype(int))
gridpts_knot, weights_knot = grid_knot.get_grid_and_weights()
tau_knot_t = tau_knot(gridpts_knot)
tau_knot_der_t = tau_trunk_der(gridpts_trunk)
density_knot = np.ones_like(gridpts_trunk)#-1+np.sin(gridpts_trunk/2)
def age_field(X, Y, tau_outer, tau_der_outer, weights_outer, tau_inner, tau_der_inner, weights_inner, density_inner):
    Z = X + 1j * Y
    p_outer = 1
    p_inner = 2
    # Outer boundary
    dtz = tau_outer[:, None, None] - Z[None, :, :]
    abs_dtau = np.abs(tau_der_outer)
    abs_dts = np.abs(dtz)
    discriminator_outer = np.sum(weights_outer[:, None, None] * abs_dtau[:,None,None] /abs_dts**p_outer, axis=0) / (2 * np.pi)

    # Inner boundary (holes)
    # dtz = tau_inner[:, None, None] - Z[None, :, :]
    # abs_dts = np.abs(dtz)
    # abs_dtau = np.abs(tau_der_inner)
    # discriminator_inner = np.sum(weights_inner[:, None, None] * abs_dtau[:,None,None] /abs_dts**p_inner * density_inner[:,None,None], axis=0) / (2 * np.pi)

    # Inner boundary (just one basis function)
    dtz = dist_knot*1 + 0*1j - Z
    abs_dtz = np.abs(dtz)
    line = -p_inner/R_knot ** (p_inner + 1) * abs_dtz + (1 + p_inner) / R_knot ** p_inner
    #discriminator_inner = np.where(abs_dtz < R_knot, line, 1/abs_dtz**p_inner)
    discriminator_inner = np.exp(-(abs_dtz/R_knot)**p_inner)


    # Streak at some angle (happens sometimes)
    A = np.angle(Z)
    dtz = np.minimum(np.abs(A - 0.2*np.pi), np.abs(A - 2*np.pi - 0.2*np.pi))
    crevace = 0.1*np.exp(-dtz**2/(0.05/np.abs(Z))**2) * np.abs(Z) **2

    return discriminator_inner * knot_strength + discriminator_outer + crevace


def mask(X,Y, tau_outer, tau_der_outer, weights_outer, tau_inner, tau_der_inner, weights_inner):
    Z = X + 1j*Y

    # Outer boundary
    dtz = tau_outer[:, None, None] - Z[None, :, :]
    discriminator_outer = np.sum(weights_outer[:, None, None] * np.imag(tau_der_outer[:,None,None] / dtz), axis=0) / (2 * np.pi)

    # Inner boundary (holes)
    dtz = tau_inner[:, None, None] - Z[None, :, :]
    discriminator_inner = np.sum(weights_inner[:, None, None] * np.imag(tau_der_inner[:, None, None] / dtz), axis=0) / (
                2 * np.pi)
    return (discriminator_outer > 0.5) #* (discriminator_inner < 0.5)


L = 0.1
n_grid = 500
x_mesh = np.linspace(-(1 + L), (1 + L), n_grid)
y_mesh = np.linspace(-(1 + L), (1 + L), n_grid)

X, Y = np.meshgrid(y_mesh, x_mesh)
Z = X + 1j * Y
mask_img = mask(X, Y, tau_trunk_t, tau_trunk_der_t, weights_trunk, tau_knot_t, tau_knot_der_t, weights_knot)
age = age_field(X, Y, tau_trunk_t, tau_trunk_der_t, weights_trunk, tau_knot_t, tau_knot_der_t, weights_knot, density_knot)

def center_at_12(array):
    a_min = np.nanmin(array)
    a_max = np.nanmax(array)
    return 1 + (array - a_min)/(a_max - a_min)

def convexify(array, n):
    if n==0:
        return array
    else:
        return convexify(np.log2(center_at_12(array)), n-1)

sigmoid = lambda x: 1/(np.exp(-x)+1)
# Function that makes sure the rings are well spread out
def concavify(array, n):
    if n == 0:
        return center_at_12(array)-1.5
    else:
        t = center_at_12(array)-1.5
        return concavify(np.where(t<=0, -np.abs(t)**(1/10), np.abs(t)**(1/2)),n-1)
        #return #concavify(sigmoid(5*t), n-1)

age = np.where(mask_img == 1, age, np.nan)
#plt.imshow(age)
age_line = age[n_grid//2,0:n_grid//2]
#plt.plot(age[n_grid//2,:])
levels = []
for i in range(len(age_line)):
    if not np.isnan(age_line[i]):
        levels.append(age_line[i])
levels = np.sort(np.array([levels[i] for i in range(0,len(levels),10)]))
#plt.plot(age[n_grid//2,0:n_grid//2])
#age = np.clip(age, 0,2)
#plt.imshow(age)


cmap = plt.get_cmap('Greys', 10)
ageLevels = np.sum((age[None, :, :] < levels[:, None, None]), axis=0)
ageLevels = np.where(mask_img, ageLevels**0.2, np.nan)
plt.imshow(ageLevels, extent=(-(1+L), (1+L), -(1+L), (1+L)), cmap='copper')# 0.3
plt.contour(age[::-1,:], extent=(-(1+L), (1+L), -(1+L), (1+L)), levels=levels, colors='black')
#plt.plot(np.real(tau_trunk_t), np.imag(tau_trunk_t), 'white')
#plt.plot(np.real(tau_knot_t), np.imag(tau_knot_t), 'white')
ax = plt.gca()
remove_axis(ax)
plt.colorbar()
plt.show()