## Write in command window:
# set DDEBACKEND=pytorch ## Not stored beyond current session
# import os > os.environ > os.getenv('DDEBACKEND')

import deepxde as dde
from deepxde import config
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from deepXDE.geometry import *

dde.backend
#geom = dde.geometry.Rectangle([0,-1], [1, 1])
bdry_func = lambda x: np.sin(3*x*np.pi*2)/10
geom = MicroProblem(boundary_function=bdry_func, x_lims=[0, 1], bbox_y_lims=[-1, 1])

def pde(x, y):
    """affine operator A for stokes equations such that Au = 0"""
    u, v, p = y[:, 0:1], y[:, 1:2], y[:,2:]

    p_x = dde.grad.jacobian(p, x, i=0, j=0)
    p_y = dde.grad.jacobian(p, x, i=0, j=1)

    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)

    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)

    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, x, i=1, j=1)

    momentum_x = -p_x + u_xx + u_yy
    momentum_y = -p_y + v_xx + v_yy
    continuity = u_x + v_y

    return [momentum_x, momentum_y, continuity]


def boundary(_, on_boundary):
    return on_boundary


def sol(x):
    x0 = np.array([1.5, 1.5])[None,:] # Outside domain to avoid blow-up
    g = [1, 0.5]
    x_hat = x - x0
    return stokeslet_2d(x_hat, g)/8/np.pi


def stokeslet_2d(x, g):
    s = np.zeros_like(x)
    r = np.linalg.norm(x, axis=1)
    g_dot_x_over_r2 = (g[0] * x[:, 0] + g[1] * x[:, 1]) / r ** 2
    lnr = np.log(r)
    s[:, 0] = -g[0] * lnr + x[:, 0] * g_dot_x_over_r2
    s[:, 1] = -g[1] * lnr + x[:, 1] * g_dot_x_over_r2
    return s

def stokeslet_3d(x, g):
    s = np.zeros_like(x)
    r = np.linalg.norm(x, axis=1)
    g_dot_x_over_r3 = (g[0] * x[:, 0] + g[1] * x[:, 1] + g[2] * x[:, 2])/r ** 3
    s[:, 0] = g[0] / r + x[:, 0] * g_dot_x_over_r3
    s[:, 1] = g[1] / r + x[:, 1] * g_dot_x_over_r3
    s[:, 2] = g[2] / r + x[:, 2] * g_dot_x_over_r3
    return s

# Boundary Data
bc_u = dde.DirichletBC(geom, lambda x: sol(x)[:, 0:1], boundary, component=0)
bc_v = dde.DirichletBC(geom, lambda x: sol(x)[:, 1:], boundary, component=1)
data = dde.data.PDE(geom, pde, [bc_u, bc_v], num_domain=1000, num_boundary=1000, num_test=1500)

# Network and Model # relu -> loss 1.0 ,  tanh?
net = dde.maps.FNN([2]+[500]*3+[3], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Train
model.compile("adam", lr=0.001)
#loss, trainstate = model.train(epochs=5000)
model.train(epochs=5000)

from deepxde import optimizers
optimizers.set_LBFGS_options(maxcor=100, ftol=0, gtol=0,maxiter=15000, maxfun=None, maxls=50)
model.compile("L-BFGS")
model.train()

#x = geom.uniform_points(3000, True)

N = 20
eps = 0.01
x = np.linspace(0, 1, N)[None, :]
y = np.linspace(-0.1, 1, N)[:, None]

# Convert to mesh matrices
z = x + 1j*y
X = np.real(z).flatten()[:, None]
Y = np.imag(z).flatten()[:, None]
xy = np.hstack([X,Y])


inside = geom.inside(xy)[:,None]
inside = np.hstack([inside,]*3)

# Approximate solution
y = model.predict(xy)             # perator=pde_laplace)
y = np.where(inside, y, np.nan)  #

# True solution
ysol = sol(xy)
ysol = np.where(inside[:,:-1], ysol, np.nan)



fig = plt.figure() #figsize=(500,100))
for i,title in zip(range(2), ["u", "v"]): # Loop through vel components
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ax.scatter(xy[:, 0], xy[:, 1], y[:, i])
    ax.scatter(xy[:, 0], xy[:, 1], ysol[:, i])
    ax.set_title(title)

#fig = plt.figure()  # figsize=(500,100))
for i, title in zip(range(2), ["u error", "v error"]):  # Loop through vel components
    ax = fig.add_subplot(2, 2, i + 3, projection='3d')
    ax.scatter(xy[:, 0], xy[:, 1], y[:, i]-ysol[:, i])
    ax.set_title(title)


plt.figure()
#plt.pcolormesh(np.real(z), np.imag(z), np.linalg.norm(y, axis=1).reshape(N,N), "PINN pressure")
plt.pcolormesh(np.real(z), np.imag(z), y[:, 2].reshape(N,N), label="PINN pressure")
plt.streamplot(np.real(z), np.imag(z), y[:, 0].reshape(N,N), y[:, 1].reshape(N,N), color='white')#, label="PINN")
plt.streamplot(np.real(z), np.imag(z), ysol[:, 0].reshape(N,N), ysol[:, 1].reshape(N,N), color='red', density=0.3)#, label="True")
plt.legend()
# du_xx + du_yy = 1 --> one solution u(x,y) = x(x-1)/2 + y(y-1)/2

