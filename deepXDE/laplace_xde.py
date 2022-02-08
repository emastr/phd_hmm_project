## Write in command window:
# set DDEBACKEND=pytorch ## Not stored beyond current session
# import os > os.environ > os.getenv('DDEBACKEND')

import deepxde as dde
from deepxde import config
import torch
import matplotlib.pyplot as plt
import tensorflow as tf

geom = dde.geometry.Rectangle([0, -1], [1, 1])


def sol(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:]
    print(x.shape)
    return x1*(x1-1)/4 + x2*(x2-1)/4

#

def pde_laplace(x, y):
    """Linear operator such that pde_x(y) = 0"""
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return dy_xx + dy_yy - 1.


def boundary(_, on_boundary):
    return on_boundary


# Boundary Data
bc = dde.DirichletBC(geom, sol, boundary)
data = dde.data.PDE(geom, pde_laplace, bc, num_domain=1000, num_boundary=1000, num_test=1500)

# Network and Model # relu -> loss 1.0 ,  tanh?
net = dde.maps.FNN([2]+[100]*4+[1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Train
model.compile("adam", lr=0.001)
#loss, trainstate = model.train(epochs=50000)
model.train(epochs=3000)
from deepxde import optimizers
optimizers.set_LBFGS_options(maxcor=100, ftol=0, gtol=0,maxiter=15000, maxfun=None, maxls=50)
model.compile("L-BFGS")
losshistory, train_state = model.train()

x = geom.uniform_points(1000, True)
y = model.predict(x)#, operator=pde_laplace)
#y = model.predict(x)
ysol = sol(x)

#plt.figure()
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], ysol-y)


#plt.figure()
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], y)
ax.scatter(x[:, 0], x[:, 1], ysol)
#plt.show()

# du_xx + du_yy = 1 --> one solution u(x,y) = x(x-1)/2 + y(y-1)/2
dde.saveplot(losshistory, train_state, issave=False, isplot=True)


