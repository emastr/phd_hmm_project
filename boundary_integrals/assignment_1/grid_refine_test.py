import numpy as np
from boundary_integrals.gaussleggrid import GaussLegGrid
import matplotlib.pyplot as plt

# Line
tau = lambda t: t # Parameter interval = [0, 1]

# Function to integrate
k = 1000
f0 = lambda t: np.sin(k * t) * np.exp(t)
F0 = lambda t: (np.sin(k*t) - k*np.cos(k*t))*np.exp(t) / (1+k**2)

f = lambda t: f0(2.001023 * t) + f0(1.3199 * t)
F = lambda t: F0(2.001023 * t)/2.001023 + F0(1.3199 * t) / 1.3199
# True value
true = F(1.) - F(0.)

nply_errs = []
# Approximate value
for n in range(15):
    grid = GaussLegGrid(np.array([0, 1]))
    grid.refine_all_nply(n)
    #plt.stem(grid.segments, grid.corners)

    gridpts, weights = grid.get_grid_and_weights()
    nply_errs.append(np.abs(GaussLegGrid.integrate_func(f, gridpts, weights) - true))

plt.semilogy(nply_errs)

#plt.scatter(gridpts, fun(gridpts))

