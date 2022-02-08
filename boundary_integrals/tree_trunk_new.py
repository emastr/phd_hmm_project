import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

np.random.seed(0)

def sample_coefs(freqs):
    coefs = 0.1 * np.abs(np.random.randn(len(freqs), 1)) / (1+freqs)
    coefs[0,0] += 1
    return coefs

# Sample K cosines, K sines.
K = 10
fig = plt.figure(figsize=(8,2))
for i in range(4):
    freqs = np.round(np.abs(np.random.randn(2 * K, 1) * 10))
    freqs[0, 0] = 0  # First element is always the constant
    fourier_basis = lambda t: np.vstack([np.cos(freqs[0:K, :] * t[None, :]), np.sin(freqs[K:, :] * t[None, :])])
    fourier_basis_der = lambda t: freqs * np.vstack([-np.sin(freqs[0:K, :] * t), np.cos(freqs[K:, :] * t)])

    coefs = sample_coefs(freqs)
    tau = lambda t: np.dot(coefs.T, fourier_basis(t)).flatten() * np.exp(1j*t)
    tau_der = lambda t: (np.dot(coefs.T, fourier_basis_der(t)) + 1j*np.dot(coefs.T, fourier_basis(t))).flatten() * np.exp(1j*t)
    t = np.linspace(0, 2*np.pi, 100)
    z = tau(t).flatten()
    plt.subplot(1, 4, i+1)
    ax = plt.gca()
    remove_axis(ax)


    plt.plot(np.real(z), np.imag(z), 'black', linewidth=2)
    # L = 0.1
    # xmax, xmin = np.max(np.real(z)), np.min(np.real(z))
    # ymax, ymin = np.max(np.imag(z)), np.min(np.imag(z))
    # lim = (1 + L) * max([xmax, -xmin, ymax, -ymin])
    # n_grid = 500
    # plt.xlim([-lim, lim])
    # plt.ylim([-lim, lim])
    # ax.set_aspect('equal', adjustable='box')
    # fig.savefig("boundary_integrals/figures/logs.pdf", bbox_inches="tight")

    # Grid segments for numerical integration
    N = 10
    t = np.linspace(0, np.pi*2,N)
    grid = GaussLegGrid(t, np.zeros_like(t).astype(int))
    gridpts, weights = grid.get_grid_and_weights()
    tau_t = tau(gridpts)
    tau_der_t = tau_der(gridpts)

    def mask(X,Y):
        Z = X + 1j*Y

        # Outer boundary
        dtz = tau_t[:, None, None] - Z[None, :, :]
        discriminator = np.sum(weights[:, None, None] * np.imag(tau_der_t[:,None,None] / dtz), axis=0) / (2 * np.pi)

        return (discriminator > 0.5) #* (discriminator_inner < 0.5)



    L = 0.1
    xmax, xmin = np.max(np.real(tau_t)), np.min(np.real(tau_t))
    ymax, ymin = np.max(np.imag(tau_t)), np.min(np.imag(tau_t))
    lim = (1+L)*max([xmax, -xmin, ymax, -ymin])
    n_grid = 500
    x_mesh = np.linspace(-lim, lim, n_grid)
    y_mesh = np.linspace(-lim, lim, n_grid)

    X, Y = np.meshgrid(y_mesh, x_mesh)
    Z = X + 1j * Y
    mask_img = mask(X, Y)
    # plt.imshow(mask_img[::-1, :], cmap='Greys')
    # fig.savefig("boundary_integrals/figures/log_masks.pdf", bbox_inches="tight")

    def age(X, Y):
        Z = X + 1j * Y
        p_outer = 1
        p_inner = 2
        # Outer boundary
        dtz = tau_t[:, None, None] - Z[None, :, :]
        abs_dtau = np.abs(tau_der_t)
        abs_dtz = np.abs(dtz)
        return np.sum(weights[:, None, None] * abs_dtau[:, None, None] / abs_dtz, axis=0)

    ages = age(X, Y)
    ages = np.where(mask_img, ages, np.nan)
    age_line = ages[n_grid//2,n_grid//2:]
    #plt.plot(ages[n_grid//2,:])
    levels = []
    for i in range(len(age_line)):
        if not np.isnan(age_line[i]):
            levels.append(age_line[i])
    levels = np.sort(np.array([levels[i] for i in range(0,len(levels),20)]))

    #plt.imshow(ageLevels, extent=(-(1+L), (1+L), -(1+L), (1+L)), cmap='copper')# 0.3
    #plt.contour(ages[:,:], extent=(-lim, lim, -lim, lim), levels=levels, colors='black')
    #plt.plot(np.real(tau_trunk_t), np.imag(tau_trunk_t), 'white')
    #plt.plot(np.real(tau_knot_t), np.imag(tau_knot_t), 'white')
    #ax.set_aspect('equal', adjustable='box')
    #fig.savefig("boundary_integrals/figures/log_rings_rings.pdf", bbox_inches="tight")

    def age_knot(X, Y):
        Z = X + 1j * Y

        # Outer boundary
        dtz = tau_t[:, None, None] - Z[None, :, :]
        abs_dtau = np.abs(tau_der_t)
        abs_dtz = np.abs(dtz)
        log_age = np.sum(weights[:, None, None] * abs_dtau[:, None, None] / abs_dtz, axis=0)

        # Knot
        z_knot = 0.8 + 0.3*1j
        age_without_knot = np.sum(weights * abs_dtau / np.abs(z_knot - tau_t))
        knot_age = log_age[n_grid//2, n_grid//2]
        exp_coef = knot_age - age_without_knot
        r_knot = 0.2
        dtz = np.abs(z_knot - Z)
        correction = np.exp(-dtz**2 / r_knot**2) * exp_coef * 0.8
        return log_age + correction


    ages = age_knot(X, Y)
    ages = np.where(mask_img, ages, np.nan)

    ageLevels = np.sum((ages[None, :, :] < levels[:, None, None]), axis=0)
    #ageLevels = np.where(mask_img, ageLevels ** 0.2, np.nan)
    # plt.imshow(ageLevels, extent=(-(1+L), (1+L), -(1+L), (1+L)), cmap='copper')# 0.3
    plt.contour(ages[:, :], extent=(-lim, lim, -lim, lim), levels=levels, colors='black')
    # plt.plot(np.real(tau_trunk_t), np.imag(tau_trunk_t), 'white')
    # plt.plot(np.real(tau_knot_t), np.imag(tau_knot_t), 'white')
    # ax.set_aspect('equal', adjustable='box')
    # fig.savefig("boundary_integrals/figures/log_rings_knot.pdf", bbox_inches="tight")

    #ages_cut = ages[:-1,:-1]
    ageLvls = ageLevels[:-1,:-1]
    season = ageLvls % 2
    grad = (np.abs(ageLevels[1:,:-1] - ageLevels[:-1,:-1]) + np.abs(ageLevels[:-1,1:] - ageLevels[:-1, :-1]))>0
    corewood = ageLvls >= np.nanmax(ageLvls)/2
    barkwood = ageLvls <= 1
    woodcol = ((1 + np.nanmax(ageLvls)-ageLvls)**0.5)*(1 + 0.2*season)*(1 - 0.2*corewood - 0.8*barkwood) * (1-0.5*grad)
    #plt.imshow(barkwood)
    #woodcol = (np.nanmax(ageLvls)-ageLvls) ** 0.5
    woodcol = np.where(mask_img[:-1,:-1], woodcol, np.nan)
    plt.imshow(woodcol[::-1], cmap='copper', extent=(-lim, lim, -lim, lim))
    #plt.imshow((corewood + barkwood)*mask_img[:-1,:-1])


fig.savefig("boundary_integrals/figures/log_rings_col.pdf", bbox_inches="tight")