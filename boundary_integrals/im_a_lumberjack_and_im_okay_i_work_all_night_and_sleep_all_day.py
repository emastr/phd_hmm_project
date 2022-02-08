import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(0)


def remove_axis(ax):
    """Removes axis from plot"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)


def sample_shape():
    """
    This function should return two periodic functions.
    The first function is a parametrisation of the boundary
        tau: [0,2pi] -> C
    The second is a derivative of the first.
        tau_der(t) = (dtau / dt) (t)
    I have set it up such that
        tau(t) = exp(it) * r(t)
    where r(t) is a real valued fourier series:

        r(t) = c1 + c2 * cos(m2*t) + ... + cK cos(mK*t) + d1 sin(n1 * t) + ... + dK sin(nK * t)

    Where coef = [c0,c1,...,cK, d1,d2,..., dK] are the fourier coefficients and
          freqs = [0,m2,...,mK, m1,m2,..., mK] are the fourier frequencies.
    High frequency contributions are dampened both by being less likely to be picked, but also
    by the coefficients being chosen as a gaussian variable divided by the frequency.
    :return:
    """
    K = 10 # Number of Fourier terms

    # Sample fourier series describing the boundary
    freqs = np.round(np.abs(np.random.randn(2 * K, 1) * 10))
    freqs[0, 0] = 0  # Make the first element also the constant

    # Fourier coefficients
    coefs = 0.05 * np.random.randn(len(freqs), 1) / (1+freqs)
    coefs[0,0] += 1 # Make sure the radius fluctuates around 1.

    # Fourier terms, stacked in a matrix (cos then sin)
    fourier_basis = lambda t: np.vstack([np.cos(freqs[0:K, :] * t[None, :]), np.sin(freqs[K:, :] * t[None, :])])

    # Fourier terms for derivative, stacked in a matrix
    fourier_basis_der = lambda t: freqs * np.vstack([-np.sin(freqs[0:K, :] * t), np.cos(freqs[K:, :] * t)])

    # Radius function
    radius = lambda t: np.dot(coefs.T, fourier_basis(t)).flatten()

    # Derivative of radius function
    radius_der = lambda t: np.dot(coefs.T, fourier_basis_der(t)).flatten()

    # Create shape function tau
    tau = lambda t: radius(t) * np.exp(1j * t)

    # Derivative
    tau_der = lambda t: (radius_der(t) + 1j * radius(t)) * np.exp(1j * t)

    return tau, tau_der


def sample_knots():
    """
    This function samples a set of knots in the form of gaussian distributions centered at different points.

    First, the number of knots is randomised.

    Then, we choose equally spaced angles for the knots, with a small random perturbation to each knot,
    as well as a global random rotation to all the knots.

    Then, we chose a radius for the knots, with a small random perturbation to each knot.

    Then we chose parameters for the gaussian distribution (amplitude, decay, axis_ratio).
    the amplitude is self-explanatory, the decay is basically the variance of the gaussian,
    and the axis_ratio is the ratio of the eigenvalues in the covariance matrix that we would
    get if we viewed the gaussian as a distribution. You could randomise these three if you want! :)


    :return:
    """
    # Number of knots: 0<= n <=5
    num_knots = np.random.randint(0,6)

    # Angles
    # Equally spaced angles + random shift of at most 2pi/20
    angles = np.linspace(0,2*np.pi,num_knots+1)[:-1] + (np.random.rand(num_knots)-0.5) * np.pi * 2/20
    angle_shift = np.random.rand() * np.pi * 2 # in [0, 2pi]
    angles = (angle_shift + angles)  # shift and project to [0,2pi]
    angle_vecs = np.exp(1j * angles)

    # Radial distance (from center of log) between 0.3 and 0.6 + small random deviation for each knot
    radial_dist = 0.4 + np.random.rand()*0.3 + np.random.randn(num_knots)*0.05
    amplitude = 4 # Size of the knot
    decay = 0.2
    axis_ratio = 1/4 # Ratio of short to long axis

    knot_centers = angle_vecs * radial_dist
    def knot_eval(z):
        # Sum up all knots!
        ans = np.zeros_like(z)
        for i in range(num_knots):
            dz = knot_centers[i] - z
            dz_rot = dz / angle_vecs[i]
            knot_eval = np.exp(-np.real(dz_rot)**2 / (decay ** 2) - np.imag(dz_rot)**2/(decay * axis_ratio)**2) * amplitude

            ans += knot_eval
        return np.real(ans)
    return knot_eval


def age_without_knots(Z, tau_t, tau_der_t, weights):
    """
    Integral of singular funnctions around the  boundary creating the age rings.
    :param Z:
    :param tau_t:
    :param tau_der_t:
    :param weights: weights for the quadrature. = dt if we do trapezoid
    :return:
    """
    dtz = tau_t[:, None, None] - Z[None, :, :]
    abs_dtau = np.abs(tau_der_t)
    abs_dtz = np.abs(dtz)
    return np.sum(weights[:, None, None] * abs_dtau[:, None, None] / abs_dtz, axis=0)


def mask(Z, tau_t, tau_der_t, weights):
    """
    Mask function for separating the log from the background.
    For this function it is important to get the quadrature weights correct.
    :param Z:
    :param tau_t:
    :param tau_der_t:
    :param weights:
    :return:
    """
    Z = X + 1j*Y
    # Outer boundary
    dtz = tau_t[:, None, None] - Z[None, :, :]
    discriminator = np.sum(weights[:, None, None] * np.imag(tau_der_t[:,None,None] / dtz), axis=0) / (2 * np.pi)
    return (discriminator > 0.5) #* (discriminator_inner < 0.5)


def sample_log_age_fcn():
    """
    Sample knots and a boundary to create a log "age" function.
    This function can then be evaluated on arbitrarily high resolution grids.
    The log age function when evaluated on a grid Z of complex variables,
    returns the age, the image mask and a vector of values that
        correspond to the age function, evaluated on the middle line through Z.
        For this to work, the grid Z should be symmetric about the origin. Sorry for spaghetti!
    Also returns tau, the function describing the boundary.
    """
    # Sample a log age function.
    # This can be used to make images of the log at arbitrary resolutions

    tau, tau_der = sample_shape()

    # Grid segments for numerical integration
    N = 100
    gridpts = np.linspace(0, np.pi * 2, N + 1)[:-1]  # Remove last segment
    weights = np.ones_like(gridpts) * (gridpts[1] - gridpts[0])

    # Boundary coords
    tau_t = tau(gridpts)

    # Boundary derivatives
    tau_der_t = tau_der(gridpts)

    # Make knot function
    knot_eval = sample_knots()

    def age_fcn(Z):
        mask_img = mask(Z, tau_t, tau_der_t, weights)
        ages = age_without_knots(Z, tau_t, tau_der_t, weights)
        ages = np.where(mask_img, ages, np.nan)
        age_line = ages[n_grid // 2, n_grid // 2:]


        # Pick equally spaced level sets in the wood.
        levels = []
        for i in range(len(age_line)):
            if not np.isnan(age_line[i]):
                levels.append(age_line[i])
        levels = np.sort(np.array(levels))

        # Add knots
        ages = ages + knot_eval(Z)

        # Set values outside log to nan
        ages = np.where(mask_img, ages, np.nan)

        # Return age function, return mask, return levels, return boundary
        return ages, mask_img, levels
    return age_fcn, tau


# Prepare some things before looping through some examples
t_for_plots = np.linspace(0, 2*np.pi, 100)
fig = plt.figure(figsize=(12,6))

# Generate 4 logs :)
for i in range(4):
    # Sample log function and log boundary
    age_fcn, bdry_fcn = sample_log_age_fcn()

    # Limits of log boundary to determine suitable padding
    bdry_pts = bdry_fcn(t_for_plots)
    x_bdry, y_bdry = np.real(bdry_pts), np.imag(bdry_pts)
    xmax, xmin = np.max(x_bdry), np.min(x_bdry)
    ymax, ymin = np.max(y_bdry), np.min(y_bdry)

    # Pad limits for plotting, to make sure entire log fits
    pad = 0.1
    lim = (1+pad)*max([xmax, -xmin, ymax, -ymin])

    # Create grid
    n_grid = 500 # Image resolution
    x_mesh = np.linspace(-lim, lim, n_grid)
    y_mesh = np.linspace(-lim, lim, n_grid)
    X, Y = np.meshgrid(y_mesh, x_mesh)
    Z = X + 1j * Y

    # Evaluate age fcn on grid
    ages, mask_img, levels = age_fcn(Z)


    # Example of simple log plot

    # Pick some level sets
    sparse_levels = np.array([levels[i] for i in range(0,len(levels),20)])

    # Plot results
    plt.subplot(2, 4, i + 1)
    # Plot outer boundary
    plt.plot(x_bdry, y_bdry, 'black', linewidth=2)
    # Plot contours
    plt.contour(ages[:, :], extent=(-lim, lim, -lim, lim), levels=sparse_levels, colors='black')
    # Make plot prettier :)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    remove_axis(ax)


    ### Example of more advanced log density
    freq = 30 # Roughly the number of rings - you could randomise this! (to g
    sigmoid = lambda x: 1 / (np.exp(-x) + 1)
    density_step = lambda r: ((sigmoid((r - 0.7) * 200) + 1) + 1)  # Density increases from 1 to 2 halfway out.
    year_rings = lambda r: ((r * freq - np.floor(r * freq))) ** 2  # Density peaks sharply
    wood_density = lambda r: (-year_rings(r) + 1.3) * density_step(r)

    percentage_from_center = (ages[None, :, :] > levels.flatten()[:, None, None]).sum(axis=0).astype(float)
    percentage_from_center /= np.nanmax(percentage_from_center) # Normalise to [0,1]
    density = wood_density(percentage_from_center)
    density = np.where(mask_img, density, 0)

    plt.subplot(2, 4, i + 1+4)
    plt.imshow(density[::-1,:], cmap='Greys_r') # Reverse row order to get the same orientation as the contour plot.
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    remove_axis(ax)


plt.show()