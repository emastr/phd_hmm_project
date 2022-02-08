import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

# Load Data
image = im.imread("curve_fitting/curve.png")
curve = image[::-1,:,0].T

# Extract the "colored pixels"
num_data = (curve != 0).sum()
train_data = np.zeros((num_data, 3))  # value, pos_x, pos_y for each data

# Loop through image and add data
k = 0
for i in range(curve.shape[0]):
    for j in range(curve.shape[1]):
        if curve[i,j] != 0:
            train_data[k, 0] = curve[i,j]
            train_data[k, 1] = (i / curve.shape[0]) * 2 - 1
            train_data[k, 2] = (j / curve.shape[1]) * 2 - 1
            k += 1


t_test = np.linspace(0, 2*np.pi, 100)[:, None]
# Initialise fourier coefficients and weights
N = 15
N_bfuncs = N * 2 -1
weights = np.arange(0, N, 1)[:, None]

weights_full = np.vstack([weights, weights[1:]]) # Weights
deriv_weights = np.vstack([-weights, weights[1:]]) # Weights as they come out after differentiating

abs_weights = np.abs(weights_full)
bfuncs = lambda t : np.vstack([np.cos(weights @ t.T), np.sin(weights[1:] @ t.T)])
bfuncsder = lambda t: np.vstack([-weights * np.sin(weights @ t.T), weights[1:] * np.cos(weights[1:] @ t.T)])
bfuncsder2 = lambda t: np.vstack([-weights**2 * np.cos(weights @ t.T), -weights[1:]**2 * np.sin(weights[1:] @ t.T)])

bfunc_variability = np.abs(np.vstack([weights, weights[1:]]))**2
coefs = 0 * np.random.normal(loc=0, scale=1, size=(N_bfuncs,2)) / (bfunc_variability +1)

# Improved initialisation (circle outside data)
max_x = np.max(train_data[:, 1])
min_x = np.min(train_data[:, 1])
rad_x = (max_x - min_x) / 2
center_x = (max_x + min_x) / 2
coefs[0,0] = center_x
coefs[1,0] = rad_x
#
max_y = np.max(train_data[:, 2])
min_y = np.min(train_data[:, 2])
rad_y = (max_y - min_y) / 2
center_y = (max_y + min_y) / 2
coefs[0,1] = center_y
coefs[N,1] = rad_y

plt.imshow(image[:,:,0], extent=(-1,1,-1,1), cmap="Greys_r")
#plt.scatter(train_data[:, 1], train_data[:, 2])
step = 0.05
K = 10000
n_t = 100
n_subsample = 300
p = 3
for k in range(K):
    subsample_idx = np.random.randint(low=0, high=train_data.shape[0], size=n_subsample)
    train_data_subsample = train_data[subsample_idx]

    # Sample points from [0,2pi]
    t = np.random.rand(n_t,1) * 2 * np.pi
    #t = np.linspace(0, 2*np.pi, n_t)[:, None]

    # Evaluate fourier series in these points
    fourier_mat = bfuncs(t)
    gamma = fourier_mat.T @ coefs

    # Divide data into sets based on which point t_n is closest
    gamma_repeat = np.repeat(gamma[:,None,:], n_subsample, axis=1)
    residual = (gamma_repeat - train_data_subsample[:, 1:])
    residual_norm = (residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2) ** 0.5
    args = np.argmin(residual_norm, axis=0)
    gamma_args = gamma[args, :]
    residual_closest = gamma_args - train_data_subsample[:, 1:]
    fourier_mat_closest = fourier_mat[:, args]

    # Add springs between the sampled points t_n
    space_distances = (gamma[:, None, :] - gamma[None, :, :])
    grid_distances = (t - t.T)
    all_forces = space_distances / (np.linalg.norm(space_distances, axis=2) ** 2 * np.abs(grid_distances)**2 + 0.01)[:,:,None]
    diag_idx = np.arange(0, all_forces.shape[0],1)
    forces = np.sum(all_forces, axis=1)

    ## Use hyperbola loss on remaining residuals (gets pulled to middle)
    # unique_args = np.unique(args).astype(int)
    # n_t_inactive = n_t - len(unique_args)
    # t_inactive = np.zeros((n_t_inactive,1))
    # idx_inactive = np.zeros((n_t_inactive,)).astype(int)
    # b = 0
    # for i in range(n_t):
    #     if i not in unique_args:
    #         t_inactive[b] = t[i]
    #         idx_inactive[b] = i
    #         b += 1
    #
    # fourier_mat_inactive = fourier_mat[:, idx_inactive]
    # gamma_inactive = gamma[idx_inactive, :]
    # residual_inactive = residual[idx_inactive, :, :]
    # gradient = np.mean(np.dot(fourier_mat_inactive, (residual_inactive / (residual_inactive ** 2 + 0.001) ** 0.5).transpose((1,0,2))), axis=1)

    ## Add a smoothing force
    # fourier_mat_der = bfuncsder(t)
    # gamma_der = fourier_mat_der.T @ coefs
    #
    # fourier_mat_der2 = bfuncsder2(t)
    # gamma_der2 = fourier_mat_der2.T @ coefs
    #
    # gamma_der_dot = np.linalg.norm(gamma_der, axis=1) ** 2 + 0.001
    # gamma_der_der2_dot = np.abs(gamma_der2[:,0]*gamma_der[:,0] + gamma_der2[:,1]*gamma_der[:,1]) + 0.001
    # sign_der_der2_dot = np.sign(gamma_der2[:,0]*gamma_der[:,0] + gamma_der2[:,1]*gamma_der[:,1])
    # rad_inv =  gamma_der_der2_dot**0.5/ gamma_der_dot
    # rad_inv_root_der = 0.5* rad_inv / gamma_der_der2_dot * sign_der_der2_dot
    #
    # curve_der = fourier_mat_der @ (-2 * gamma_der * (rad_inv / gamma_der_dot)[:,None] + 1/2 * gamma_der2  * rad_inv_root_der[:,None])\
    #             + fourier_mat_der2 @ (1/2 * gamma_der * rad_inv_root_der[:,None])
    # curve_der /= n_t


    # Second derivative norm regulariser with decay
    coefs = coefs - step * (fourier_mat_closest @ residual_closest / residual_closest.shape[0] + 0.001*coefs * (abs_weights +1)**4 * 1/(k+1)**1)

    # Add springyness to gridpoints
    #coefs = coefs - step * (fourier_mat_closest @ (residual_closest / (residual_closest **2 + 0.01) ** 0.5) / residual_closest.shape[0] + 0 * 1e-5 * fourier_mat @ forces)

    # Smoothing action (curvature penalty)
    #coefs = coefs - step * (fourier_mat_closest @ (residual_closest / (residual_closest ** 2 + 0.01) ** 0.5) /
    #                        residual_closest.shape[0] + curve_der * 0.01/(k+1)**2)

    # Small standard loss added
    #coefs = coefs - step * (fourier_mat_closest @ (residual_closest / (residual_closest ** 2 + 0.001) ** 0.5) /
    #                        residual_closest.shape[0] + 0.01*gradient)

    # Just hyperbola doesn't work
    #coefs = coefs - 0.001 * (np.dot(fourier_mat , (residual / (residual ** 2 + 0.01) ** 0.5).transpose((1, 0, 2)))).mean(axis=1) / residual.shape[0]

    # Hyperbola with subsets
    #coefs = coefs - step * (fourier_mat_closest @ (residual_closest / (residual_closest ** 2 + 0.001) ** 0.5))

    #plt.plot(gamma[:, 0], gamma[:, 1])
    #plt.plot(t_test, gamma)
    if k%100 == 0:
        print(f"{k}/{K} ({k/K * 100}%) done.")
        gamma = bfuncs(t_test).T @ coefs
        plt.plot(gamma[:, 0], gamma[:, 1], color=[1 - k/K, 0, k/K])
#plt.scatter(gamma[:,0], gamma[:, 1])


func = {"weights": weights, "coefs": coefs}
np.save("boundary_integrals/saved_cossin_bdries/curve", func, allow_pickle=True)
#plt.scatter(train_data_subsample[:, 1], train_data_subsample[:, 2], c=args)