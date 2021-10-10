import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

# Load Data
image = im.imread("curve_fitting/curve.png")
curve = image[:,:,0]

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

plt.scatter(train_data[:,1], train_data[:,2])


t_test = np.linspace(0, 2*np.pi, 100)[:, None]
# Initialise fourier coefficients and weights
N = 10
N_bfuncs = N * 2 -1
weights = np.arange(0, N, 1)[:, None]
bfuncs = lambda t : np.vstack([np.cos(weights @ t.T), np.sin(weights[1:] @ t.T)])
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

plt.scatter(train_data[:, 1], train_data[:, 2])
step = 0.01
K = 150000
n_t = 100
n_subsample = 200
p = 3
for k in range(K):
    subsample_idx = np.random.randint(low=0, high=train_data.shape[0], size=n_subsample)
    train_data_subsample = train_data[subsample_idx]
    # Sample points from [0,2pi]
    t = np.random.rand(n_t,1) * 2 * np.pi
    # Evaluate fourier series in these points
    fourier_mat = bfuncs(t)
    gamma = fourier_mat.T @ coefs
    # Divide data into sets based on which parameter choice t is closest
    gamma_repeat = np.repeat(gamma[:,None,:], n_subsample, axis=1)
    residual = (gamma_repeat - train_data_subsample[:, 1:])
    residual_norm = (residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2) ** 0.5
    args = np.argmin(residual_norm, axis=0)
    gamma_args = gamma[args, :]
    residual_closest = gamma_args - train_data_subsample[:, 1:]
    fourier_mat_closest = fourier_mat[:, args]
    #args = np.argsort((residual[:,0]**2 + residual[:, 1]**2)**0.5)
    #residual = gamma - residual[args, :]
    #fourier_mat.shape

    coefs = coefs - step * fourier_mat_closest @ residual_closest / residual_closest.shape[0]

    gamma = bfuncs(t_test).T @ coefs
    #plt.plot(gamma[:, 0], gamma[:, 1])
    #plt.plot(t_test, gamma)
    if k%100 == 0:
        plt.plot(gamma[:, 0], gamma[:, 1], color=[1 - k/K, 0, 0])
#plt.scatter(gamma[:,0], gamma[:, 1])

plt.scatter(train_data_subsample[:, 1], train_data_subsample[:, 2], c=args)