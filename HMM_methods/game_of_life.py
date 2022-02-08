import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg



img = im.imread("HMM_methods/IMG_4044.jpg")
img_grey = img.mean(axis=2)
C = img_grey.astype(float)/255

plt.imshow(C, cmap="Greys_r")

Cbw = (np.random.randn(*C.shape) < C)
plt.imshow(Cbw, cmap="Greys_r")


neighbor_kernel = np.ones((3,3))
neighbor_kernel[1,1] = 0

plt.imshow(C, cmap="Greys_r")
for i in range(100):
    C = sg.convolve2d(C, neighbor_kernel, mode="same", boundary="wrap")#, fillvalue=0)
Cbw = 1-Cbw
K = 200
for k in range(K):
    num_neighbors = sg.convolve2d(Cbw, neighbor_kernel, mode="same", boundary="wrap")#, fillvalue=0)
    two_neighbors = num_neighbors == 2
    thr_neighbors = num_neighbors == 3
    Cbw = Cbw * two_neighbors + thr_neighbors
    if k % 20 == 0:
        plt.figure()
        plt.imshow(Cbw)