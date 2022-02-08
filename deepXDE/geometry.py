import deepxde as dde
from deepxde import geometry as gty
from deepxde.geometry.geometry import *
import matplotlib.pyplot as plt

class MicroProblem(Geometry):

    def __init__(self, boundary_function, x_lims, bbox_y_lims):
        """
        Micro Problem For the Multi Scale Stokes Flow.
        :param tau      parametrisation of the lower boundary
        :param epsilon  positive real number such that max(tau) <= epsilon (important for sampling)
        :param height:  height above the boundary to put the top bounding box.

        x_min=0
        |
        v                       v
        _________________________   ___  <===== y_top
        |                       |    |
        |                       |    |
        |        Geometry       |  height
        |                       |    |
        |                       |    |
        ____      ____      ____|   _|_
            \    /    \    /         |epsilon
             \__/      \__/         _|_  <====== y_bot

        """

        self.dim = 2
        self.x_lims = x_lims
        self.y_lims = bbox_y_lims
        self.bdry_fcn = boundary_function


        # Weights for sampling uniformly from boundaries: [left, top, right, bot]
        x = np.linspace(x_lims[0], x_lims[1], 200)
        y = boundary_function(x)
        segment_length = np.sum(((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)**0.5)

        self.segment_lengths = np.array([bbox_y_lims[1]-boundary_function(x_lims[0]),
                                x_lims[1]-x_lims[0],
                                bbox_y_lims[1] - boundary_function(x_lims[1]),
                                segment_length])
        #self.segment_weights /= np.sum(self.segment_weights) #normalise.


        # Construct bounding box to sample from (We don't need points from the boundary)
        self.bbox = gty.Rectangle(xmin=[self.x_lims[0], self.y_lims[0]], xmax=[self.x_lims[1], self.y_lims[1]])


    def inside(self, x):
        inside_bbox = self.bbox.inside(x)
        inside_bdry = np.where(inside_bbox, x[:, 1] >= self.bdry_fcn(x[:, 0]),
                               np.full(inside_bbox.shape, False, dtype=bool))
        return inside_bdry

    def on_boundary(self, x, tol=1e-8):

        # If on bbox boundary
        on_bbox_boundary = self.bbox.on_boundary(x, tol=tol)
        on_rough_boundary = np.isclose(x[:, 1], self.bdry_fcn(x[:, 0]), atol=tol)
        inside = self.inside(x)
        return np.logical_and(inside, np.logical_or(on_bbox_boundary, on_rough_boundary))

    def random_boundary_points(self, n, random="pseudo"):
        x = np.zeros((n, 2))
        segment_no = np.random.choice(a=list(range(4)), size=(n), p=self.segment_lengths/np.sum(self.segment_lengths))
        bdry_param = np.random.rand(n)

        # Left bdry

        for idx in range(n):
            if segment_no[idx] == 0:
                x[idx, 0] = self.x_lims[0]
                x[idx, 1] = bdry_param[idx] * self.segment_lengths[0] + self.bdry_fcn(self.x_lims[0])
            elif segment_no[idx] == 1:
                x[idx, 0] = bdry_param[idx] * self.segment_lengths[1] + self.x_lims[0]
                x[idx, 1] = self.y_lims[1]
            elif segment_no[idx] == 2:
                x[idx, 0] = self.x_lims[1]
                x[idx, 1] = bdry_param[idx] * self.segment_lengths[2] + self.bdry_fcn(self.x_lims[1])
            else:
                x_rand = bdry_param[idx] * self.segment_lengths[0] + self.bdry_fcn(self.x_lims[0])
                x[idx, 0] = x_rand
                x[idx, 1] = self.bdry_fcn(x_rand)

        return x

    def random_points(self, n, random="pseudo"):
        # Importance sampling
        # Slow and steady
        x = self.bbox.random_points(n)
        not_inside = np.logical_not(self.inside(x))
        while np.any(not_inside):
            x[not_inside] = self.bbox.random_points(np.sum(not_inside))
            not_inside = np.logical_not(self.inside(x))
        return x

    def uniform_points(self, n, boundary=True):
        """Uniform grid of points in the bounding box. Apply 'inside' mask to see which is inside the domain"""
        return self.bbox.uniform_points(n, boundary=boundary)

    @staticmethod
    def test_indicators():

        # Create Test Geometry
        x_lims = [0,1]
        y_lims= [-0.1,1]
        boundary = lambda x: np.sin(20*x)/10
        microProb = MicroProblem(boundary_function=boundary, x_lims=x_lims, bbox_y_lims=y_lims)


        # Discretise domain with grid
        N = 100
        eps = 0.01
        x = np.linspace(x_lims[0]-eps, x_lims[1]+eps, N)[None, :]
        y = np.linspace(y_lims[0]-eps, y_lims[1]+eps, N)[:, None]

        # Convert to mesh matrices
        z = x + 1j*y
        X = np.real(z).flatten()[:, None]
        Y = np.imag(z).flatten()[:, None]
        xy = np.hstack([X,Y])

        # Plot "inside" function
        inside = microProb.inside(xy)

        # Only some should be on the boundary
        on_bdry = microProb.on_boundary(xy, tol=1e-2)

        fig=plt.figure(figsize=(500,100))

        fig.add_subplot(1, 2, 1)
        plt.pcolormesh(np.real(z), np.imag(z), inside.reshape(N, N))
        plt.plot(x.T, boundary(x).T, 'red')

        fig.add_subplot(1, 2, 2)
        plt.pcolormesh(np.real(z), np.imag(z), on_bdry.reshape(N, N))

    @staticmethod
    def test_samplers():
        x_lims = [0,1]
        y_lims= [-0.1,1]
        boundary = lambda x: np.sin(20*x)/10
        microProb = MicroProblem(boundary_function=boundary, x_lims=x_lims, bbox_y_lims=y_lims)


        N = 10000

        x = microProb.random_boundary_points(N)
        plt.scatter(x[:,0], x[:, 1])


        x = microProb.random_points(N)
        plt.scatter(x[:, 0], x[:, 1])



