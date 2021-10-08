import numpy as np

class GaussLegGrid:


    def __init__(self, segments, corners=None):
        """
        Discretisation of grid
        :param segments: on the form [t_0, t_1, t_2, ..., t_n]
                         where each segment [t_i, t_{i+1}] is smooth.
        :param tau:
        :param taup:
        :param taupp:
        """
        self.ABSCISSA = np.array([-0.0950125098376374,
                                   0.0950125098376374,
                                  -0.2816035507792589,
                                   0.2816035507792589,
                                  -0.4580167776572274,
                                   0.4580167776572274,
                                  -0.6178762444026438,
                                   0.6178762444026438,
                                  -0.7554044083550030,
                                   0.7554044083550030,
                                  -0.8656312023878318,
                                   0.8656312023878318,
                                  -0.9445750230732326,
                                   0.9445750230732326,
                                  -0.9894009349916499,
                                   0.9894009349916499])

        self.WEIGHTS = np.array([0.1894506104550685,
                                 0.1894506104550685,
                                 0.1826034150449236,
                                 0.1826034150449236,
                                 0.1691565193950025,
                                 0.1691565193950025,
                                 0.1495959888165767,
                                 0.1495959888165767,
                                 0.1246289712555339,
                                 0.1246289712555339,
                                 0.0951585116824928,
                                 0.0951585116824928,
                                 0.0622535239386479,
                                 0.0622535239386479,
                                 0.0271524594117541,
                                 0.0271524594117541])
        args = np.argsort(self.ABSCISSA)
        self.ABSCISSA = self.ABSCISSA[args]
        self.WEIGHTS = self.WEIGHTS[args]

        self.segments = segments
        if corners is None:
            self.corners = np.ones_like(segments).astype(int) # This can be refined.
        else:
            self.corners = corners

    def get_grid_and_weights(self):
        n_segments = len(self.segments) - 1
        segment_size = len(self.ABSCISSA)
        grid = np.zeros((n_segments * segment_size,))
        weights = np.zeros((n_segments * segment_size,))
        for n in range(n_segments):
            grid[n * segment_size: (n+1)*segment_size] = self.rescale_abscissa(self.segments[n], self.segments[n+1])
            weights[n * segment_size: (n+1)*segment_size] = self.rescale_weights(self.segments[n], self.segments[n+1])

        return grid, weights

    def rescale_abscissa(self, a, b):
        return (a + b)/2 + self.ABSCISSA * (b - a)/2

    def rescale_weights(self, a, b):
        return self.WEIGHTS * (b - a)/2

    def refine_all(self):
        self.segments, _ = GaussLegGrid.refine_segments_at_corners(self.segments, np.ones_like(self.corners))
        old_corners = self.corners
        self.corners = np.zeros_like(self.segments).astype(int)
        for n in range(len(old_corners)):
            self.corners[n * 2] = old_corners[n]

    def refine_all_nply(self, n):
        for i in range(n):
            self.refine_all()

    def refine_corners(self):
        self.segments, self.corners = GaussLegGrid.refine_segments_at_corners(self.segments, self.corners)

    def refine_corners_nply(self, n):
        for i in range(n):
            self.refine_corners()

    @staticmethod
    def refine_segments_at_corners(segments, corners):
        old_size = len(corners)
        tight_corners = np.roll(corners, 1) * corners
        num_tight_corners = np.sum(tight_corners)
        num_corners = np.sum(corners)

        # No need to refine twice if corners are directly connected (tight) # Remove last element, should be a cycle.
        new_size = old_size + 2 * num_corners - num_tight_corners
        new_corners = np.zeros(new_size)
        new_segments = np.zeros(new_size)

        new_idx = np.arange(0, old_size, 1)# + np.cumsum(2 * self.sharp_corners)
        new_idx = new_idx\
                  + np.roll(np.cumsum(corners), 1)\
                  + np.roll(np.cumsum(corners), 0)\
                  - np.cumsum(tight_corners)\
                  + corners[-1]*corners[0] - 1
        new_idx[0] = 0 # First corner never moves.
        new_corners[new_idx] = corners
        new_segments[new_idx] = segments

        for i in range(1, len(new_corners)-1):
           if new_corners[i] != 1:
               if new_corners[i+1] == 1 or new_corners[i-1] == 1:
                   new_segments[i] = (new_segments[i-1] + new_segments[i+1])/2

        # Ignore the last grid point
        segments = new_segments[:-1]
        corners = new_corners.astype(int)[:-1]

        return segments, corners

    @staticmethod
    def refine_segments_at_corners_nply(self, n, segments, corners):
        for i in range(n):
            segments, corners = GaussLegGrid.refine_segments_at_corners(segments, corners)

    @staticmethod
    def integrate_vec(fvals, weights):
        return np.sum(fvals * weights, axis=0)

    @staticmethod
    def integrate_func(func, gridpts, weights):
        return GaussLegGrid.integrate_vec(func(gridpts), weights)


# def refine_corners(self):
#     old_size = len(self.sharp_corners)
#     tight_corners = np.roll(self.sharp_corners, 1) * self.sharp_corners
#     num_tight_corners = np.sum(tight_corners)
#     num_sharp_corners = np.sum(self.sharp_corners)
#
#     # No need to refine twice if corners are directly connected (tight) # Remove last element, should be a cycle.
#     new_size = old_size + 2 * num_sharp_corners - num_tight_corners
#     new_sharp_corners = np.zeros(new_size)
#     new_segments = np.zeros(new_size)
#
#     new_idx = np.arange(0, old_size, 1)# + np.cumsum(2 * self.sharp_corners)
#     new_idx = new_idx\
#               + np.roll(np.cumsum(self.sharp_corners), 1)\
#               + np.roll(np.cumsum(self.sharp_corners), 0)\
#               - np.cumsum(tight_corners)\
#               + self.sharp_corners[-1]*self.sharp_corners[0] - 1
#     new_idx[0] = 0 # First corner never moves.
#     new_sharp_corners[new_idx] = self.sharp_corners
#     new_segments[new_idx] = self.segments
#
#     for i in range(1, len(new_sharp_corners)-1):
#        if new_sharp_corners[i] != 1:
#            if new_sharp_corners[i+1] == 1 or new_sharp_corners[i-1] == 1:
#                new_segments[i] = (new_segments[i-1] + new_segments[i+1])/2
#
#     # Ignore the last grid point
#     self.segments = new_segments[:-1]
#     self.sharp_corners = new_sharp_corners.astype(int)[:-1]
#
# def refine_corners_nply(self, n):
#     for i in range(n):
#         self.refine_corners()