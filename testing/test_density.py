"""
"""
import numpy as np
from numpy.testing import assert_allclose

import unittest
import spatialstats as ss

# Seed, for reproducibility
np.random.seed(0)

class TestDensity(unittest.TestCase):

    def test_uniform_density(self):
        """
        Test calculating the density field when there
        is a single point in each bin.
        """
        N = 10

        # This creates a grid of points
        points = np.meshgrid(np.arange(N), np.arange(N))
        points = np.array([points[0].flatten(), points[1].flatten()]).T + 0.5

        # Now calculate our density field
        bounds = np.array([[0, N], [0, N]])
        # Use a square kernel of size 1, so we should just get a value of
        # 1 for each entry in the density field.
        field = ss.density_field(points, fixed_bounds=bounds, lattice_spacing=1, kernel='ones', kernel_size=1)

        # Field should be an array of just ones
        assert_allclose(field, np.ones_like(field), atol=0.01, rtol=0.01)
