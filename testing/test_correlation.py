"""
"""
import numpy as np
from numpy.testing import assert_allclose

import unittest
import spatialstats as ss

# Seed, for reproducibility
np.random.seed(0)

class TestMorisita(unittest.TestCase):

    def test_poisson_2d(self):
        """
        Calculate the morisita index for independently sampled 2d points, and
        check that the value is close to 1.
        """

        N = 100**2
        d = 2
        points = ss.generate_poisson(N, d)

        m_arr, morisita_arr = ss.morisita_index(points)

        assert_allclose(morisita_arr, 1, rtol=1e-2, atol=1e-3)

    def test_poisson_3d(self):
        """
        Calculate the morisita index for independently sampled 3d points, and
        check that the value is close to 1.
        """

        N = 20**3
        d = 3
        points = ss.generate_poisson(N, d)

        m_arr, morisita_arr = ss.morisita_index(points)

        assert_allclose(morisita_arr, 1, rtol=1e-2, atol=1e-3)

    def test_morisita_quadrat_iod_compare(self):
        r"""
        Calculate the morisita index and the quadrat index of dispersion
        and check that they are related by the derived expression:

        .. math::

            M(m) = \frac{1}{n - 1)} \left[ I(m) (m - 1) + n - m \right]

        We could also compute I(m) this way, but I find that it has a higher
        error, so it's better to compare just the morisita index.
        """

        N = 100**2
        d = 2
        points = ss.generate_poisson(N, d)

        m_arr, morisita_arr = ss.morisita_index(points)
        print(len(m_arr))
        m_arr, iod_arr = ss.quadrat_iod(points)
        print(len(m_arr))

        #calculated_iod_arr = m_arr / (N * (m_arr - 1)) * ((N - 1) * morisita_arr - (N/m_arr - 1))
        calculated_morisita_arr = 1 / (N - 1) * (iod_arr * (m_arr - 1) + N - m_arr)

        #assert_allclose(iod_arr, calculated_iod_arr, rtol=1e-2, atol=1e-3)
        assert_allclose(morisita_arr, calculated_morisita_arr, rtol=1e-2, atol=1e-3)
