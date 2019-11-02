import unittest
import numpy as np
import scipy
import scipy.stats
import scipy.special
import sys
sys.path.append('../src/kldmwr/')
import parest


def norm_cdf(x, p):
    return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1])


def norm_pdf(x, p):
    return scipy.stats.norm.pdf(x, loc=p[0], scale=p[1])


class TestParest(unittest.TestCase):

    def test_carmps(self):
        x = [2, 1, 0]
        p_0 = [0, 1]
        res_test = parest.carmps(x, norm_cdf, p_0)
        np.testing.assert_almost_equal(res_test[0], 1, decimal=6)

    def test_jmmps(self):
        x = [3, 2, 1]
        p_0 = [0, 1]
        res_test = parest.carmps(x, norm_cdf, p_0)
        np.testing.assert_almost_equal(res_test[0], 2, decimal=6)


class TestParestObservation01(unittest.TestCase):

    def setUp(self):
        x = [1., 2., 3., 3., 4., 5., 5., 5., 6.]
        self.testparest = parest.Observation(x)

    def test_x(self):
        x2 = [1., 2., 3., 3., 4., 5., 5., 5., 6.]
        np.testing.assert_equal(self.testparest.x, x2)

    def test_n_unq(self):
        n_unq_set = 6
        res = self.testparest.n_unq
        self.assertEqual(res, n_unq_set)

    def test_handle_ties(self):
        unq, lambdas = self.testparest.handle_ties()
        np.testing.assert_equal(unq, [1., 2., 3., 4., 5., 6.])
        np.testing.assert_equal(lambdas, [1, 1, 2, 1, 3, 1])

    def test_handle_ties_ra(self):
        wnus_ra = self.testparest.handle_ties_ra()
        wnus_ra_set = [0.5, 1., 1.5, 1.5, 2., 2., 0.5]
        np.testing.assert_almost_equal(wnus_ra, wnus_ra_set, decimal=4)

    def test_handle_ties_tr(self):
        wnus_tr = self.testparest.handle_ties_tr()
        wnus_tr_set = [1, 1, 2, 1, 3, 1, 1]
        np.testing.assert_almost_equal(wnus_tr, wnus_tr_set, decimal=4)

    def test_ramps_attributes(self):
        self.n_unq = len(self.testparest.unq)
        self.fn_dist = None
        self.fn_dens = None
        self.fn_ipf = None
        self.pars_fixed = None
        self.dens = np.zeros(self.n_unq)
        self.dist = np.zeros(self.n_unq + 2)
        self.dist[self.n_unq + 1] = 1.0
        self.spacings = np.zeros(self.n_unq + 1)
        self.log_spacings = np.zeros(self.n_unq + 1)
        self.ipfcis = np.zeros(len(self.testparest.x))
        self.invcis = np.zeros(len(self.testparest.x))
        self.k_ipf = 0.0
        self.assertEqual(self.testparest.fn_dist, None)
        self.assertEqual(self.testparest.fn_dens, None)
        self.assertEqual(self.testparest.fn_ipf, None)
        np.testing.assert_equal(self.testparest.dens, np.zeros(6))
        np.testing.assert_equal(self.testparest.dist[: - 1], np.zeros(7))
        a = self.testparest.dist[-1]
        np.testing.assert_almost_equal(a, 1.0, decimal=5)
        np.testing.assert_equal(self.testparest.spacings, np.zeros(7))
        np.testing.assert_equal(self.testparest.log_spacings, np.zeros(7))
        np.testing.assert_equal(self.testparest.ipfcis, np.zeros(9))
        np.testing.assert_equal(self.testparest.invcis, np.zeros(9))

    def test_set_x(self):
        y = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
        self.testparest.set_x(y)
        np.testing.assert_equal(self.testparest.x, y)


class TestParestObservation(unittest.TestCase):
    def setUp(self):
        x = [.2, .2, .4, .6, .6, .8, .8, .8]
        self.testparest02 = parest.Observation(x)
    
    @staticmethod
    def ipf(x, p):
        return x[0] * p[0] * 0

    def test_mlraps(self):
        self.testparest02.fn_dist = norm_cdf
        self.testparest02.fn_ipf = self.ipf
        p = [0.3, 0.5]
        self.testparest02.wnus = self.testparest02.handle_ties_ra()
        res = self.testparest02.nlps(p)
        a = [
            14.6654
        ]
        np.testing.assert_almost_equal(a, res, decimal=4)

    def test_nltrps(self):
        self.testparest02.fn_dist = norm_cdf
        self.testparest02.fn_ipf = self.ipf
        p = [0.3, 0.5]
        self.testparest02.wnus = self.testparest02.handle_ties_tr()
        res = self.testparest02.nlps(p)
        a = [
            15.7289
        ]
        np.testing.assert_almost_equal(a, res, decimal=4)

    def test_nll(self):
        self.testparest02.fn_dens = norm_pdf
        self.testparest02.fn_ipf = self.ipf
        p = [0.3, 0.5]
        self.testparest02.unq, self.testparest02.lambdas = \
            self.testparest02.handle_ties()
        res = self.testparest02.nll(p)
        a = [
            3.7263
        ]
        np.testing.assert_almost_equal(a, res, decimal=4)


if __name__ == '__main__':
    unittest.main()
