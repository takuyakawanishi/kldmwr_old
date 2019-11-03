"""This module is for parameter estimation.

   The implemented methods are

   Maximum Likelihood
   Maximum Product of Spacings (MPS)
       by Cheng and Amin(1983) and Raneaby (1984)
   Jiang's Modified Maximum Product Spacings (JMMPS)
       by Jiang (2014)

"""


import numpy as np
from scipy.optimize import minimize
import warnings


warnings.filterwarnings('ignore')


##############################################################################
#
#  Class Observation
#
##############################################################################


class Observation(object):

    """Class of data instance for parameter estmation.

    Args:
        x: array like
            The data itself.

    Attributes:
        x: numpy.ndarray
            Data.
        unq: numpy.ndarray of float
            The vector of unique values of x, sorted.
        lambdas: numpy.ndarray of int
            The vector of number of occurrence of each value in unq.
        n_unq: integer
            The number of unique values in x.
        fn_dist: function
            Distribution function to fit.
        fn_dens: function
            Density function to fit.
        dist: numpy.ndarray of float
            Vector of values of the distribution function.
        dens: numpy.ndarray of float
            Vector of values of the density function.
        spacings: numpy.ndarray of float
            Vector of values of spacings.
        log_spacings: numpy.ndarray of float
            Vector of values of log-spacings.

    """

    def __init__(self, x):

        self.x = None
        self.unq = None
        self.lambdas = None
        self.wnus = None
        self.n_unq = None
        self.fn_dist = None
        self.fn_dens = None
        self.dist = None
        self.dens = None
        self.spacings = None
        self.log_spacings = None
        #
        # Interior penalty function
        #
        self.fn_ipf = None
        self.ipfcis = None
        self.invcis = None
        self.k_ipf = None
        self.r_ipf = None
        self.set_x(x)

    def set_x(self, x):

        self.x = np.sort(np.array(x))
        self.unq, self.lambdas = self.handle_ties()
        self.n_unq = len(self.unq)
        self.dens = np.zeros(self.n_unq)
        self.dist = np.zeros(self.n_unq + 2)
        self.dist[self.n_unq + 1] = 1.0
        self.spacings = np.zeros(self.n_unq + 1)
        self.log_spacings = np.zeros(self.n_unq + 1)
        #
        # For interior penalty function
        #
        self.ipfcis = np.zeros(len(self.x))
        self.invcis = np.zeros(len(self.x))
        self.k_ipf = 0.0

    ##########################################################################
    #  Data Handling Essentials (called at the creation of instance).
    ##########################################################################

    def handle_ties(self):
        unq, lambdas = np.unique(self.x, return_counts=True)
        return unq, lambdas

    def handle_ties_tr(self):
        wnus_tr = np.zeros(len(self.unq) + 1)
        wnus_tr[:-1] = self.lambdas
        wnus_tr[-1] = 1.0
        return wnus_tr

    def handle_ties_ra(self):
        wnus_ra = np.zeros(len(self.unq) + 1)
        wnus_ra[0] = 0.5 * self.lambdas[0]
        wnus_ra[1: -1] = 0.5 * (self.lambdas[1:] + self.lambdas[:-1])
        wnus_ra[-1] = 0.5 * self.lambdas[-1]
        return wnus_ra

    ##########################################################################
    #  Finding MLE.
    ##########################################################################

    def find_mle(self, density, pars_to_fit, method='Nelder-Mead',
                 variant=None, ipf=None, k_ipf=2.0, r_ipf=0.05,
                 xatol=1.e-6, fatol=1.e-6, maxfev=10000, maxiter=10000):

        """Returns the ML estimator, log-ML, success or not, and
           the return value of the scipy.minimize

        Args:
            density: function
                Density function defined by the user.
            pars_to_fit: array like
                Initial values of the parameters.
            method: string
                The method used in scipy.optimize.minimize, the default is
                'Nelder-Mead'.
            variant: optional, string
                Default is None.
            ipf: optional, function
                Interior penalty function.
            k_ipf: optional, float
                Initial K-factor interior penalty function. Default = 2.
            r_ipf: optional, float
                Multiplier of IPF, default = 0.05.
            xatol: optional, float
                Paremeter for scipy.optimize.minimize.
            fatol: optional, float
                Paremeter for scipy.optimize.minimize.
            maxfev: optional, integer
                Parameter for scipy.optimize.minimize.
            maxiter: optional, integer
                Parameter for scipy.optimize.minimize.

        Returns:
            Tuple of (tuple of (vector of parameters, log-ml, success(boolean),
                return value of scipy.optimize.minimize)).
        """

        self.fn_dens = density
        cond1 = variant is not None
        cond2 = variant is not 'ml'
        if cond1 and cond2:
            print("Warning: 'variant' is neither None nor 'ml' but '",
                  variant, "'.")
        self.fn_ipf = ipf if ipf is not None else self.ipf_zero
        self.k_ipf = k_ipf if ipf is not None else 0.0
        self.r_ipf = r_ipf if ipf is not None else 0.0
        p = np.copy(pars_to_fit)
        epsilon = 1.0
        res = None
        while epsilon > 1e-6:
            res = minimize(self.nll, p,
                           options={'maxfev': maxfev, 'maxiter': maxiter,
                                    'xatol': xatol, 'fatol': fatol},
                           method=method)
            self.k_ipf *= self.r_ipf
            p = np.array(res.x)
            epsilon = self.fn_ipf(self.x, p) * self.k_ipf
        return res.x, - self.nll(res.x), res.success, res

    ##########################################################################
    #  Finding MPSE, or JMMPSE.
    ##########################################################################

    def find_mpse(self, distribution, pars_to_fit, method='Nelder-Mead',
                  variant='ra', ipf=None, k_ipf=2., r_ipf=.05,
                  xatol=1e-6, fatol=1e-6, maxfev=5000, maxiter=5000):

        """Returns the MPS estimator, log-PS, success or not, and
           the return value of the scipy.minimize

        Args:
            distribution: function
                Distribution function defined by the user.
            pars_to_fit: array like
                Initial values of the parameters.
            method: string
                The method used in scipy.optimize.minimize,default is
                'Nelder-Mead'.
            variant: optional, string
                Type of the MPS, either 'ra' for Observation or 'tr' for MPS,
                default is 'ra'.
            ipf: optional, function
                Interior penalty function.
            k_ipf: optional, float
                Initial K-factor for interior penalty function. Default = 2.
            r_ipf: optional, float
                Multiplier of IPF, default = 0.05.
            xatol: optional, float
                Parameter for scipy.optimize.minimize.
            fatol: optional, float
                Parameter for scipy.optimize.minimize.
            maxfev: optional, integer
                Parameter for scipy.optimize.minimize.
            maxiter: optional, integer
                Parameter for scipy.optimize.minimize.
        Returns:
            Tuple of (tuple of parameters, log-mpl, success(boolean),
                return value of scipy.optimize.minimize).
        """
        self.fn_dist = distribution
        if variant is not 'ra' and variant is not 'tr':
            print("Warning: variant is not 'ra' nor 'tr'; 'ra' is assumed.")
            self.wnus = self.handle_ties_ra()
        elif variant == 'ra':
            self.wnus = self.handle_ties_ra()
        elif variant == 'tr':
            self.wnus = self.handle_ties_tr()
        self.fn_ipf = ipf if ipf is not None else self.ipf_zero
        self.k_ipf = k_ipf if ipf is not None else 0.0
        self.r_ipf = r_ipf if ipf is not None else 0.0
        p = np.copy(pars_to_fit)
        epsilon = 1.0
        res = None
        while epsilon > 1e-6:
            res = minimize(
                self.nlps, p,
                options={'maxfev': maxfev, 'maxiter': maxiter,
                         'xatol': xatol, 'fatol': fatol
                         },
                method=method,
            )
            self.k_ipf *= self.r_ipf
            p = np.array(res.x)
            epsilon = self.fn_ipf(self.x, p) * self.k_ipf
        return res.x, - self.nlps(res.x), res.success, res

    @staticmethod
    def ipf_zero(x, p):
        _ = (x, p)
        return 0.0

    ##########################################################################
    #  Nagated Log Likelihood Function
    ##########################################################################

    def nll(self, p):
        self.dens = self.fn_dens(self.x, p)
        return - np.sum(np.log(self.dens)) + self.fn_ipf(self.x, p) * \
            self.k_ipf

    ##########################################################################
    #  Negated Log Product of Spacings Function
    ##########################################################################

    def nlps(self, p):
        self.dist[1: self.n_unq + 1] = self.fn_dist(self.unq, p)
        self.spacings = np.diff(self.dist)
        self.log_spacings = np.log(self.spacings)
        return - np.dot(self.wnus, self.log_spacings) + \
            self.fn_ipf(self.x, p) * self.k_ipf

##############################################################################
#
#  Wrapper functions
#
##############################################################################


def carmps(x, cdf, p_0):
    """

    Args:
        x: array-like, float
            sample
        cdf: function
        p_0 ():

    Returns:

    """


    x = np.array(x)
    x.sort()
    dat = Observation(x)
    res = dat.find_mpse(cdf, p_0, variant='tr')
    return res[0]


def jmmps(x, cdf, p_0):
    """Returns the estimator by the modified MPS
    by Jiang (2014).

    :param x: numerical vector
        Sample of random variables.
    :param cdf: function
        Cumulative distribution function which accept
        p_0 as the parameter vector.
    :param p_0: numerical vector
        Parameter vector, initial guess.
    :return:
        The vector of JMMPS estimators.
    """
    x = np.array(x)
    dat = Observation(x)
    res = dat.find_mpse(cdf, p_0, variant='ra')
    return res[0]

##############################################################################
#
#  Example
#  Generalized extreme value parameter estimation.
#
##############################################################################


def gev_cdf(x, p):
    y = np.array(x)
    return np.exp(- (1.0 + p[2] * (y - p[0]) / p[1]) ** (- 1.0 / p[2]))


def ipf_gev(x, p):
    ipfcis = p[1] + p[2] * x - p[2] * p[0]
    invcis = np.divide(1.0, ipfcis)
    return np.sum(invcis)


def gev_mpse(x, variant='ra'):

    """Return the MPS estimator of the parameters of GEV distribution.

    Args:
        x: array like,
            Data.
        variant: optional, string
            Type of the MPS, either 'ra' or 'tr,' default is 'ra.'

    Returns:
        tupple of float: (mu, sigma, xi)
    """
    inst = Observation(x)
    p = np.array([np.min(x), 1.0, 1.0])
    res, lps, success, _ = inst.find_mpse(gev_cdf, p, variant=variant)
    return res


def main():

    x = [-0.3955, -0.3948, -0.3913, -0.3161, -0.1657, 0.3129, 0.3386, 0.5979,
         1.4713, 1.8779, 1.9742, 2.0540, 2.6206, 4.9880, 10.3371]

    print('gev_mpse, ra = ', gev_mpse(x))
    print('gev_mpse, tr = ', gev_mpse(x, variant='tr'))


if __name__ == '__main__':
    main()


"""
Returns the MPS estimator for traditional MPS
    by Cheng and Amin (1983) and Raneaby (1984)

    :param x: numerical vector
        Sample of random variables.
    :param cdf: function
        Cumulative distribution function which accept
        p_0 as the parameter vector.
    :param p_0: numerical vector
        Parameter vector, initial guess.
    :return:
        The vector of MPS estimators.
"""
