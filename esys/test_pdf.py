import numpy as np
from numpy.testing import assert_almost_equal
from post_velocity_pdf import data_to_pdf, box_to_pdf
from scipy.integrate import simps
from scipy.stats.distributions import norm, uniform


N = 1000
L = 1
vmax = 0.5
np.random.seed(1)
pos = 2 * L * (np.random.rand(N, 3) - 0.5)

sigma = 0.1
vel = sigma * np.random.randn(N, 1)
data = np.concatenate((pos, vel), axis=1)


def relative_L2_error(f, ftrue, x):
    L2_error = simps((f - ftrue)**2, x)
    L2_norm = simps(ftrue**2, x)
    return L2_error / L2_norm


def test_box_normal_pdf():
    vx = np.linspace(-vmax, vmax, 1000)
    f = box_to_pdf(data, vx)
    ftrue = norm(0, sigma).pdf(vx)
    error = relative_L2_error(f, ftrue, vx)
    assert_almost_equal(error, 0, decimal=2)


def test_normal_pdf():
    x = np.linspace(-vmax, vmax, 100)
    fapprox = data_to_pdf(data[:, 3], x)
    ftrue = norm(0, sigma).pdf(x)
    error = relative_L2_error(fapprox, ftrue, x)
    assert_almost_equal(error, 0, decimal=2)


def test_uniform():
    np.random.seed(12)
    a = np.sqrt(6) * sigma
    data_uniform = uniform(-a, a).rvs(N)
    x = np.linspace(-vmax, vmax, 100)
    fapprox = data_to_pdf(data_uniform, x)
    ftrue = uniform(-a, a).pdf(x)
    error = relative_L2_error(fapprox, ftrue, x)
    assert_almost_equal(error, 0, decimal=2)


def test_independence():
    """
    This is supposed to check whether f2(x,y) = f1(x)f(y)
    """
    np.random.seed(123)
    N = 50000
    sigma = 0.2
    vmax = 1
    resolution = 0.2
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    data = np.random.multivariate_normal(mean, cov, N)
    x, y = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    coords = np.vstack([item.ravel() for item in [x, y]])
    f2 = data_to_pdf(data, coords)
    f11 = data_to_pdf(data[:, 0], x.flatten())
    f12 = data_to_pdf(data[:, 1], y.flatten())

    error = np.sum(np.abs((f2 - f11 * f12)) * resolution**2) / (2 * vmax)**2
    assert_almost_equal(error, 0, decimal=1)
