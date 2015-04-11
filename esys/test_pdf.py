import numpy as np
from numpy.testing import assert_almost_equal
from post_velocity_pdf import data_to_pdf
from post_velocity_pdf import save_contour_plot


def test_normal():
    np.random.seed(1)


def test_uniform():
    np.random.seed(1)


def test_independence():
    """
    This is supposed to check whether f2(x,y) = f1(x)f(y)
    """
    np.random.seed(1234)
    N = 50000
    sigma = 0.1
    vmax = 1
    resolution = 0.05
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    data = np.random.multivariate_normal(mean, cov, N)
    x, y = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    coords = np.vstack([item.ravel() for item in [x, y]])
    f2 = data_to_pdf(data, coords)
    f11 = data_to_pdf(data[:, 0], x.flatten())
    f12 = data_to_pdf(data[:, 1], y.flatten())

    cumulative_error = np.sum((f2 - f11 * f12) * resolution**2)
    assert_almost_equal(cumulative_error, 0, decimal=3)
