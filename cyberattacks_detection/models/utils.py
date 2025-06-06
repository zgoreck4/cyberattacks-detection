import numpy as np


def gaussian_func(x, c, sigma):
    """
    Compute the value of a Gaussian radial basis function.

    Parameters
    ----------
    x : ndarray
        Input vector.
    c : ndarray
        Center of the Gaussian.
    sigma : float
        Spread (standard deviation) of the Gaussian.

    Returns
    -------
    float
        Activation value for the given input.
    """
    distance = np.linalg.norm(x - c, 2)
    return np.exp(-distance**2 / (2 * sigma**2))


def grad_gaussian_func(x, c, sigma):
    """
    Compute the gradient of the Gaussian RBF with respect to sigma.

    Parameters
    ----------
    x : ndarray
        Input vector.
    c : ndarray
        Center of the Gaussian.
    sigma : float
        Spread (standard deviation) of the Gaussian.

    Returns
    -------
    float
        Gradient of the Gaussian function with respect to sigma.
    """
    distance = np.linalg.norm(x - c, 2)
    return gaussian_func(x, c, sigma) * (distance**2) * 1/(sigma**3)

def min_max_scale(s, min_value, max_value):
    """
    Apply min-max normalization to scale data to [0, 1].

    Parameters
    ----------
    s : ndarray
        Input array.
    min_value : ndarray or float
        Minimum value(s) for scaling.
    max_value : ndarray or float
        Maximum value(s) for scaling.

    Returns
    -------
    ndarray
        Scaled array.
    """
    return (s - min_value) / (max_value - min_value)


def reverse_min_max_scale(s, min_value, max_value):
    """
    Invert the min-max normalization to restore original scale.

    Parameters
    ----------
    s : ndarray
        Scaled array.
    min_value : ndarray or float
        Original minimum value(s).
    max_value : ndarray or float
        Original maximum value(s).

    Returns
    -------
    ndarray
        Array transformed back to original scale.
    """
    return s * (max_value - min_value) + min_value