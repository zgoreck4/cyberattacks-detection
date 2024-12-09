import numpy as np

def gaussian_func(x, c, sigma):
    distance = np.linalg.norm(x - c, 2)
    return np.exp(-distance**2 / (2 * sigma**2))

def grad_gaussian_func(x, c, sigma):
    distance = np.linalg.norm(x - c, 2)
    return gaussian_func(x, c, sigma) * (distance**2) * 1/(sigma**3)

def min_max_scale(s, min_value, max_value):
    return (s-min_value)/(max_value - min_value)

def reverse_min_max_scale(s, min_value, max_value):
    return s * (max_value - min_value) + min_value