import numpy as np
from numpy.typing import NDArray
from typing import Tuple

g = 981 # cm/s^2

def simulate(
    x0: NDArray[np.float64], 
    x_max: float, 
    x_min: float, 
    gamma_a: float, 
    gamma_b: float,
    S: NDArray[np.float64], 
    a: NDArray[np.float64], 
    c: NDArray[np.float64], 
    q: NDArray[np.float64],
    T: int, 
    T_s: int, 
    tau_u: int=0, 
    tau_y: int=0,
    active_noise: bool=False,
    noise_sigma: float=0.1, 
    e_sigma: float=0.005
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    
    """
    Function to simulate four tanks system.

    Parameters:
    -----------
    x0: float
        Initial value for the liquid level in tanks
    x_max: float
        Max level in tanks
    x_min: float
        Min level in tanks
    gamma_a: float
        Valve A constant
    gamma_b: float
        Valve B constant
    S: NDArray[np.float64]
        Array of cross-sectional area of tanks
    a: NDArray[np.float64]
        Array of cross-sectional area of the outlet hole
    c: NDArray[np.float64]
        Array of calibrated constants
    q: NDArray[np.float64]
        Command flow
    T: int
        Simulation time
    T_s: int
        Sampling time
    tau_u: int=0
        u delay
    tau_y: int=0
        y delay
    active_noise: bool=False
        variable to enable noises
    noise_sigma: float=0.1
        noise sigma
    e_sigma: float=0.005
        Measurement error

    Returns
    -----------
    x: NDArray[np.float64]
        Liquid level in tanks
    y: NDArray[np.float64]
        Measured tank level
    z: NDArray[np.float64]
        Perfomance variable to be controlled (output tank level)
    """
    n_sampl = T//T_s+1
    p = np.reshape(a/S * np.sqrt(2*g), (4, -1))

    A = np.array(
    [[-1, 0, 1, 0],
     [0, -1, 0, 1],
     [0, 0, -1, 0],
     [0, 0, 0, -1]])

    B = np.array(
        [[gamma_a/S[0], 0],
        [0, gamma_b/S[1]],
        [0, (1-gamma_b)/S[2]],
        [(1-gamma_a)/S[3], 0]])

    C = np.array(
        [[c[0], 0, 0, 0],
        [0, c[1], 0, 0],
        [0, 0, c[2], 0],
        [0, 0, 0, c[3]]])

    F = np.array(
        [[1, 0, 0, 0],
        [0, 1, 0, 0]])
    
    x = np.empty((4, n_sampl))
    for i in range(len(x0)):
        x[i, 0:max(tau_u, tau_y, 1)] = x0[i]
    y = np.empty((4, n_sampl))
    z = F @ x

    for t in range(max(tau_u, tau_y, 1), n_sampl):
        x[:, [t]] = x[:, [t-1]] + T_s * (A @ p * np.sqrt(x[:, [t-1]]) + B @ q[:, [t-1-tau_u]] + np.random.randn(4,1)*noise_sigma*active_noise)
        x[:, t] = np.clip(x[:, t], x_min, x_max)
        y[:, [t]] = C @ x[:, [t-tau_y]] + np.random.randn(4,1)*e_sigma*active_noise
        z[:, [t]] = F @ x[:, [t]]
        
    return x, y, z