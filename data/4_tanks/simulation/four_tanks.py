import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from controller import PID_digital

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
    Ts: int, 
    tau_u: int=0, 
    tau_y: int=0,
    active_noise: bool=False,
    qd: NDArray[np.float64]=np.array([0]),
    noise_sigma: float=0.1, 
    e_sigma: float=0.005,
    clip=False
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
    Ts: int
        Sampling time
    tau_u: int=0
        u delay
    tau_y: int=0
        y delay
    active_noise: bool=False
        variable to enable noises
    qd: NDArray[np.float64]=np.array([0])
        unknown flow
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
    n_sampl = T//Ts+1
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
        x[:, [t]] = x[:, [t-1]] + Ts * (A @ (p * np.sqrt(x[:, [t-1]])) + B @ q[:, [t-1-tau_u]] + qd[:, [t-1]])
        x[:, t] = np.clip(x[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if clip:
            x[:, t] = np.clip(x[:, t], x_min, x_max)
        y[:, [t]] = C @ x[:, [t-tau_y]] + np.random.randn(4,1)*e_sigma*active_noise
        z[:, [t]] = F @ x[:, [t]]
        
    return x, y, z


def simulate_close_loop(
    x0: NDArray[np.float64], 
    x_max: float, 
    x_min: float, 
    gamma_a: float, 
    gamma_b: float,
    S: NDArray[np.float64], 
    a: NDArray[np.float64], 
    c: NDArray[np.float64], 
    SP_x: NDArray[np.float64],
    T: int, 
    Ts: int,
    kp: float,
    Ti: float,
    Td: float, 
    tau_u: int=0, 
    tau_y: int=0,
    active_noise: bool=False,
    qd: NDArray[np.float64]=np.array([0]),
    noise_sigma: float=0.1, 
    e_sigma: float=0.005,
    clip=False
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
    Ts: int
        Sampling time
    tau_u: int=0
        u delay
    tau_y: int=0
        y delay
    active_noise: bool=False
        variable to enable noises
    qd: NDArray[np.float64]=np.array([0])
        unknown flow
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
    n_sampl = T//Ts
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
    e = np.zeros((2, n_sampl))

    for t in range(max(tau_u, tau_y, 1), n_sampl):
        # TODO implementacja regulatora dla tau_u
        e[:][t-1] = z[:][t-1] - SP_x[:][t-1]
        qa = PID_digital(kp, Ti, Td, Ts, e[0], t-1)
        qb = PID_digital(kp, Ti, Td, Ts, e[1], t-1)
        q = np.vstack((qa, qb))
        print(q)
        # x[:, [t]] = x[:, [t-1]] + Ts * (A @ (p * np.sqrt(x[:, [t-1]])) + B @ q[:, [t-1-tau_u]] + qd[:, [t-1]])
        x[:, [t]] = x[:, [t-1]] + Ts * (A @ (p * np.sqrt(x[:, [t-1]])) + B @ q + qd[:, [t-1]])
        x[:, t] = np.clip(x[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if clip:
            x[:, t] = np.clip(x[:, t], x_min, x_max)
        y[:, [t]] = C @ x[:, [t-tau_y]] + np.random.randn(4,1)*e_sigma*active_noise
        z[:, [t]] = F @ x[:, [t]]
        
    return x, y, z