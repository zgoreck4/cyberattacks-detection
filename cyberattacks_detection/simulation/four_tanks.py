import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.constants import g
from .controller import PIDController

g = 981 # cm/s^2
# g = g*100

def simulate(
    h0: NDArray[np.float64], 
    h_max: float, 
    h_min: float, 
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
    qd: NDArray[np.float64]=np.array([0]),
    noise_sigma: float=0.1, 
    e_sigma: float=0.005,
    clip=False
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    
    """
    Function to simulate four tanks system.

    Parameters:
    -----------
    h0: float
        Initial value for the liquid level in tanks
    h_max: float
        Max level in tanks
    h_min: float
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
    qd: NDArray[np.float64]=np.array([0])
        unknown flow
    e_sigma: float=0.005
        Measurement error

    Returns
    -----------
    h: NDArray[np.float64]
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
    
    h = np.empty((4, n_sampl))
    for i in range(len(h0)):
        h[i, 0:max(tau_u, tau_y, 1)] = h0[i]
    y = np.empty((4, n_sampl))
    z = F @ h

    for t in range(max(tau_u, tau_y, 1), n_sampl):
        # x = h[:, [t-1]] - h_min
        # h[:, [t]] = h[:, [t-1]] + T_s * (A @ (p * np.sqrt(x)) + B @ q[:, [t-1-tau_u]] + qd[:, [t-1]])
        h[:, [t]] = h[:, [t-1]] + T_s * (A @ (p * np.sqrt(h[:, [t-1]])) + B @ q[:, [t-1-tau_u]] + qd[:, [t-1]])
        h[:, t] = np.clip(h[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if clip:
            for i in range(np.shape(h)[0]):
                h[i, t] = np.clip(h[i, t], h_min[i], h_max[i])
        y[:, [t]] = C @ h[:, [t-tau_y]] + np.random.randn(4,1)*e_sigma*active_noise
        z[:, [t]] = F @ h[:, [t]]
        
    return h, y, z


def simulate_close_loop(
    h0: NDArray[np.float64], 
    h_max: float, 
    h_min: float, 
    qa_max: float,
    qb_max: float,
    gamma_a: float, 
    gamma_b: float,
    S: NDArray[np.float64], 
    a: NDArray[np.float64], 
    c: NDArray[np.float64], 
    SP_h: NDArray[np.float64],
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
    h0: float
        Initial value for the liquid level in tanks
    h_max: float
        Max level in tanks
    h_min: float
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
    h: NDArray[np.float64]
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
    
    h = np.empty((4, n_sampl))
    for i in range(len(h0)):
        h[i, 0:max(tau_u, tau_y, 3)] = h0[i]
    y = np.empty((4, n_sampl))
    z = F @ h
    qa = 1630000/3600
    qb = 2000000/3600
    q = np.ones((2, n_sampl)) *[[qa], [qb]]
    e = SP_h - z

    pid_a = PIDController(kp, Ti, Td, Ts, np.shape(SP_h)[1])
    pid_b = PIDController(kp, Ti, Td, Ts, np.shape(SP_h)[1])

    for t in range(max(tau_u, tau_y, 3), n_sampl):
        e[:, [t-1]] = SP_h[:, [t-1]] - z[:, [t-1]]
        # print(e[:, [t-1]])
        # TODO implementacja regulatora dla tau_u
        # TODO implementacja dla szumu, bo może trzeba we wzorze na z użyć y zamiast h
        qa += pid_a.calc_dCV(SP_h[1, :], z[1, :], t-1)
        qb += pid_b.calc_dCV(SP_h[0, :], z[0, :], t-1)
        # print(f"{qa=:.4f}")
        # print(f"{qb=:.4f}")
        qa = min(qa, qa_max)
        qa = max(qa, 0)
        qb = min(qb, qb_max)
        qb = max(qb, 0)
        q[:, [t-1]] = np.vstack((qa, qb))
        h[:, [t]] = h[:, [t-1]] + Ts * (A @ (p * np.sqrt(h[:, [t-1]])) + B @ q[:, [t-1-tau_u]] + qd[:, [t-1]])
        if qd[:, [t]].any()<0:
            h[:, t] = np.clip(h[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if clip:
            h[:, t+1] = np.clip(h[:, t], h_min, h_max)
        y[:, [t]] = C @ h[:, [t-tau_y]] + np.random.randn(4,1)*e_sigma*active_noise
        z[:, [t]] = F @ h[:, [t]]

    q[:, [n_sampl-1]] = None
    e[:, [n_sampl-1]] = None

    return h, y, z, q, e