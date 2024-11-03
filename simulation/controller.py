import numpy as np
from numpy.typing import NDArray

def PID_digital(kp: float, Ti: float, Td: float, Ts: float, e: NDArray[np.float64], k: int) -> float:
    return kp * (e[k] + Ts/Ti * sum(e[:k]) + Td/Ts * (e[k] - e[k-1]))

def PID_digital_inc(kp: float, Ti: float, Td: float, Ts: float, e: NDArray[np.float64], k: int) -> float:
    return kp * (e[k] - e[k-1] + Ts/Ti * e[k-1] + Td/Ts * (e[k] - 2*e[k-1] + e[k-2]))