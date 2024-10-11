import numpy as np
from numpy.typing import NDArray

def PID_digital(kp: float, Ti: float, Td: float, Ts: float, e: NDArray[np.float64], k: int) -> float:
    return kp * (e[k] + Ts/Ti * sum(e[:k]) + Td/Ts * (e[k] - e[k-1]))
