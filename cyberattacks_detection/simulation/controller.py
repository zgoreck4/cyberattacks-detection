import numpy as np
from numpy.typing import NDArray


class PIDController:

    def __init__(self,
                kp: float,
                Ti: float,
                Td: float,
                Ts: float,
                lenght: int) -> None:
        self.kp = kp
        self.Ti = Ti
        self.Td = Td
        self.Ts = Ts
        self.lenght = lenght
        self.e = np.zeros([self.lenght])

    def calc_CV(self, SP: NDArray[np.float64], z: NDArray[np.float64], k: int) -> float:
        # TODO: zastanowić się czy z będzie tylko miało x1, x2 czy wszystkie x
        self.e[k] = SP[k] - z[k]
        return self.kp * (self.e[k] + self.Ts/self.Ti * sum(self.e[:k]) + self.Td/self.Ts * (self.e[k] - self.e[k-1]))

    def calc_dCV(self, SP: NDArray[np.float64], z: NDArray[np.float64], k: int) -> float:
        # incremental version of digital PID
        # TODO: zastanowić się czy z będzie tylko miało x1, x2 czy wszystkie x
        self.e[k] = SP[k] - z[k]
        return self.kp * (self.e[k] + self.Ts/self.Ti * self.e[k-1] + self.Td/self.Ts * (self.e[k] - 2*self.e[k-1] + self.e[k-2]))