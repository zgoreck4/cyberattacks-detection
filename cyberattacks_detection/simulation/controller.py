import numpy as np
from numpy.typing import NDArray

class PIDController:
    """
    Discrete PID controller implementation.

    Parameters
    ----------
    kp : float
        Proportional gain.
    Ti : float
        Integral time constant.
    Td : float
        Derivative time constant.
    Ts : float
        Sampling period.
    n_sampl : int
        Number of samples to store error history.

    Attributes
    ----------
    e : np.ndarray
        Array storing error values for each sample.
    """

    def __init__(
        self,
        kp: float,
        Ti: float,
        Td: float,
        Ts: float,
        n_sampl: int
    ) -> None:
        self.kp = kp
        self.Ti = Ti
        self.Td = Td
        self.Ts = Ts
        self.n_sampl = n_sampl
        self.e = np.zeros([self.n_sampl])  # Stores error history

    def calc_CV(self, SP: NDArray[np.float64], z: NDArray[np.float64], k: int) -> float:
        """
        Calculate control variable (CV) using the classical PID formula.

        Parameters
        ----------
        SP : NDArray[np.float64]
            Set point array.
        z : NDArray[np.float64]
            Process variable measurement array.
        k : int
            Current time step index.

        Returns
        -------
        float
            Control variable output at step k.
        """
        self.e[k] = SP[k] - z[k]  # Calculate error at step k

        # Return PID control signal
        return self.kp * (self.e[k] + self.Ts/self.Ti * sum(self.e[:k]) + self.Td/self.Ts * (self.e[k] - self.e[k-1]))

    def calc_dCV(self, SP: NDArray[np.float64], z: NDArray[np.float64], k: int) -> float:
        """
        Calculate incremental change in control variable (dCV) using digital PID formula.

        Parameters
        ----------
        SP : NDArray[np.float64]
            Set point array.
        z : NDArray[np.float64]
            Process variable measurement array.
        k : int
            Current time step index.

        Returns
        -------
        float
            Incremental control output at step k.
        """
        self.e[k] = SP[k] - z[k]  # Calculate error at step k

        # Retrieve errors from previous steps if available, else 0
        e_k = self.e[k]
        e_k_1 = self.e[k - 1] if k - 1 >= 0 else 0
        e_k_2 = self.e[k - 2] if k - 2 >= 0 else 0

        # Incremental PID implementation (dCV)
        return self.kp * (
            e_k - e_k_1 +                        # Error increment
            (self.Ts / self.Ti) * e_k_1 +       # Integral term
            (self.Td / self.Ts) * (e_k - 2 * e_k_1 + e_k_2)  # Derivative term (discrete second difference)
        )