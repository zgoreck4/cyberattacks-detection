import numpy as np

class Sensor:
    """
    Class representing a sensor measuring the liquid levels in tanks with optional delay and scaling.

    Attributes
    ----------
    tau_y : int
        Measurement delay.
    y : np.ndarray
        Measured tank levels over time, shape (4, n_sampl).
    C : np.ndarray
        Calibration matrix (diagonal) for sensor readings.
    """

    def __init__(self, n_sampl: int, tau_y: int, c: np.ndarray) -> None:
        """
        Initialize sensor object.

        Parameters
        ----------
        n_sampl : int
            Number of simulation samples (time steps).
        tau_y : int
            Measurement delay.
        c : np.ndarray
            Calibration constants for each tank sensor.
        """
        self.tau_y = tau_y
        # Initialize array for sensor measurements (4 tanks x n_sampl steps)
        self.y = np.empty((4, n_sampl))
        # Diagonal calibration matrix scaling each tank's measurement
        self.C = np.array(
            [[c[0], 0, 0, 0],
             [0, c[1], 0, 0],
             [0, 0, c[2], 0],
             [0, 0, 0, c[3]]]
        )

    def set_init_state(self, h: np.ndarray) -> None:
        """
        Initialize sensor measurements from initial tank levels.

        Parameters
        ----------
        h : np.ndarray
            Initial tank levels (4 tanks x time steps).
        """
        # Apply calibration matrix to initial levels to initialize measurements
        self.y[:, :len(h)] = self.C @ h

    def measure(self, h: np.ndarray, t: int) -> None:
        """
        Measure tank levels at time step t, applying calibration and delay.

        Parameters
        ----------
        h : np.ndarray
            Current tank levels (4 tanks x time steps).
        t : int
            Current time step index.
        """
        # Calculate measurement with delay tau_y and calibration matrix
        self.y[:, [t]] = self.C @ h[:, [t - self.tau_y]]