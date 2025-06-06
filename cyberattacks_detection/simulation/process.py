import numpy as np
from numpy.typing import NDArray

g = 981  # cm/s^2, gravitational acceleration in centimeters per second squared

class FourTankProcess:
    """
    Class representing the four-tank process.

    Attributes
    ----------
    Ts : float
        Sampling time in seconds.
    a : NDArray
        Array of cross-sectional areas of the outflow openings.
    S : NDArray
        Array of the tank volumes.
    gamma_a : float
        Flow rate constant for tank A.
    gamma_b : float
        Flow rate constant for tank B.
    h_max : NDArray
        Maximum tank levels.
    h_min : NDArray
        Minimum tank levels.
    tau_u : int, optional
        Time delay for the control input, default is 0.
    tau_y : int, optional
        Time delay for the output, default is 0.
    clip : bool, optional
        Whether to clip tank levels within bounds of h_min and h_max, default is False.
    qd : NDArray, optional
        Array of disturbances, default is an array with a single 0.

    Methods
    -------
    __setattr__(name, value)
        Custom setter for updating parameters and recalculating matrices.
    calc_p_B_matrix()
        Calculates the p and B matrices based on the system parameters.
    set_init_state(h0)
        Initializes the tank levels with given initial values.
    update_state(q, t)
        Updates the state of the system (tank levels) based on the inputs and disturbances.
    """
    
    def __init__(self, n_sampl: int, Ts: float, a: NDArray, S: NDArray, gamma_a: float, gamma_b: float, 
                 h_max: NDArray, h_min: NDArray, tau_y: int = 0, tau_u: int = 0, clip: bool = False, 
                 qd: NDArray = np.array([0])) -> None:
        """
        Initializes the FourTankProcess object with the provided parameters.

        Parameters
        ----------
        n_sampl : int
            Number of samples (time steps) for the simulation.
        Ts : float
            Sampling time in seconds.
        a : NDArray
            Cross-sectional area of the outflow openings for each tank.
        S : NDArray
            Volume of each tank.
        gamma_a : float
            Flow rate constant for tank A.
        gamma_b : float
            Flow rate constant for tank B.
        h_max : NDArray
            Maximum height values for each tank.
        h_min : NDArray
            Minimum height values for each tank.
        tau_y : int, optional
            Time delay for the output, default is 0.
        tau_u : int, optional
            Time delay for the control input, default is 0.
        clip : bool, optional
            Whether to clip the tank levels within bounds of h_min and h_max, default is False.
        qd : NDArray, optional
            Array of disturbances, default is an array with a single 0.
        """
        self.Ts = Ts
        self.a = a
        self.S = S
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.h_max = h_max
        self.h_min = h_min
        self.tau_u = tau_u
        self.tau_y = tau_y
        self.clip = clip
        self.qd = qd

        self.A = np.array(
        [[-1, 0, 1, 0],
         [0, -1, 0, 1],
         [0, 0, -1, 0],
         [0, 0, 0, -1]])  # Matrix A representing the flow rates between tanks
         
        self.h = np.empty((4, n_sampl))  # Tank heights, initialized as an empty array
    
    def __setattr__(self, name: str, value: any) -> None:
        """
        Custom setter for updating parameters and recalculating matrices when needed.

        This method is called automatically when an attribute is set. It checks if important parameters have changed
        and recalculates the p and B matrices if necessary.

        Parameters
        ----------
        name : str
            The name of the attribute being set.
        value : any
            The value being assigned to the attribute.
        """
        # Use default behavior for setting attributes
        object.__setattr__(self, name, value)
        
        # Check if key parameters were updated and recalculate matrices if necessary
        if name in {"a", "S", "gamma_a", "gamma_b"}:
            if hasattr(self, "S") and hasattr(self, "a") and hasattr(self, "gamma_a") and hasattr(self, "gamma_b"):
                self.calc_p_B_matrix()

    def calc_p_B_matrix(self) -> None:
        """
        Calculates the p and B matrices, which are used in the state update equations.

        This method is called automatically when the key parameters (a, S, gamma_a, gamma_b) are updated.
        The p matrix represents the flow rates, and the B matrix represents the tank input-output relationships.
        """
        print("calc_p_B_matrix")
        # p matrix represents the flow rates, calculated using the formula
        self.p = np.reshape(self.a / self.S * np.sqrt(2 * g), (4, -1))
        
        # B matrix defines the relationship between the control inputs and the tanks
        self.B = np.array(
            [[self.gamma_a / self.S[0], 0],
             [0, self.gamma_b / self.S[1]],
             [0, (1 - self.gamma_b) / self.S[2]],
             [(1 - self.gamma_a) / self.S[3], 0]])

    def set_init_state(self, h0: NDArray) -> None:
        """
        Initializes the tank heights at the beginning of the simulation.

        Parameters
        ----------
        h0 : NDArray
            The initial tank heights for each of the four tanks.
        """
        for i in range(len(h0)):
            self.h[i, 0:max(self.tau_u, self.tau_y, 4)] = h0[i]

    def update_state(self, q: NDArray, t: int) -> None:
        """
        Updates the state (tank heights) based on the inputs and disturbances.

        This method calculates the new heights of the tanks at time step `t` based on the previous heights,
        the input flow rates, and any disturbances. It also clips the tank heights if necessary.

        Parameters
        ----------
        q : NDArray
            The flow rates into the tanks, including control inputs.
        t : int
            The current time step in the simulation.
        """
        # Update tank heights using the state-space equations
        self.h[:, [t]] = self.h[:, [t-1]] + self.Ts * (
            self.A @ (self.p * np.sqrt(self.h[:, [t-1]])) + 
            self.B @ q[:, [t-1-self.tau_u]] + 
            self.qd[:, [t-1]]
        )
        
        # Ensure tank heights are non-negative
        self.h[:, t] = np.clip(self.h[:, t], 0, None)  # Prevent negative heights
        
        # If the 'clip' flag is set, clip the heights within the max and min bounds
        if self.clip:
            self.h[:, t] = np.clip(self.h[:, t], self.h_min, self.h_max)
