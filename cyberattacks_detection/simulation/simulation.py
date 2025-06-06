import numpy as np
from .controller import PIDController
from .process import FourTankProcess
from .sensor import Sensor
from .cyber_attack import CyberAttack
from numpy.typing import NDArray
import warnings
import keras
from ..models import reverse_min_max_scale

g = 981  # cm/s^2, gravitational acceleration in centimeters per second squared

class Simulation:
    """
    A class that simulates a four-tank system, including sensor measurements, control (PID), 
    and the possibility of cyberattacks on the system.

    Attributes
    ----------
    qa_max : float
        Maximum flow rate for tank A.
    qb_max : float
        Maximum flow rate for tank B.
    Ts : float
        Sampling time.
    tau_u : int
        Time delay for the control input.
    n_sampl : int
        Number of samples in the simulation.
    tau_y : int
        Time delay for the sensor output.
    process : FourTankProcess
        The four-tank process object representing the system's state and dynamics.
    sensor : Sensor
        The sensor object that measures the tank levels.
    cyberattack_detector : object or None
        The cyberattack detection system, or None if detection is not enabled.
    pid_a : PIDController
        PID controller for tank A.
    pid_b : PIDController
        PID controller for tank B.

    Methods
    -------
    _set_init_state(h0, attack_scenario, num_tank, **kwargs)
        Initializes the state of the simulation.
    _calc_q(t)
        Calculates the control inputs (flow rates) for the system.
    _prepare_recurrent_model_inputs(k, model, h_idx, recursion_mode=False)
        Prepares inputs for a recurrent model.
    _prepare_model_inputs(k, model, recursion_mode=False)
        Prepares inputs for the predictive model.
    run(h0, close_loop=True, model_list=None, recursion_mode=False, attack_scenario=None, 
        attack_time=None, num_tank=None, variability=False, param_name=None, param_value=None, 
        time_change=None, **kwargs)
        Runs the simulation for the given parameters.
    """
    
    def __init__(self,
                h_max: float, 
                h_min: float, 
                qa_max: float,
                qb_max: float,
                gamma_a: float, 
                gamma_b: float,
                S: NDArray[np.float64], 
                a: NDArray[np.float64], 
                c: NDArray[np.float64], 
                T: int, 
                Ts: int,
                kp: float,
                Ti: float,
                Td: float, 
                tau_u: int=0, 
                tau_y: int=0,
                qd: NDArray[np.float64]=np.array([0]),
                clip=False,
                cyberattack_detector=None,
                ) -> None:
        """
        Initializes the Simulation object with the given parameters.

        Parameters
        ----------
        h_max : float
            Maximum height for each tank.
        h_min : float
            Minimum height for each tank.
        qa_max : float
            Maximum flow rate for tank A.
        qb_max : float
            Maximum flow rate for tank B.
        gamma_a : float
            Flow rate constant for tank A.
        gamma_b : float
            Flow rate constant for tank B.
        S : NDArray[np.float64]
            Array representing the tank volumes.
        a : NDArray[np.float64]
            Array of cross-sectional areas of the outflow openings for each tank.
        c : NDArray[np.float64]
            Sensor coefficients.
        T : int
            Total number of time steps.
        Ts : int
            Sampling time.
        kp : float
            Proportional gain for the PID controller.
        Ti : float
            Integral time for the PID controller.
        Td : float
            Derivative time for the PID controller.
        tau_u : int, optional
            Time delay for the control input (default is 0).
        tau_y : int, optional
            Time delay for the output (default is 0).
        qd : NDArray[np.float64], optional
            Array of disturbances (default is an array with a single 0).
        clip : bool, optional
            Whether to clip the tank levels (default is False).
        cyberattack_detector : object or None, optional
            The cyberattack detection system (default is None).
        """
        self.qa_max = qa_max
        self.qb_max = qb_max
        self.Ts = Ts
        self.tau_u = tau_u
        self.n_sampl = T // Ts + 1
        self.tau_u = tau_u
        self.tau_y = tau_y

        # Initialize the process (FourTankProcess), sensor (Sensor), and cyberattack detector (if provided)
        self.process = FourTankProcess(self.n_sampl, self.Ts, a, S, gamma_a, gamma_b, h_max, h_min, self.tau_y, self.tau_u, clip, qd)
        self.sensor = Sensor(self.n_sampl, self.tau_y, c)
        self.cyberattack_detector = cyberattack_detector

        self.F = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0]])

        # Initialize PID controllers for both tanks
        self.pid_a = PIDController(kp, Ti, Td, self.Ts, self.n_sampl)
        self.pid_b = PIDController(kp, Ti, Td, self.Ts, self.n_sampl)

    def _set_init_state(self, h0, attack_scenario, num_tank, **kwargs) -> None:
        """
        Initializes the state of the simulation with given initial tank heights.

        Parameters
        ----------
        h0 : NDArray
            Initial heights of the tanks.
        attack_scenario : int or None
            The attack scenario to simulate, or None if no attack is applied.
        num_tank : int
            The tank number for the attack scenario.
        """
        # Set initial tank levels
        self.process.set_init_state(h0)
        self.z = self.F @ self.process.h
        self.sensor.set_init_state(self.process.h[:, :len(h0)])

        # Initialize the cyberattack if specified
        if attack_scenario is not None:
            self.cyberattack = CyberAttack(self.process, self.sensor, attack_scenario, num_tank, **kwargs)

    def _calc_q(self, t: int) -> None:
        """
        Calculates the control inputs (flow rates) for the system.

        Parameters
        ----------
        t : int
            The current time step.
        """
        # Calculate the error (difference between setpoint and measured value)
        self.e[:, [t-1]] = self.SP_h[:, [t-1]] - self.z[:, [t-1]]
        
        # Update the flow rates using PID controllers
        self.qa += self.pid_a.calc_dCV(self.SP_h[1, :], self.z[1, :], t-1)
        self.qb += self.pid_b.calc_dCV(self.SP_h[0, :], self.z[0, :], t-1)

        # Ensure that flow rates are within limits
        self.qa = min(self.qa, self.qa_max)
        self.qa = max(self.qa, 0)
        self.qb = min(self.qb, self.qb_max)
        self.qb = max(self.qb, 0)
        self.q[:, [t-1]] = np.vstack((self.qa, self.qb))

    def _prepare_recurrent_model_inputs(self, k: int, model, h_idx: int, recursion_mode: bool = False) -> NDArray:
        """
        Prepares the inputs for the recurrent model at a specific time step.

        Parameters
        ----------
        k : int
            The current time step.
        model : keras.Model
            The recurrent model to use for prediction.
        h_idx : int
            The index of the tank for which to prepare the inputs.
        recursion_mode : bool, optional
            Whether the model is used in a recurrent mode (default is False).

        Returns
        -------
        NDArray
            The input data prepared for the model.
        """
        inputs = []
        _, time_steps, num_features = model.input_shape
        if num_features == 6: # model in state space - only time_steps = 1
            inputs = self.q[:, [k-1]]
            if recursion_mode:
                h_input = self.h_model[:, [k-1]]
                inputs = np.concatenate((inputs, h_input), axis=0)
            else:
                h_input = self.sensor.y[:, [k-1]]
                inputs = np.concatenate((inputs, h_input), axis=0)
        else:
            inputs = np.transpose(self.q[:, k-time_steps:k-1+1])
            if recursion_mode:
                h_input = np.reshape(self.h_model[h_idx, k-time_steps:k-1+1], (time_steps, -1))
                inputs = np.concatenate((inputs, h_input), axis=1)
            else:
                h_input = np.reshape(self.sensor.y[h_idx, k-time_steps:k-1+1], (time_steps, -1))
                inputs = np.concatenate((inputs, h_input), axis=1)

        inputs = np.reshape(np.array(inputs), (1, time_steps, -1))
        return inputs
    
    def _prepare_model_inputs(self, k, model, recursion_mode=False):
        """
        Prepare inputs for the predictive model based on model_features and history.
        Returns:
            Array of inputs for the model.
        """
        inputs = []
        for feature in model.feature_names_in_:
            if feature == "q_A(k-1)":
                inputs.append(self.q[0, k-1])
            elif feature == "q_B(k-1)":
                inputs.append(self.q[1, k-1])
            elif feature == "x2(k-1)":
                if recursion_mode:
                    inputs.append(self.h_model[1, k-1])
                else:
                    inputs.append(self.sensor.y[1, k-1])
            elif feature == "x3(k-1)":
                if recursion_mode:
                    inputs.append(self.h_model[2, k-1])
                else:
                    inputs.append(self.sensor.y[2, k-1])
            elif feature == "x4(k-1)":
                if recursion_mode:
                    inputs.append(self.h_model[3, k-1])
                else:
                    inputs.append(self.sensor.y[3, k-1])
            elif feature == "x1(k-1)":
                if recursion_mode:
                    inputs.append(self.h_model[0, k-1])
                else:
                    inputs.append(self.sensor.y[0, k-1])
            elif feature == "q_A(k-2)":
                inputs.append(self.q[0, k-2])
            elif feature == "q_B(k-2)":
                inputs.append(self.q[1, k-2])
            elif feature == "x1(k-2)":
                if recursion_mode:
                    inputs.append(self.h_model[0, k-2])
                else:
                    inputs.append(self.sensor.y[0, k-2])
            elif feature == "x2(k-2)":
                if recursion_mode:
                    inputs.append(self.h_model[1, k-2])
                else:
                    inputs.append(self.sensor.y[1, k-2])
            elif feature == "x3(k-2)":
                if recursion_mode:
                    inputs.append(self.h_model[2, k-2])
                else:                    
                    inputs.append(self.sensor.y[2, k-2])
            elif feature == "x4(k-2)":
                if recursion_mode:
                    inputs.append(self.h_model[3, k-2])
                else:
                    inputs.append(self.sensor.y[3, k-2])
            elif feature == "q_A(k-3)":
                inputs.append(self.q[0, k-3])
            elif feature == "q_B(k-3)":
                inputs.append(self.q[1, k-3])
            elif feature == "x1(k-3)":
                if recursion_mode:
                    inputs.append(self.h_model[0, k-3])
                else:
                    inputs.append(self.sensor.y[0, k-3])
            elif feature == "x2(k-3)":
                if recursion_mode:
                    inputs.append(self.h_model[1, k-3])
                else:
                    inputs.append(self.sensor.y[1, k-3])
            elif feature == "x3(k-3)":
                if recursion_mode:
                    inputs.append(self.h_model[2, k-3])
                else:
                    inputs.append(self.sensor.y[2, k-3])
            elif feature == "x4(k-3)":
                if recursion_mode:
                    inputs.append(self.h_model[3, k-3])
                else:
                    inputs.append(self.sensor.y[3, k-3])
            elif feature == "q_A(k-4)":
                inputs.append(self.q[0, k-4])
            elif feature == "q_B(k-4)":
                inputs.append(self.q[1, k-4])
            elif feature == "x1(k-4)":
                if recursion_mode:
                    inputs.append(self.h_model[0, k-4])
                else:
                    inputs.append(self.sensor.y[0, k-4])
            elif feature == "x2(k-4)":
                if recursion_mode:
                    inputs.append(self.h_model[1, k-4])
                else:
                    inputs.append(self.sensor.y[1, k-4])
            elif feature == "x3(k-4)":
                if recursion_mode:
                    inputs.append(self.h_model[2, k-4])
                else:
                    inputs.append(self.sensor.y[2, k-4])
            elif feature == "x4(k-4)":
                if recursion_mode:
                    inputs.append(self.h_model[3, k-4])
                else:
                    inputs.append(self.sensor.y[3, k-4])
            else:
                raise ValueError(f"Unknown feature: {feature}")
        return np.array([inputs])

    def run(self,
            h0,
            close_loop=True,
            model_list=None,
            recursion_mode=False,
            attack_scenario=None,
            attack_time=None,
            num_tank=None,
            variability=False,
            param_name=None,
            param_value=None,
            time_change=None,
            **kwargs) -> tuple:
        """
        Runs the simulation over the specified time steps and applies the control system.

        Parameters
        ----------
        h0 : NDArray
            Initial heights for the tanks.
        close_loop : bool, optional
            Whether to use closed-loop control (default is True).
        model_list : list of models, optional
            A list of predictive models to use for simulation (default is None).
        recursion_mode : bool, optional
            Whether to use recursion in the model (default is False).
        attack_scenario : int or None, optional
            The attack scenario to simulate (default is None).
        attack_time : int or None, optional
            The time step at which the attack occurs (default is None).
        num_tank : int or None, optional
            The tank number affected by the attack (default is None).
        variability : bool, optional
            Whether to introduce variability in the parameters (default is False).
        param_name : str, optional
            The parameter name to vary (default is None).
        param_value : float, optional
            The value to assign to the parameter (default is None).
        time_change : int or None, optional
            The time step at which the parameter change occurs (default is None).

        Returns
        -------
        tuple
            The simulated tank heights, sensor measurements, output, flow rates, errors, and model predictions.
        """
        self._set_init_state(h0, attack_scenario, num_tank, **kwargs)

        if close_loop:
            self.SP_h = kwargs['SP_h']
            self.qa = kwargs['qa0']
            self.qb = kwargs['qb0']
            self.e = self.SP_h - self.z
            self.q = np.ones((2, self.n_sampl)) * [[self.qa], [self.qb]]
        else:
            self.q = kwargs['q']
            self.e = None
        if model_list is not None:
            self.h_model = self.process.h[:len(model_list), :].copy()
        else:
            self.h_model = self.process.h.copy()

        for t in range(max(self.tau_u, self.tau_y, 4), self.n_sampl):
            if t % 500 == 0:
                print(f"Krok: {t}/{self.n_sampl}")
            if close_loop:
                self._calc_q(t)
            if (variability) and (t == time_change):
                try:
                    self.process.__setattr__(param_name, param_value)
                    print(f"Zmieniono wartość {param_name} na {self.process.__getattribute__(param_name)}")
                except:
                    print("Incorrect process parameter name")
            self.process.update_state(self.q, t)
            self.sensor.measure(self.process.h, t)
            if (attack_scenario is not None) and (t >= attack_time):
                self.cyberattack.apply_attack(t)
            self.z[:, [t]] = self.F @ self.sensor.y[:, [t]]

            if model_list is not None:
                h_model_t = []
                for i, model in enumerate(model_list):
                    if isinstance(model, keras.Model):
                        inputs = self._prepare_recurrent_model_inputs(t, model, i, recursion_mode)
                        with warnings.catch_warnings():
                            y_pred_sc = model.predict(inputs, verbose=0)[0][0]
                            y_pred = reverse_min_max_scale(y_pred_sc, self.process.h_min[i][0], self.process.h_max[i][0])
                            h_model_t.append(y_pred)
                    else:
                        inputs = self._prepare_model_inputs(t, model, recursion_mode)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names")
                            h_model_t.append(model.predict(inputs)[0])
                self.h_model[:, [t]] = np.reshape(np.array(h_model_t), (-1, 1))

            if self.cyberattack_detector is not None:
                self.cyberattack_detector.detect(self.sensor.y[:len(self.h_model)], self.h_model, t)

        self.q[:, [self.n_sampl - 1]] = np.nan
        if close_loop:
            self.e[:, [self.n_sampl - 1]] = None
        if self.cyberattack_detector is not None:
            attack_signal = np.array(self.cyberattack_detector.attack_signal)
            expanded_attack_signal = np.empty((self.n_sampl, attack_signal.shape[1]), dtype=object)
            expanded_attack_signal.fill(None)
            expanded_attack_signal[-attack_signal.shape[0]:, :] = attack_signal
        else:
            expanded_attack_signal = None

        return self.process.h, self.sensor.y, self.z, self.q, self.e, self.h_model, expanded_attack_signal
