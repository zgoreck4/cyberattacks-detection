import numpy as np
from .controller import PIDController
from .process import FourTankProcess
from .sensor import Sensor
from .cyber_attack import CyberAttack
from numpy.typing import NDArray
import warnings

g = 981 # cm/s^2

class Simulation:
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
                # noise_sigma: float=0.1, 
                # e_sigma: float=0.005,
                clip=False,
                cyberattack_detector=None,
                ) -> None:
        
        self.qa_max = qa_max
        self.qb_max = qb_max
        self.Ts = Ts
        self.tau_u = tau_u
        self.n_sampl = T//Ts+1
        self.tau_u = tau_u
        self.tau_y = tau_y

        self.process = FourTankProcess(self.n_sampl, self.Ts, a, S, gamma_a, gamma_b, h_max, h_min, self.tau_y, self.tau_u, clip, qd)
        self.sensor  = Sensor(self.n_sampl, self.tau_y, c)
        self.cyberattack_detector = cyberattack_detector

        self.F = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0]])

        self.pid_a = PIDController(kp, Ti, Td, self.Ts, self.n_sampl)
        self.pid_b = PIDController(kp, Ti, Td, self.Ts, self.n_sampl)

    
    def _set_init_state(self, h0, attack_scenario, num_tank, **kwargs):
        self.process.set_init_state(h0)
        self.z = self.F @ self.process.h
        # self.e = self.SP_h - self.z
        if attack_scenario is not None:
            self.cyberattack = CyberAttack(self.process, self.sensor, attack_scenario, num_tank, **kwargs)

    
    def _calc_q(self, t):
        self.e[:, [t-1]] = self.SP_h[:, [t-1]] - self.z[:, [t-1]]
        # print(e[:, [t-1]])
        # TODO implementacja regulatora dla tau_u
        # TODO implementacja dla szumu, bo może trzeba we wzorze na z użyć y zamiast h
        self.qa += self.pid_a.calc_dCV(self.SP_h[1, :], self.z[1, :], t-1)
        self.qb += self.pid_b.calc_dCV(self.SP_h[0, :], self.z[0, :], t-1)
        # print(f"{qa=:.4f}")
        # print(f"{qb=:.4f}")
        self.qa = min(self.qa, self.qa_max)
        self.qa = max(self.qa, 0)
        self.qb = min(self.qb, self.qb_max)
        self.qb = max(self.qb, 0)
        self.q[:, [t-1]] = np.vstack((self.qa, self.qb))

    
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
            **kwargs):
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

        self.h_model = self.process.h[:len(model_list), :].copy()

        for t in range(max(self.tau_u, self.tau_y, 3), self.n_sampl):
            if close_loop:
                self._calc_q(t)
            self.process.update_state(self.q, t)
            self.sensor.measure(self.process.h, t)
            if (attack_scenario is not None) and (t >= attack_time):
                self.cyberattack.apply_attack(t)
            self.z[:, [t]] = self.F @ self.sensor.y[:, [t]]

            if (model_list is not None) and (t > 6):
                h_model_t = []
                for model in model_list:
                    inputs = self._prepare_model_inputs(t, model, recursion_mode)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="X does not have valid feature names")
                        h_model_t.append(model.predict(inputs)[0])
                self.h_model[:, [t]] = np.array(h_model_t)
            else:
                self.h_model[:, [t]] = self.process.h[:len(model_list), [t]]

            if self.cyberattack_detector is not None:
                self.cyberattack_detector.detect(self.sensor.y[:len(self.h_model)], self.h_model, t)


        self.q[:, [self.n_sampl-1]] = np.nan
        if close_loop:
            self.e[:, [self.n_sampl-1]] = None
        if self.cyberattack_detector is not None:
                attack_signal = np.array(self.cyberattack_detector.attack_signal)
                expanded_attack_signal = np.empty((self.n_sampl, attack_signal.shape[1]), dtype=object)
                expanded_attack_signal.fill(None)
                # Copy the original array data into the new array
                expanded_attack_signal[-attack_signal.shape[0]:, :] = attack_signal
        else:
            expanded_attack_signal = None

        return self.process.h, self.sensor.y, self.z, self.q, self.e, self.h_model, expanded_attack_signal