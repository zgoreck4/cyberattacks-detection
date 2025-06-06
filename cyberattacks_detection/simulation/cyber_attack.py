from __future__ import annotations  # for forward type references as strings
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type hints, not at runtime
    from .process import FourTankProcess
    from .sensor import Sensor


class CyberAttack:
    """
    Implements different cyberattack scenarios on sensor measurements
    in a FourTankProcess system.

    Parameters
    ----------
    process : FourTankProcess
        The process model being attacked.
    sensor : Sensor
        Sensor object providing measurement access.
    attack_scenario : int
        Index of attack type:
        0 = freeze sensor value,
        1 = gradual increase,
        2 = delayed measurement,
        3 = delayed measurement + gradual increase.
    num_tank_list : list[int]
        Indices of tanks/sensors to attack.
    attack_value : float, optional
        Initial attack increment value (for gradual increase), by default None.
    tau_y_ca : int, optional
        Delay steps for delayed attack, by default 0.
    kwargs
        Additional arguments (ignored).
    """

    def __init__(
        self,
        process: "FourTankProcess",
        sensor: "Sensor",
        attack_scenario: int,
        num_tank_list: list[int],
        attack_value: float = None,
        tau_y_ca: int = 0,
        **kwargs,
    ) -> None:
        self.process = process
        self.sensor = sensor
        self.num_tank_list = num_tank_list

        # Assign attack method based on attack_scenario
        if attack_scenario == 0:
            self.attack_scenario = self.froze
        elif attack_scenario == 1:
            self.attack_scenario = self.grad_increase
            self.attack_value = attack_value
            self.step_value = attack_value  # incremental step for each call
        elif attack_scenario == 2:
            self.tau_y_ca = tau_y_ca
            self.attack_scenario = self.add_delay
        elif attack_scenario == 3:
            self.tau_y_ca = tau_y_ca
            self.attack_value = attack_value
            self.step_value = attack_value
            self.attack_scenario = self.add_delay_grad_increase

    def froze(self, k: int) -> None:
        """Freeze sensor output at time k to its previous value (k-1)."""
        self.sensor.y[self.num_tank_list, k] = self.sensor.y[self.num_tank_list, k - 1]

    def grad_increase(self, k: int) -> None:
        """
        Gradually increase sensor output at time k by a step value.
        Attack value is incremented after each call.
        """
        self.sensor.y[self.num_tank_list, k] += self.attack_value
        self.attack_value += self.step_value

    def add_delay(self, k: int) -> None:
        """Replace sensor output at time k with delayed true process value."""
        self.sensor.y[self.num_tank_list, k] = self.process.h[self.num_tank_list, k - self.tau_y_ca]

    def add_delay_grad_increase(self, k: int) -> None:
        """
        Apply delayed measurement and gradual increase attacks combined.
        Calls add_delay first, then grad_increase.
        """
        self.add_delay(k)
        self.grad_increase(k)

    def apply_attack(self, k: int) -> None:
        """
        Apply the selected attack scenario at time step k.

        This method uses polymorphism to call the appropriate attack method
        without the caller needing to know which one it is.
        """
        self.attack_scenario(k)