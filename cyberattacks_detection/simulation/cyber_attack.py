from __future__ import annotations  # for types as strings
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints, not at runtime
    from .process import FourTankProcess
    from .sensor import Sensor

class CyberAttack:
    def __init__(self, process: "FourTankProcess", sensor: "Sensor", attack_scenario: int, num_tank_list:list[int], attack_value: float=None, tau_y: int=0, **kwargs) -> None:
        self.process = process
        self.sensor = sensor
        self.num_tank_list = num_tank_list
        if attack_scenario == 0:
            self.attack_scenario = self.froze
        elif attack_scenario == 1:
            self.attack_scenario = self.grad_increase
            self.attack_value = attack_value
            self.step_value = attack_value
        elif attack_scenario == 2:
            self.tau_y = tau_y
            self.attack_scenario = self.add_delay
        elif attack_scenario == 3:
            self.tau_y = tau_y
            self.attack_value = attack_value
            self.step_value = attack_value
            self.attack_scenario = self.add_delay_grad_increase
        
    def froze(self, k: int) -> None:
        self.sensor.y[self.num_tank_list, k] = self.sensor.y[self.num_tank_list, k-1]

    def grad_increase(self, k: int) -> None:
        self.sensor.y[self.num_tank_list, k] = self.sensor.y[self.num_tank_list, k] + self.attack_value
        self.attack_value += self.step_value

    def add_delay(self, k: int) -> None:
        self.sensor.y[self.num_tank_list, k] = self.process.h[self.num_tank_list, k-self.tau_y]

    def add_delay_grad_increase(self, k: int) -> None:
        self.add_delay(k)
        self.grad_increase(k)

    def apply_attack(self, k: int) -> None:
        # TODO: chcemy wywoływać metodę w taki sam sposób nie wiedząc dokładnie jaka to metoda - polimorfizm 
        self.attack_scenario(k)
    
