import numpy as np

class Sensor:
    def __init__(self, n_sampl, tau_y, c) -> None:
        self.tau_y = tau_y
        self.y = np.empty((4, n_sampl))
        self.C = np.array(
        [[c[0], 0, 0, 0],
        [0, c[1], 0, 0],
        [0, 0, c[2], 0],
        [0, 0, 0, c[3]]])

    def measure(self, h, t) -> None:
        # można dodać jeszcze szum do pomiaru
        self.y[:, [t]] = self.C @ h[:, [t-self.tau_y]] # + np.random.randn(4,1)*e_sigma*active_noise