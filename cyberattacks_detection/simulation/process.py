import numpy as np
from numpy.typing import NDArray

g = 981 # cm/s^2

class FourTankProcess:
    def __init__(self, n_sampl, Ts, a, S, gamma_a, gamma_b, h_max, h_min, tau_y=0, tau_u=0, clip=False, qd=np.array([0])) -> None:
        self.Ts = Ts
        self.a = a
        self.S = S
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        # po inicjalizacji niezbędnych zmiennych automatycznie obliczane są maceirze p i B
        # dzieje się to w tym miejscu
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
        [0, 0, 0, -1]])      
        self.h = np.empty((4, n_sampl))
    
    def __setattr__(self, name, value):
        # Use default behavior for setting attributes
        object.__setattr__(self, name, value)
        # Check if key attributes were updated
        if name in {"a", "S", "gamma_a", "gamma_b"}:
            if hasattr(self, "S") and hasattr(self, "a") and hasattr(self, "gamma_a") and hasattr(self, "gamma_b"):
                self.calc_p_B_matrix()

    def calc_p_B_matrix(self):
        print("calc_p_B_matrix")
        self.p = np.reshape(self.a/self.S * np.sqrt(2*g), (4, -1))
        self.B = np.array(
            [[self.gamma_a/self.S[0], 0],
            [0, self.gamma_b/self.S[1]],
            [0, (1-self.gamma_b)/self.S[2]],
            [(1-self.gamma_a)/self.S[3], 0]])

    def set_init_state(self, h0):
        for i in range(len(h0)):
            self.h[i, 0:max(self.tau_u, self.tau_y, 4)] = h0[i]

    def update_state(self, q, t):
        self.h[:, [t]] = self.h[:, [t-1]] + self.Ts * (self.A @ (self.p * np.sqrt(self.h[:, [t-1]])) + self.B @ q[:, [t-1-self.tau_u]] + self.qd[:, [t-1]])
        # if qd[:, [t]].any()<0:
        self.h[:, t] = np.clip(self.h[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if self.clip:
            self.h[:, t] = np.clip(self.h[:, t], self.h_min, self.h_max)