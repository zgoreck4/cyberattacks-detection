import numpy as np
from numpy.typing import NDArray

g = 981 # cm/s^2

class FourTankProcess:
    def __init__(self, n_sampl, Ts, a, S, gamma_a, gamma_b, h_max, h_min, h0, tau_y=0, tau_u=0, clip=False, qd=np.array([0])) -> None:
        self.Ts = Ts
        self.a = a
        self.S = S
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.h_max = h_max
        self.h_min = h_min
        self.h0 = h0
        self.tau_u = tau_u
        self.clip = clip
        self.qd = qd

        self.p = np.reshape(a/S * np.sqrt(2*g), (4, -1))
        self.A = np.array(
        [[-1, 0, 1, 0],
        [0, -1, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1]])
        self.B = np.array(
            [[gamma_a/S[0], 0],
            [0, gamma_b/S[1]],
            [0, (1-gamma_b)/S[2]],
            [(1-gamma_a)/S[3], 0]])
        
        self.h = np.empty((4, n_sampl))
        for i in range(len(h0)):
            self.h[i, 0:max(tau_u, tau_y, 3)] = h0[i]

    def update_state(self, q, t):
        self.h[:, [t]] = self.h[:, [t-1]] + self.Ts * (self.A @ (self.p * np.sqrt(self.h[:, [t-1]])) + self.B @ q[:, [t-1-self.tau_u]] + self.qd[:, [t-1]])
        # if qd[:, [t]].any()<0:
        self.h[:, t] = np.clip(self.h[:, t], 0, None) # przycinanie gdyby po dodaniu szumu otrzymano ujemną wartość
        if self.clip:
            self.h[:, t+1] = np.clip(self.h[:, t+1], self.h_min, self.h_max)