from controller import PID_digital
import numpy as np

e = np.ones((2, 10))*1
kp = 5
Ti = 3
Ti = 15
Td = 1.5
k = 3
Ts = 1
print(PID_digital(kp, Ti, Td, Ts, e[0], k))

print(sum(e[0][:k]))