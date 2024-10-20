import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from four_tanks import simulate

T = 500
T_s = 1
time = np.arange(0, T+1, T_s)
T = max(time)
n_sampl = T//T_s+1
tau_u = 0
tau_y = 0
x0 = [65, 66, 65, 66]
# x0 = [12.4, 1.8, 12.7, 1.4]
# x0 = [12.6, 4.8, 13, 4.9]
active_noise = False # wartość False wyłącza zakłócenia, wartość True włącza
noise_sigma = 0.1
e_sigma = 0.005

# # cm
# qa = 3.15*3.14 #3*3.33
# qb = 3.15*3.29 #3*3.35
# q = np.vstack((np.ones((1, n_sampl))* qa, np.ones((1, n_sampl))* qb))
# x_max = np.inf
# x_min = -np.inf
# gamma_a = 0.43
# gamma_b = 0.34
# S = np.array([28, 32, 28, 32])
# a = np.array([0.071, 0.057, 0.071, 0.057]) # przekrój otworu wylotowego
# c = np.array([0.5, 0.5, 0.5, 0.5])

# cm
qa = 1630000/3600
qb = 2000000/3600
q = np.vstack((np.ones((1, n_sampl))* qa, np.ones((1, n_sampl))* qb))
qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)
x_max = 136
x_min = 20
gamma_a = 0.3
gamma_b = 0.4
S = np.array([60, 60, 60, 60])
a = np.array([1.31, 1.51, 0.927, 0.882]) # przekrój otworu wylotowego
c = np.array([0.5, 0.5, 0.5, 0.5])

x, y, z = simulate(x0, x_max, x_min, gamma_a, gamma_b, S, a, c, q, T, T_s, tau_u, tau_y, active_noise, qd, noise_sigma, e_sigma)

bench_name = '2'
operating_point = ''

# title_end = f" dla {bench_name}. benchmarku i punktu pracy {operating_point}"
title_end = f" dla {bench_name}. benchmarku"

fig2=plt.figure(figsize=(8, 12))
plt.subplot(2, 1, 1)
plt.plot(time, x[0], label='Zbiornik 1')
plt.plot(time, x[1], label='Zbiornik 2')
plt.plot(time, x[2], label='Zbiornik 3')
plt.plot(time, x[3], label='Zbiornik 4')
plt.axhline(y=x_max, color='black', linestyle='--', label='h_max')
plt.axhline(y=x_min, color='black', linestyle='--', label='h_min')
plt.xlabel('Czas [s]')
plt.ylabel('Poziom wody [cm]')
plt.title("Poziom wody w 4 zbiornikach"+title_end)
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, q[0], '.', label='Pompa A')
plt.plot(time, q[1], '.', label='Pompa B')
plt.xlabel('Czas [s]')
plt.ylabel('Przepływ pompy [cm/s]')
plt.title('Przepływ pomp'+title_end)
plt.legend()
plt.grid()

plt.subplots_adjust(hspace=0.3)

plt.savefig(f"img/sub_bench{bench_name}_{operating_point}.png")
plt.show()