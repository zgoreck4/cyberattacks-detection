import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from four_tanks import simulate

T = 400
T_s = 1
time = np.arange(0, T+1, T_s)
T = max(time)
n_sampl = T//T_s+1
tau_u = 0
tau_y = 0
x0 = 165
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
x_max = np.inf
x_min = -np.inf
gamma_a = 0.3
gamma_b = 0.4
S = np.array([60, 60, 60, 60])
a = np.array([1.31, 1.51, 0.927, 0.882]) # przekrój otworu wylotowego
c = np.array([0.5, 0.5, 0.5, 0.5])

x, y, z = simulate(x0, x_max, x_min, gamma_a, gamma_b, S, a, c, q, T, T_s, tau_u, tau_y, active_noise, noise_sigma, e_sigma)

bench_name = '2'

title_end = f" dla $x_0$={x0}, $\gamma_a$={gamma_a}, $\gamma_b$={gamma_b}"

fig2=plt.figure(figsize=(8, 12))
plt.subplot(2, 1, 1)
plt.plot(time, x[0], label='Zbiornik 1')
plt.plot(time, x[1], label='Zbiornik 2')
plt.plot(time, x[2], label='Zbiornik 3')
plt.plot(time, x[3], label='Zbiornik 4')
plt.axhline(y=x_max, color='black', linestyle='--', label='h_max')
plt.axhline(y=x_min, color='black', linestyle='--', label='h_min')
plt.xlabel('Czas [s]')
plt.ylabel('Poziom wody [m]')
plt.title("Poziom wody w 4 zbiornikach"+title_end)
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, q[0], '.', label='Pompa A')
plt.plot(time, q[1], '.', label='Pompa B')
plt.xlabel('Czas [s]')
plt.ylabel('Przepływ pompy [m/s]')
plt.title('Przepływ pomp'+title_end)
plt.legend()
plt.grid()

plt.subplots_adjust(hspace=0.3)

plt.savefig(f"img/sub_{x0}_{qa}_{qb}_{gamma_a}_{gamma_b}_bench{bench_name}.png")
plt.show()

fig2 = plt.figure(figsize=(8, 12))

# First subplot: Water levels in 4 tanks
ax1 = plt.subplot(2, 1, 1)
ax1.plot(time, x[0], label='Zbiornik 1')
ax1.plot(time, x[1], label='Zbiornik 2')
ax1.plot(time, x[2], label='Zbiornik 3')
ax1.plot(time, x[3], label='Zbiornik 4')
ax1.axhline(y=x_max, color='black', linestyle='--', label='h_max')
ax1.axhline(y=x_min, color='black', linestyle='--', label='h_min')
ax1.set_xlabel('Czas [s]')
ax1.set_ylabel('Poziom wody [m]')
ax1.set_title("Poziom wody w 4 zbiornikach" + title_end)
ax1.legend()
ax1.grid()

# Inset for enlargement
# Define the time range and water level range you want to zoom into
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


time_min, time_max = 350, 400  # Example range for time
level_min, level_max = 62, 67  # Example range for water level

ax_inset = inset_axes(ax1, width="40%", height="40%", loc='upper right')
ax_inset.plot(time, x[0], label='Zbiornik 1')
ax_inset.plot(time, x[1], label='Zbiornik 2')
ax_inset.plot(time, x[2], label='Zbiornik 3')
ax_inset.plot(time, x[3], label='Zbiornik 4')
ax_inset.set_xlim(time_min, time_max)
ax_inset.set_ylim(level_min, level_max)
ax_inset.grid()

# Second subplot: Pump flows
ax2 = plt.subplot(2, 1, 2)
ax2.plot(time, q[0], '.', label='Pompa A')
ax2.plot(time, q[1], '.', label='Pompa B')
ax2.set_xlabel('Czas [s]')
ax2.set_ylabel('Przepływ pompy [m/s]')
ax2.set_title('Przepływ pomp' + title_end)
ax2.legend()
ax2.grid()

plt.subplots_adjust(hspace=0.3)

plt.show()