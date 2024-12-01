import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from ..simulation import simulate, simulate_close_loop

close_loop = False

tau_u = 0
tau_y = 0
# h0 = [65, 66, 65, 66]
# dla gamma_a = 0.3, gamma_b = 0.4
h0 = [65.37515073890378, 64.98201463176996, 65.90206440432354, 65.8157923349714]
# h0 = [60.88756594079625, 68.99253334710369, 0, 0]
# dla gamma = 0.8
# h0 = [66.54149974283987, 63.981536657570146, 7.322451600480413, 5.372717741630307]
# dla gamma_a = 0.7, gamma_b = 0.6
# h0 = [86.33849517038516, 49.20411218062061, 29.28980640192163, 12.088614918668185]
# h0 = [12.4, 1.8, 12.7, 1.4]
# h0 = [12.6, 4.8, 13, 4.9]
active_noise = False # wartość False wyłącza zakłócenia, wartość True włącza
noise_sigma = 0.1
e_sigma = 0.005

# cm
step_dur = 3000/5 # 200
SP_h1 = np.array([h0[0], 60, 60, 100, 70, 70, 105])
SP_h2 = np.array([h0[1], h0[1], 50, 80, 60, 85, 110])
SP_h = np.vstack((SP_h1, SP_h2))
SP_h = np.repeat(SP_h, step_dur, axis=1)
print(f"{SP_h}")

n_sampl = np.shape(SP_h)[1]
T_s = 1
T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
time = np.arange(0, T, T_s)
T = max(time)

qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)

h_max = [[136],
         [136],
         [130],
         [130]]
h_min = [[20],
         [20],
         [20],
         [20]]

qa_max = 3260000/3600
qb_max = 4000000/3600
gamma_a = 0.3
gamma_b = 0.4
S = np.array([60, 60, 60, 60])
a = np.array([1.31, 1.51, 0.927, 0.882]) # przekrój otworu wylotowego
c = np.array([0.5, 0.5, 0.5, 0.5])

kp = 2
Ti = 15 # 1000000000000000000000
Td = 0 # 1.5

if close_loop:
    h, y, z, q, e = simulate_close_loop(h0, h_max, h_min, qa_max, qb_max, gamma_a, gamma_b, S, a, c, SP_h, T, T_s, kp, Ti, Td, tau_u, tau_y, active_noise, qd, noise_sigma, e_sigma)
else:
    # cm
    qa = 1630000/3600
    qb = 2000000/3600
    q = np.vstack((np.ones((1, n_sampl))* qa, np.ones((1, n_sampl))* qb))
    h, y, z = simulate(h0, h_max, h_min, gamma_a, gamma_b, S, a, c, q, T, T_s, tau_u, tau_y, active_noise, qd, noise_sigma, e_sigma)

for i in range(4):
    print(F"h{i+1} = {h[i, [-1]][0]}")
print(", ".join([str(hi) for hi in h[:, -1]]))

bench_name = '2'
operating_point = ''

# title_end = f" dla {bench_name}. benchmarku i punktu pracy {operating_point}"
title_end = f" dla {bench_name}. benchmarku"

# Set global font sizes for different elements
plt.rcParams.update({
    'axes.titlesize': 10,    # Titles of subplots
    'axes.labelsize': 9,     # Labels for axes
    'xtick.labelsize': 9,    # X-axis tick labels
    'ytick.labelsize': 9,    # Y-axis tick labels
    'legend.fontsize': 7,     # Legend font size
    'figure.titlesize': 12    # Overall figure title size (if used)
})

plot_path = Path(__file__).parent.parent / "plots"

if close_loop:

    fig2=plt.figure(figsize=(8, 9))
    plt.subplot(3, 1, 1)
    plt.plot(time, q[0], '.', label='Pompa A')
    plt.plot(time, q[1], '.', label='Pompa B')
    plt.axhline(y=qa_max, color='black', linestyle='--', label='Pompa A max')
    plt.axhline(y=qb_max, color='grey', linestyle='--', label='Pompa B max')
    plt.xlabel('Czas [s]')
    plt.ylabel('Przepływ pompy [cm/s]')
    plt.title('Przepływ pomp'+title_end)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time, h[0], label='Zbiornik 1')
    plt.plot(time, h[1], label='Zbiornik 2')
    plt.plot(time, SP_h[0], label='SP zbiornik 1', linestyle='-.')
    plt.plot(time, SP_h[1], label='SP zbiornik 2', linestyle=(0, (5, 5)))
    plt.axhline(y=h_max[0], color='black', linestyle='--', label='h_max12')
    plt.axhline(y=h_min[0], color='black', linestyle='--', label='h_min')
    plt.xlabel('Czas [s]')
    plt.ylabel('Poziom wody [cm]')
    plt.title("SP i PV"+title_end)
    plt.legend()
    plt.grid()

    # plt.subplot(3, 1, 3)
    # plt.plot(time, e[0], label='Zbiornik 1')
    # plt.plot(time, e[1], label='Zbiornik 2')
    # plt.legend()
    # plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time, h[0], label='Zbiornik 1')
    plt.plot(time, h[1], label='Zbiornik 2')
    plt.plot(time, h[2], label='Zbiornik 3')
    plt.plot(time, h[3], label='Zbiornik 4')
    # plt.axhline(y=h_max[0], color='black', linestyle='--', label='h_max1')
    # plt.axhline(y=h_max[1], color='black', linestyle='--', label='h_max2')
    # plt.axhline(y=h_max[2], color='black', linestyle='--', label='h_max3')
    # plt.axhline(y=h_max[3], color='black', linestyle='--', label='h_max4')
    # plt.axhline(y=h_min[0], color='black', linestyle='--', label='h_min1')
    # plt.axhline(y=h_min[1], color='black', linestyle='--', label='h_min2')
    # plt.axhline(y=h_min[2], color='black', linestyle='--', label='h_min3')
    # plt.axhline(y=h_min[3], color='black', linestyle='--', label='h_min4')
    plt.axhline(y=h_max[0], color='black', linestyle='--', label='h_max12')
    plt.axhline(y=h_max[2], color='black', linestyle='--', label='h_max34')
    plt.axhline(y=h_min[0], color='black', linestyle='--', label='h_min')
    plt.xlabel('Czas [s]')
    plt.ylabel('Poziom wody [cm]')
    plt.title("Poziom wody w 4 zbiornikach"+title_end)
    plt.legend()
    plt.grid()

    plt.subplots_adjust(hspace=0.5)

    # plt.savefig(f"{plot_path}/sub_bench{bench_name}_{operating_point}.png")
    plt.show()

else:
    fig2=plt.figure(figsize=(6, 7))
    plt.subplot(2, 1, 1)
    plt.plot(time, h[0], label='Zbiornik 1')
    plt.plot(time, h[1], label='Zbiornik 2')
    plt.plot(time, h[2], label='Zbiornik 3')
    plt.plot(time, h[3], label='Zbiornik 4')
    # plt.axhline(y=h_max[0], color='black', linestyle='--', label='h_max1')
    # plt.axhline(y=h_max[1], color='black', linestyle='--', label='h_max2')
    # plt.axhline(y=h_max[2], color='black', linestyle='--', label='h_max3')
    # plt.axhline(y=h_max[3], color='black', linestyle='--', label='h_max4')
    # plt.axhline(y=h_min[0], color='black', linestyle='--', label='h_min1')
    # plt.axhline(y=h_min[1], color='black', linestyle='--', label='h_min2')
    # plt.axhline(y=h_min[2], color='black', linestyle='--', label='h_min3')
    # plt.axhline(y=h_min[3], color='black', linestyle='--', label='h_min4')
    plt.axhline(y=h_max[0], color='black', linestyle='--', label='h_max12')
    plt.axhline(y=h_max[2], color='black', linestyle='--', label='h_max34')
    plt.axhline(y=h_min[0], color='black', linestyle='--', label='h_min')
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

    # plt.savefig(f"{plot_path}/sub_bench{bench_name}_{operating_point}.png")
    plt.show()