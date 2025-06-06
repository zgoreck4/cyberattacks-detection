import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cycler
from ..simulation import Simulation
from pathlib import Path

dataset_name = "uczące"

close_loop = False
attack_scenario = None

tau_u = 0
tau_y = 0
active_noise = True  # True enables noise, False disables it
noise_sigma = 0.15  # Standard deviation of noise

qa_max = 3260000 / 3600  # Maximum flow rate for qa [cm^3/s]
qb_max = 4000000 / 3600  # Maximum flow rate for qb [cm^3/s]
q_min = 0  # Minimum flow rate

# Define different datasets based on dataset_name
if dataset_name == "uczące":
    np.random.seed(0)
    n_step_val = 15
    qa = np.array([
        1.63 * 1000000 / 3600, 692.39, 507.13, 585.72, 905, 706.45, 320.03,
        581.83, 432.22, 438.76, 50, 220, 650.32, 469.31, 513.07
    ])
    qb = np.array([
        2 * 1000000 / 3600, 611.17, 804.57, 521.36, 413.21, 50, 664.49,
        699.63, 431.86, 933.85, 313.16, 563.18, 524.36, 811.02, 1111
    ])

elif dataset_name == 'walidacyjne':
    np.random.seed(32)
    n_step_val = 5
    qa = np.hstack((
        np.array(1.63 * 1000000 / 3600),
        np.random.rand(n_step_val - 1) * qa_max * 0.8 + q_min
    ))
    qb = np.hstack((
        np.array(2 * 1000000 / 3600),
        np.random.rand(n_step_val - 1) * qb_max * 0.8 + q_min
    ))

elif dataset_name == 'testowe':
    np.random.seed(32)
    n_step_val = 5
    qa = np.hstack((
        np.array(1.63 * 1000000 / 3600),
        np.random.rand(n_step_val - 1) * qa_max * 0.8 + q_min
    ))
    qb = np.hstack((
        np.array(2 * 1000000 / 3600),
        np.random.rand(n_step_val - 1) * qb_max * 0.8 + q_min
    ))

SP_h = None  # Setpoint for tank levels (not used in open loop)

step_dur = 200  # Duration of each step in simulation time units
n_sampl = n_step_val * step_dur  # Total number of simulation samples
T_s = 1  # Sampling period (seconds)
T = n_sampl * T_s - 1  # Total simulation time
time = np.arange(0, T + 1, T_s)  # Time vector

h0 = [65, 66, 65, 66]  # Initial tank levels

# Clip inflows to their min and max allowed values
qa = np.clip(qa, q_min, qa_max)
qb = np.clip(qb, q_min, qb_max)

# Repeat each flow value for step_dur samples to create piecewise constant input signals
q = np.vstack((qa, qb))
q = np.repeat(np.round(q, 2), step_dur, axis=1)

# Generate random noise for the four tanks, scaled by noise_sigma and active_noise flag
qd = np.round(np.random.randn(4, n_sampl) * noise_sigma * active_noise, 4)

# System parameters
h_max = 136  # Maximum tank level
h_min = 20   # Minimum tank level
gamma_a = 0.3
gamma_b = 0.4
S = np.array([60, 60, 60, 60])  # Cross-sectional area of tanks
a = np.array([1.31, 1.51, 0.927, 0.882])  # Outlet orifice areas
c = np.array([1, 1, 1, 1])  # Sensor calibration coefficients (assumed 1)

# PID controller parameters
kp = 2
Ti = 15
Td = 0

# Initialize simulation object
simulation = Simulation(
    h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
    S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd
)

# Run simulation in open-loop mode with predefined input flows q
h, y, z, q, e, _, _ = simulation.run(
    h0, close_loop, SP_h=SP_h, q=q,
    qa0=1630000 / 3600, qb0=2000000 / 3600,
    attack_scenario=attack_scenario
)

# Prepare DataFrame with results for saving
result = pd.DataFrame(
    np.vstack((q, qd, h)).T,
    columns=[
        'q_A [cm^3/s]', 'q_B [cm^3/s]', 'q_d1 [cm^3/s]', 'q_d2 [cm^3/s]',
        'q_d3 [cm^3/s]', 'q_d4 [cm^3/s]', 'h1 [cm]', 'h2 [cm]', 'h3 [cm]', 'h4 [cm]'
    ],
    index=time
)

# Define file path for dataset saving depending on noise flag
if active_noise:
    dataset_path = Path(__file__).parent.parent / f"data/four_tanks/result_ol_with_noise_{dataset_name}_v5.csv"
else:
    dataset_path = Path(__file__).parent.parent / f"data/four_tanks/result_ol_without_noise_{dataset_name}_v5.csv"

result.to_csv(dataset_path, sep=';')  # Save results to CSV

# Prepare plot title suffix based on noise activation
if active_noise:
    title_end = " z zakłóceniami"
else:
    title_end = " bez zakłóceń"

plot_path = Path(__file__).parent.parent / "plots/v4"
plot_path.mkdir(parents=True, exist_ok=True)  # Ensure plot directory exists

# Update matplotlib plot styles globally
plt.rcParams.update({
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'axes.prop_cycle': cycler.cycler(
        color=['tab:blue', 'tab:orange', 'tab:green', 'tab:brown',
               'tab:purple', 'tab:cyan', 'tab:pink', 'tab:olive']
    ),
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
})

def plot_data(h, q, time, dataset_name):
    """
    Plot tank levels and input flows over time.

    Args:
        h (np.ndarray): Tank levels, shape (4, n_samples)
        q (np.ndarray): Inflows q_A and q_B, shape (2, n_samples)
        time (np.ndarray): Time vector
        dataset_name (str): Name of dataset used (for title and saving)
    """
    fig = plt.figure(figsize=(6, 10))

    # Plot tank levels with max/min level reference lines
    plt.subplot(3, 1, 1)
    plt.grid()
    plt.axhline(y=h_max, color='black', linestyle='--', label='$h_{max}$')
    plt.axhline(y=h_min, color='black', linestyle='--', label='$h_{min}$')
    plt.plot(time, h[0], label='$h_1$', linestyle='-')
    plt.plot(time, h[1], label='$h_2$', linestyle='-.')
    plt.plot(time, h[2], label='$h_3$', linestyle=(5, (10, 3)))
    plt.plot(time, h[3], label='$h_4$', linestyle=':')
    plt.ylabel('h [cm]')
    plt.legend()

    # Plot inflow q_A with max limit
    plt.subplot(3, 1, 2)
    plt.grid()
    plt.axhline(y=qa_max, color='black', linestyle='--', label='$q_{Amax}$')
    plt.stairs(q[0], np.append(time, time[-1] + 1), label='$q_A$')
    plt.ylabel('$q_A [cm^3/s]$')
    plt.legend()

    # Plot inflow q_B with max limit
    plt.subplot(3, 1, 3)
    plt.grid()
    plt.axhline(y=qb_max, color='black', linestyle='--', label='$q_{Bmax}$')
    plt.stairs(q[1], np.append(time, time[-1] + 1), label='$q_B$')
    plt.xlabel('t [s]')
    plt.ylabel('$q_B [cm^3/s]$')
    plt.legend()

    plt.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"Dane {dataset_name}{title_end}")

    # Save plot to file
    plt.savefig(f"{plot_path}/data_ol_{active_noise}_{dataset_name}_v5.pdf", bbox_inches='tight')
    plt.show()

plot_data(h, q, time, dataset_name)