import numpy as np
import matplotlib.pyplot as plt
import cycler
from pathlib import Path
from ..simulation import Simulation
import pickle
from ..models import ELM, RBFNN
from ..detection import CyberattackDetector
from matplotlib.ticker import MaxNLocator
import re
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix

def main_function() -> None:

    save_mode = True
    close_loop = True
    recursion_mode = True
    SPh_var = 1

    active_noise = False # wartość False wyłącza zakłócenia, wartość True włącza
    noise_sigma = 0.15 # 0.1

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
    q_min = 0
    gamma_a = 0.3
    gamma_b = 0.4
    S = np.array([60, 60, 60, 60])
    a = np.array([1.31, 1.51, 0.927, 0.882]) # przekrój otworu wylotowego
    c = np.array([1, 1, 1, 1])

    kp = 2
    Ti = 15 # 1000000000000000000000
    Td = 0 # 1.5

    # cm
    step_dur = 3000/5 # 200

    SP_h = None
    q = None

    # Set global font sizes for different elements
    plt.rcParams.update({
        'axes.titlesize': 10,    # Titles of subplots
        'axes.labelsize': 9,     # Labels for axes
        'axes.prop_cycle': cycler.cycler(
            color=['tab:blue',
                   'tab:orange',
                   'tab:green',
                   'tab:brown',
                   'tab:purple',
                   'tab:cyan',
                   'tab:pink',
                   'tab:olive']
            ),
        'xtick.labelsize': 9,    # X-axis tick labels
        'ytick.labelsize': 9,    # Y-axis tick labels
        'legend.fontsize': 7,     # Legend font size
        'figure.titlesize': 12    # Overall figure title size (if used)
    })

    plot_path = Path(__file__).parent.parent / "plots/v2" 
    result_path = Path(__file__).parent.parent / "results"
    model_path = Path(__file__).parent.parent / "saved_models"

    # h = []
    y = []
    z = []
    q = []
    e = []
    h_model = []
    attack_signal = []

    model_type_tuple = ('lr', 'elm', 'rbf')
    model_dict = {'lr':'Regresja liniowa', 'elm': 'Sieć ELM', 'rbf': 'Sieć RBF'}

    for model_type in model_type_tuple:

        if model_type == 'lr':
            model1 = pickle.load(open(f"{model_path}/lr_x1.sav", 'rb'))
            model2 = pickle.load(open(f"{model_path}/lr_x2.sav", 'rb'))
            model3 = pickle.load(open(f"{model_path}/lr_x3.sav", 'rb'))
            model4 = pickle.load(open(f"{model_path}/lr_x4.sav", 'rb'))
            model_list = [model1, model2, model3, model4]
        elif model_type == 'elm':
            model1 = ELM(0, 0)
            model1.load_model(f"{model_path}/elm_x1.npz")
            model2 = ELM(0, 0)
            model2.load_model(f"{model_path}/elm_x2.npz")
            model3 = ELM(0, 0)
            model3.load_model(f"{model_path}/elm_x3.npz")
            model4 = ELM(0, 0)
            model4.load_model(f"{model_path}/elm_x4.npz")
            model_list = [model1, model2, model3, model4]
        elif model_type == 'rbf':
            model1 = RBFNN(None)
            # model1.load_model(f"{model_path}/rbf_x1.npz")
            model1.load_model(f"{model_path}/rbf_x1_v0.npz")
            model2 = RBFNN(None)
            model2.load_model(f"{model_path}/rbf_x2.npz")
            model3 = RBFNN(None)
            model3.load_model(f"{model_path}/rbf_x3.npz")
            model4 = RBFNN(None)
            model4.load_model(f"{model_path}/rbf_x4.npz")
            model_list = [model1, model2, model3, model4]
        else:
            model_list = None

        # należy ustawić próg w detektorze na podstawie normalnej pracy
        if SPh_var==0:
            SP_h1 = np.array([h0[0], h0[0], 80, 80, 100, 100, 90, 90, 40])
            SP_h2 = np.array([h0[1], 55,   55, 95, 95, 105, 105, 60, 60])
        elif SPh_var==1:
            SP_h1 = np.array([h0[0], h0[0], 50, 50, 80, 80, 100, 100, 90, 90, 40])
            SP_h2 = np.array([h0[1], 55,   55, 70, 70, 95, 95, 105, 105, 60, 60])
        elif SPh_var == 2:
            SP_h1 = np.array([h0[0], 60, 60, 100, 70, 70, 105])
            SP_h2 = np.array([h0[1], h0[1], 50, 80, 60, 85, 110])
        elif SPh_var == 3:
            SP_h1 = np.array([h0[0], h0[0], 70, 70, 95, 95, 90, 90, 40])
            SP_h2 = np.array([h0[1], 50,   50, 90, 90, 105, 105, 60, 60])
        SP_h = np.vstack((SP_h1, SP_h2))
        SP_h = np.repeat(SP_h, step_dur, axis=1)

        n_sampl = np.shape(SP_h)[1]
        T_s = 1
        T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
        time = np.arange(0, T, T_s)
        T = max(time)
        qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)
        simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd
                                # , noise_sigma, e_sigma
                                )
        h, _, _, q, _, h_model1, _ = simulation.run(h0,
                                    close_loop,
                                    model_list=model_list,
                                    recursion_mode=recursion_mode,
                                    SP_h=SP_h,
                                    q=q,
                                    qa0=1630000/3600,
                                    qb0=2000000/3600,
                                    attack_scenario=None)
        h_model.append(h_model1)
        
        # cyberattack_detector = CyberattackDetector(window=window_detection)
        # cyberattack_detector.calc_threshold(h[:len(h_model), :], h_model, method=threshold_method)

    h_model = np.array(h_model)

    print(np.shape(time))
    print(np.shape(q))
    print(q[0])

    plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(2, 1, 1, sharex=None)
    ax1.plot(time, q[0], label='$q_A$')
    ax1.plot(time, q[1], label='$q_B$')
    # plt.axhline(y=qa_max, color='black', linestyle='--', label='Pompa A max')
    # plt.axhline(y=qb_max, color='grey', linestyle='--', label='Pompa B max')
    ax1.set_ylabel('$q [cm^3/s]$')
    ax1.set_title('Przepływ pomp')
    ax1.legend(loc='center left')
    ax1.grid()

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(time, h[0], label='$h_1$')
    ax2.plot(time, h[1], label='$h_2$')
    ax2.step(time, SP_h[0], label='$w_1$', linestyle='-.')
    ax2.step(time, SP_h[1], label='$w_2$', linestyle=(0, (5, 5)))
    ax2.set_ylabel('h [cm]')
    ax2.set_title("Zmierzony i zadany poziom w zbiornikach")
    ax2.legend(loc='center left')
    ax2.grid()

    if save_mode:
        plt.savefig(f"{plot_path}/SP_PV_SPh_{SPh_var}_noise_{active_noise}.png",
                    bbox_inches ='tight')
        
    plt.show()

    def calc_NRMSE(true, predict, y_name_idx):
        RMSE = metrics.root_mean_squared_error(true, predict)  
        return RMSE/(h_max[y_name_idx][0] - h_min[y_name_idx][0])

    for i, h_i in enumerate(h):
        print("-"*15)
        print(f"Zbiornik {i+1}.")
        for h_model_model, model_type in zip(h_model, model_type_tuple):
            NRMSE = calc_NRMSE(h_i, h_model_model[i], i)
            print(f"NRMSE {model_type.upper()}: {NRMSE:.4f}")

    plt.figure(figsize=(16, 8.5))
    plt.suptitle("Poziom rzeczywisty i przewidywany w stanie normalnym")
    for i, h_i in enumerate(h):
        ax1 = plt.subplot(2, 2, i+1)
        ax1.plot(time, h_i, label=f'pomiar $h_{i+1}$')
        plt.axhline(y=h_max[i], color='black', linestyle='--', label=f'h_max{i+1}')
        plt.axhline(y=h_min[i], color='black', linestyle='--', label=f'h_min{i+1}')
        for h_model_model, model_type in zip(h_model, model_type_tuple):
            ax1.plot(time, h_model_model[i, :], linestyle='--', label=rf'model {model_type.upper()} $\hat{{h_{i+1}}}$')
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel(f'$h_{i+1} [cm]$')
        ax1.set_title(f"{i+1}. zbiornik")
        ax1.legend(loc='best')
        ax1.grid()
    # Adjust layout for spacing
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust for the title
    plt.subplots_adjust(hspace=0.2)  # Add vertical space between subplots

    if save_mode:
        plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_noise_{active_noise}_SPh_{SPh_var}_noise_{active_noise}.png",
                    bbox_inches ='tight')
    
    plt.show()

if __name__ == "__main__":
    main_function()