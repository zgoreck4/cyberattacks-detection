import numpy as np
import matplotlib.pyplot as plt
import cycler
from pathlib import Path
from ..simulation import Simulation
import pickle
from ..models import ELM, RBFNN
from ..detection import CyberattackDetector
from matplotlib.ticker import MaxNLocator

def main_function() -> None:

    save_mode = False
    close_loop = True
    attack_scenario = 0 # 3
    num_tank = 0
    attack_value = 0.05
    tau_y_ca = 10
    model_type = 'elm' # None
    recursion_mode = True
    window_detection = 50
    threshold_method = 'z-score' # 'percentile'

    variability=True
    param_name='gamma_a'
    param_value=0.2

    active_noise = False # wartość False wyłącza zakłócenia, wartość True włącza
    noise_sigma = 0.2 # 0.1
    
    if variability==False:
        param_name=None
        param_value=None

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

    model_path = Path(__file__).parent.parent / "saved_models"

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
        model1.load_model(f"{model_path}/rbf_x1.npz")
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
    if ((attack_scenario is not None) or (variability == True)) and model_list is not None:
        SP_h1 = np.array([h0[0], h0[0], 70, 70, 95, 95, 90
                          #, 90, 40
                          ])
        SP_h2 = np.array([h0[1], 50,   50, 90, 90, 105, 105
                          # , 60, 60
                          ])
        SP_h = np.vstack((SP_h1, SP_h2))
        SP_h = np.repeat(SP_h, step_dur, axis=1)

        n_sampl = np.shape(SP_h)[1]
        T_s = 1
        T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
        time = np.arange(0, T, T_s)
        T = max(time)
        qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)
        simulation_normal = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd
                                # , noise_sigma, e_sigma
                                )
        h, y, z, q, e, h_model, attack_signal = simulation_normal.run(h0,
                                    close_loop,
                                    model_list=model_list,
                                    recursion_mode=recursion_mode,
                                    SP_h=SP_h,
                                    q=q,
                                    qa0=1630000/3600,
                                    qb0=2000000/3600,
                                    attack_scenario=None)
        
        cyberattack_detector = CyberattackDetector(window=window_detection)
        cyberattack_detector.calc_threshold(h[:len(h_model), :], h_model, method=threshold_method)

        plt.figure(figsize=(8, 9))
        plt.title("Poziom rzeczywisty i przewidywany w stanie normalnym")
        for i, (h_i, h_model_i) in enumerate(zip(h, h_model)):
            ax1 = plt.subplot(4, 1, i+1)
            ax1.plot(time, h_i, label=f'pomiar $h_{i+1}$')
            # plt.axhline(y=h_max[i], color='black', linestyle='--', label=f'h_max{i+1}')
            # plt.axhline(y=h_min[i], color='black', linestyle='--', label=f'h_min{i+1}')
            if model_type is not None:
                ax1.plot(time, h_model_i, linestyle='--', label=rf'model {model_type.upper()} $\hat{{h_{i+1}}}$')
            ax1.set_xlabel('k')
            ax1.set_ylabel(f'$h_{i+1} [cm]$')
            # ax1.title(f"Poziom wody w {i+1} zbiorniku")
            ax1.legend(loc='best', bbox_to_anchor=(0, 0, 0.5, 1.0))
            ax1.grid()
        plt.show()

    else:
        cyberattack_detector = None

    if close_loop:
        SP_h1 = np.array([h0[0], 60,    60, 60, 100, 100, 100])
        SP_h2 = np.array([h0[1], h0[1], 50, 80, 80, 105, 90])
        # SP_h1 = np.array([h0[0], 80])
        # SP_h2 = np.array([h0[1], h0[1]])
        SP_h = np.vstack((SP_h1, SP_h2))
        SP_h = np.repeat(SP_h, step_dur, axis=1)
        # print(f"{SP_h}")
        n_sampl = np.shape(SP_h)[1]

    else:
        n_step = 4
        qa = np.hstack((np.array(1.63*1000000/3600), np.random.rand(n_step-1)*qa_max*0.8))
        qb = np.hstack((np.array(2*1000000/3600), np.random.rand(n_step-1)*qb_max*0.8))
        qa = np.clip(qa, q_min, qa_max)
        qb = np.clip(qb, q_min, qb_max)
        q = np.vstack((qa, qb))
        q = np.repeat(np.round(q, 2), step_dur, axis=1)
        n_sampl = np.shape(q)[1]

    T_s = 1
    T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
    time = np.arange(0, T, T_s)
    T = max(time)

    attack_time = n_sampl//2    
    time_change=n_sampl//3

    qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)

    simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                            S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd, cyberattack_detector=cyberattack_detector
                            # , noise_sigma, e_sigma
                            )

    h, y, z, q, e, h_model, attack_signal = simulation.run(h0,
                                close_loop,
                                model_list=model_list,
                                recursion_mode=recursion_mode,
                                SP_h=SP_h,
                                q=q,
                                qa0=1630000/3600,
                                qb0=2000000/3600,
                                attack_scenario=attack_scenario,
                                num_tank=num_tank,
                                attack_time=attack_time,
                                attack_value=attack_value,
                                tau_y_ca=tau_y_ca,
                                variability=variability,
                                param_name=param_name,
                                param_value=param_value,
                                time_change=time_change
                                )

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

    plot_path = Path(__file__).parent.parent / "plots"

    attack_binary = np.hstack((np.zeros(attack_time+1), np.ones(T-attack_time)))
    change_binary = np.hstack((np.zeros(time_change+1), np.ones(T-time_change)))

    if close_loop:

        fig2=plt.figure(figsize=(8, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time, q[0], label='$q_A$')
        ax1.plot(time, q[1], label='$q_B$')
        # plt.axhline(y=qa_max, color='black', linestyle='--', label='Pompa A max')
        # plt.axhline(y=qb_max, color='grey', linestyle='--', label='Pompa B max')
        ax1.set_xlabel('k')
        ax1.set_ylabel('$q [cm^3/s]$')
        ax1.set_title('Przepływ pomp')
        ax1.legend(loc='best', bbox_to_anchor=(0, 0, 0.1, 1.0))
        ax1.grid()

        if (attack_scenario is not None) or (variability is not None):
            ax1_secondary = ax1.twinx()
            ax1_secondary.set_ylabel('Sygnał binarny')
            ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))
        if attack_scenario is not None:
            ax1_secondary.plot(time, attack_binary, color='tab:red', linestyle='--', label='cyberatak')
        if variability==True:
            ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle='--', label='zmiana param')
        if (attack_scenario is not None) or (variability is not None):
            ax1_secondary.legend(loc='best', bbox_to_anchor=(0.8, 0., 0.2, 1.0))


        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(time, z[0], label='$h_1$')
        ax2.plot(time, z[1], label='$h_2$')
        print(np.hstack((time, [max(time)+1])))
        ax2.step(time, SP_h[0], label='$w_1$', linestyle='-.')
        ax2.step(time, SP_h[1], label='$w_2$', linestyle=(0, (5, 5)))
        # ax2.axhline(y=h_max[0], color='black', linestyle='--', label='h_max12')
        # ax2.axhline(y=h_min[0], color='black', linestyle='--', label='h_min')
        ax2.set_xlabel('k')
        ax2.set_ylabel('h [cm]')
        ax2.set_title("Zmierzony i zadany poziom w zbiornikach")
        ax2.legend(loc='best', bbox_to_anchor=(0, 0, 0.1, 1.0))
        ax2.grid()

        if (attack_scenario is not None) or (variability is not None):
            ax2_secondary = ax2.twinx()
            ax2_secondary.set_ylabel('Sygnał binarny')
            ax2_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))
        if attack_scenario is not None:
            ax2_secondary.plot(time, attack_binary, color='tab:red', linestyle='--', label='cyberatak')
        if variability==True:
            ax2_secondary.plot(time, change_binary, color='tab:pink', linestyle='--', label='zmiana param')
        if (attack_scenario is not None) or (variability is not None):
            ax2_secondary.legend(loc='best', bbox_to_anchor=(0.8, 0., 0.2, 1.0))

        # plt.subplot(3, 1, 3)
        # plt.plot(time, e[0], label='Zbiornik 1')
        # plt.plot(time, e[1], label='Zbiornik 2')
        # plt.legend()
        # plt.grid()

        plt.subplots_adjust(hspace=0.3)

        if save_mode:
            plt.savefig(f"{plot_path}/SP_PV_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_method_{threshold_method}_variability_{param_name}_{param_value}_noise_{active_noise}.png")
        plt.show()

        if model_type is not None:
            title_part = ' - rzeczywisty i modelowany'
        else:
            title_part = ''
        if recursion_mode:
            title_recursion = 'z rekurencją'
        else:
            title_recursion = 'bez rekurencji'

        fig2=plt.figure(figsize=(8, 9))
        # fig2.suptitle(f'Poziom wody w 2 zbiornikach {title_part} w trybie {title_recursion}')
        fig2.suptitle(f'Poziom wody w 2 zbiornikach {title_part}')
        for i, (zi, hi) in enumerate(zip(z[:2, :], h[:2, :])):
            ax1 = plt.subplot(2, 1, i+1)
            ax1.plot(time, zi, label=f'pomiar $h_{i+1}$')
            # plt.axhline(y=h_max[i], color='black', linestyle='--', label=f'h_max{i+1}')
            # plt.axhline(y=h_min[i], color='black', linestyle='--', label=f'h_min{i+1}')
            if model_type is not None:
                ax1.plot(time, h_model[i], linestyle='--', label=rf'model {model_type.upper()} $\hat{{h_{i+1}}}$')
            ax1.set_xlabel('k')
            ax1.set_ylabel(f'$h_{i+1} [cm]$')
            # ax1.title(f"Poziom wody w {i+1} zbiorniku")
            ax1.grid()

            if attack_scenario is not None:
                ax1.plot(time, hi, linestyle='-.', label=f'rzecz. $h_{i+1}$')
                # Add secondary y-axis to the first subplot
                ax1_secondary = ax1.twinx()
                ax1_secondary.plot(time, attack_binary, color='red', linestyle='--', label='cyberatak')
                if model_type is not None:
                    ax1_secondary.fill_between(time, attack_signal[:, i].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')
                ax1_secondary.set_ylabel('Sygnał binarny cyberataku')
                # set y-axis to only show integer values
                ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax1_secondary.legend(loc='best', bbox_to_anchor=(0.8, 0., 0.2, 1.0))
            ax1.legend(loc='best', bbox_to_anchor=(0, 0, 0.2, 1.0))

        # plt.subplots_adjust(hspace=0.5)

        if save_mode:
            plt.savefig(f"{plot_path}/h__{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_method_{threshold_method}_variability_{param_name}_{param_value}_noise_{active_noise}.png")
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
        plt.title("Poziom wody w 4 zbiornikach")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(time, q[0], '.', label='Pompa A')
        plt.plot(time, q[1], '.', label='Pompa B')
        plt.xlabel('Czas [s]')
        plt.ylabel('Przepływ pompy [cm/s]')
        plt.title('Przepływ pomp')
        plt.legend()
        plt.grid()

        plt.subplots_adjust(hspace=0.3)

        if save_mode:
            plt.savefig(f"{plot_path}/sub_bench{bench_name}_{operating_point}_variability_{param_name}_{param_value}_noise_{active_noise}.png")
        plt.show()

if __name__ == "__main__":
    main_function()