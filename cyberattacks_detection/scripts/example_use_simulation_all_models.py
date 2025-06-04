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
import keras
import pandas as pd
from sklearn.metrics import confusion_matrix

seed=32
rng = np.random.default_rng(seed)

def rmse(y, y_hat, window):
    """
    funkcja licząca błąd średniokwadratowy w przesuwnym oknie
    """
    se = ((y - y_hat)**2)
    return se.rolling(window, min_periods=1).mean()**0.5
def mae(y, y_hat, window):
    """
    funkcja licząca średni błąd bezwzględny w przesuwnym oknie
    """
    se = abs(y - y_hat)
    return se.rolling(window, min_periods=1).mean()

def main_function() -> None:
    save_mode = True
    close_loop = True
    attack_scenario = 2 # 1 # None
    num_tank = 0
    attack_value = 0.05
    tau_y_ca = 50
    # model_type = 'lr' # None
    recursion_mode = True # True
    detection_from_file = False
    window_detection = 110
    residual_calc_func = 'rmse'
    threshold_method = 'percentile' # 'max' # 'z-score' # 'percentile'
    kwargs = {'n_std': 3, 'percentile': 99}

    variability=False
    # param_name='gamma_a'
    # param_value=0.2 # z 0.3 na 0.2
    param_name = 'a'
    param_value = np.array([1.2, 1.51, 0.927, 0.882])

    active_noise = True # wartość False wyłącza zakłócenia, wartość True włącza
    noise_sigma = 0.15 # 0.15
    if not active_noise:
        noise_sigma = 0
    
    if variability==False:
        param_name=None
        param_value=None

    threshold_method_dict = {'z-score': f'z-score (z={kwargs["n_std"]})', 'percentile': f'oparta na percent. ({kwargs["percentile"]}%)'}
    recursion_mode_dict = {True: 'rekurencyjnym', False: 'bez rekurencji'}

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
    result_path = Path(__file__).parent.parent / "results/v2"
    plot_path = Path(__file__).parent.parent / "plots/v3/good_param"

    h = []
    y = []
    z = []
    q = []
    e = []
    h_model = []
    attack_signal = []
    residuals = []
    residualsi = []
    threshold = []

    model_type_tuple = (
                        'lr',
                        'elm', 'rbf', 'gru',
                        'lstm',
                        'lstm-mlp',
                        )
    model_normal_file_names = (f"model_lr_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                               , f"model_elm_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                                , f"model_rbf_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                                , f"model_gru_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                                , f"model_lstm_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                                , f"model_lstm-mlp_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
                                )
    model_file_names = (f"model_lr_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        , f"model_elm_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        , f"model_rbf_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        , f"model_gru_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        , f"model_lstm_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        , f"model_lstm-mlp_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
                        )
    # model_file_names = (f"model_lr_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     , f"model_elm_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     , f"model_rbf_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     , f"model_gru_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     , f"model_lstm_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     , f"model_lstm-mlp_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window110_met_percentile_nstd3_perc98.5_noise_{noise_sigma}.csv"
    #                     )
    # model_normal_file_names = (None, None, None, None, None, None)
    # model_file_names = (None, None, None, None, None, None)
    model_dict = {'lr':'Regresja liniowa', 'elm': 'Sieć ELM', 'rbf': 'Sieć RBF', 'lstm': 'Sieć LSTM', 'gru': 'Sieć GRU', 'lstm-mlp': 'Sieć LSTM-MLP'}

    for model_type, model_normal_file_name, model_file_name in zip(model_type_tuple, model_normal_file_names, model_file_names):
        print(model_type)

        if (attack_scenario is not None and model_normal_file_name is None) or (model_file_name is None):

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
            elif model_type == 'lstm':
                model1 = keras.models.load_model(f"{model_path}/lstm_x1.keras")
                model2 = keras.models.load_model(f"{model_path}/lstm_x2.keras")
                model3 = keras.models.load_model(f"{model_path}/lstm_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/lstm_x4.keras")
                model_list = [model1, model2, model3, model4]
            elif model_type == 'gru':
                model1 = keras.models.load_model(f"{model_path}/gru_x1.keras")
                model2 = keras.models.load_model(f"{model_path}/gru_x2.keras")
                model3 = keras.models.load_model(f"{model_path}/gru_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/gru_x4.keras")
                model_list = [model1, model2, model3, model4]
            elif model_type == 'lstm-mlp':
                # model1 = keras.models.load_model(f"{model_path}/lstm_mlp_x1.keras")
                # model2 = keras.models.load_model(f"{model_path}/lstm_mlp_x2.keras")
                model1 = keras.models.load_model(f"{model_path}/lstm_mlp_x1_statespace.keras")
                model2 = keras.models.load_model(f"{model_path}/lstm_mlp_x2_statespace.keras")
                model3 = keras.models.load_model(f"{model_path}/lstm_mlp_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/lstm_mlp_x4.keras")
                # model3 = keras.models.load_model(f"{model_path}/lstm_x3.keras")
                # model4 = keras.models.load_model(f"{model_path}/lstm_x4.keras")
                model_list = [model1, model2, model3, model4]
            else:
                model_list = None

        # należy ustawić próg w detektorze na podstawie normalnej pracy
        if ((attack_scenario is not None) or (variability == True)):
            SP_h1 = np.array([h0[0], h0[0], 50, 50, 80, 80, 100, 100, 90, 90, 40])
            SP_h2 = np.array([h0[1], 55,    55, 70, 70, 95, 95, 105, 105, 60, 60])
            # SP_h1 = np.array([h0[0], h0[0], 80, 80, 100, 100, 90])
            # SP_h2 = np.array([h0[1], 55,   55, 95, 95, 105, 105])
            SP_h = np.vstack((SP_h1, SP_h2))
            SP_h = np.repeat(SP_h, step_dur, axis=1)

            n_sampl = np.shape(SP_h)[1]
            T_s = 1
            T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
            time = np.arange(0, T, T_s)
            T = max(time)
            qd = np.round(rng.standard_normal(size=(4,n_sampl))*noise_sigma*active_noise, 4)
            if model_normal_file_name is None:
                simulation_normal = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                        S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd
                                        # , noise_sigma, e_sigma
                                        )
                h_normal, _, _, _, _, h_model_normal, _ = simulation_normal.run(h0,
                                            close_loop,
                                            model_list=model_list,
                                            recursion_mode=recursion_mode,
                                            SP_h=SP_h,
                                            q=q,
                                            qa0=1630000/3600,
                                            qb0=2000000/3600,
                                            attack_scenario=None)                 
                df = pd.DataFrame(np.concatenate((h_normal, h_model_normal), axis=0), index= ['x1', 'x2', 'x3', 'x4', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']).T
                df.to_csv(f"{result_path}/model_{model_type}_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
                      
            else:
                df = pd.read_csv(f"{result_path}/{model_normal_file_name}", sep=';')
                h_normal = df[['x1', 'x2', 'x3', 'x4']].T.values
                h_model_normal = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
            
            cyberattack_detector = CyberattackDetector(window=window_detection, residual_calc_func=residual_calc_func)
            cyberattack_detector.calc_threshold(h_normal[:len(h_model_normal), :], h_model_normal, method=threshold_method, **kwargs)
            thresholdi = np.array(cyberattack_detector.threshold)

            # plt.figure(figsize=(8, 9))
            # plt.title("Poziom rzeczywisty i przewidywany w stanie normalnym")
            # for i, (h_i, h_model_i) in enumerate(zip(h_normal, h_model_normal)):
            #     ax1 = plt.subplot(4, 1, i+1)
            #     ax1.plot(time, h_i, label=f'pomiar $h_{i+1}$')
            #     # plt.axhline(y=h_max[i], color='black', linestyle='--', label=f'h_max{i+1}')
            #     # plt.axhline(y=h_min[i], color='black', linestyle='--', label=f'h_min{i+1}')
            #     if model_type is not None:
            #         ax1.plot(time, h_model_i, linestyle='--', label=rf'model {model_type.upper()} $\hat{{h_{i+1}}}$')
            #     ax1.set_xlabel('t [s]')
            #     ax1.set_ylabel(f'$h_{i+1} [cm]$')
            #     # ax1.title(f"Poziom wody w {i+1} zbiorniku")
            #     ax1.legend(loc='best', bbox_to_anchor=(0, 0, 0.5, 1.0))
            #     ax1.grid()
            # plt.show()

        else:
            cyberattack_detector = None

        SP_h1 = np.array([h0[0], h0[0], 70, 70, 95, 95, 90])
        SP_h2 = np.array([h0[1], 50,   50, 90, 90, 105, 105])
        # SP_h1 = np.array([h0[0], 80])
        # SP_h2 = np.array([h0[1], h0[1]])
        SP_h = np.vstack((SP_h1, SP_h2))
        SP_h = np.repeat(SP_h, step_dur, axis=1)
        # print(f"{SP_h}")
        n_sampl = np.shape(SP_h)[1]

        T_s = 1
        T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
        time = np.arange(0, T, T_s)
        T = max(time)

        attack_time = n_sampl//2 # n_sampl//3 * 2
        time_change=n_sampl//4

        qd = np.round(rng.standard_normal(size=(4,n_sampl))*noise_sigma*active_noise, 4)

        if model_file_name is None:

            simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                    S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd, cyberattack_detector=cyberattack_detector
                                    # , noise_sigma, e_sigma
                                    )

            hi, yi, zi, qi, ei, h_modeli, attack_signali = simulation.run(h0,
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
                                        time_change=time_change)
            print(np.shape(hi))
            print(np.shape(attack_signali))
            print(np.shape(ei))
            df = pd.DataFrame(np.concatenate((hi, yi, zi, qi, ei, h_modeli, attack_signali.T), axis=0), index= ['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4', 'z1', 'z2', 'q_A', 'q_B', 'e1', 'e2', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred', 'attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']).T
            df.to_csv(f"{result_path}/model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
        else:
            df = pd.read_csv(f"{result_path}/{model_file_name}", sep=';')
            hi = df[['x1', 'x2', 'x3', 'x4']].T.values
            yi = df[['y1', 'y2', 'y3', 'y4']].T.values
            zi = df[['z1', 'z2']].T.values
            qi = df[['q_A', 'q_B']].T.values
            ei = df[['e1', 'e2']].values
            h_modeli = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
            if detection_from_file:
                attack_signali = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
                residualsi = []
            else:
                for i in range(1, np.shape(hi)[0]+1):
                    df[f'res{i}'] = rmse(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                    df[f'res{i}'] = mae(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']] = df[['res1', 'res2', 'res3', 'res4']] > cyberattack_detector.threshold
                attack_signali = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
                residualsi = df[['res1', 'res2', 'res3', 'res4']].T.values
                df.to_csv(f"{result_path}/model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
        
        h.append(hi)
        y.append(yi)
        z.append(zi)
        q.append(qi)
        e.append(ei)
        y.append(yi)
        h_model.append(h_modeli)
        attack_signal.append(attack_signali)
        residuals.append(residualsi)
        threshold.append(thresholdi)

    # Set global font sizes for different elements
    plt.rcParams.update({
        'axes.titlesize': 11,    # Titles of subplots
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

    plot_path = Path(__file__).parent.parent / "plots/art"
    
    result_path = Path(__file__).parent.parent / "results"

    attack_binary = np.hstack((np.zeros(attack_time+1), np.ones(T-attack_time)))
    change_binary = np.hstack((np.zeros(time_change+1), np.ones(T-time_change)))
    
    def calc_NRMSE(true, predict, y_name_idx):
        RMSE = metrics.root_mean_squared_error(true, predict)    
        return RMSE/(h_max[0][y_name_idx] - h_min[0][y_name_idx])

    if (attack_scenario is None) and (variability==False):

        fig2 = plt.figure(figsize=(13, 2))
        fig2.suptitle("Stan normalny procesu")

        # ax1 = plt.subplot(3, 1, 1, sharex=None)
        # ax1.plot(time, q[0][0], label='$q_A$')
        # ax1.plot(time, q[0][1], label='$q_B$')
        # # plt.axhline(y=qa_max, color='black', linestyle='--', label='Pompa A max')
        # # plt.axhline(y=qb_max, color='grey', linestyle='--', label='Pompa B max')
        # ax1.set_ylabel('$q [cm^3/s]$')
        # ax1.set_title('Przepływ pomp')
        # ax1.legend(loc='center left')
        # ax1.grid()

        # ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        # ax2.plot(time, z[0][0], label='$h_1$')
        # # TODO: do wyrzucenia???
        # ax2.plot(time, z[0][1], label='$h_2$')
        # print(np.hstack((time, [max(time)+1])))
        # ax2.step(time, SP_h[0], label='$w_1$', linestyle='-.')
        # ax2.step(time, SP_h[1], label='$w_2$', linestyle=(0, (5, 5)))
        # ax2.set_ylabel('h [cm]')
        # ax2.set_title("Zmierzony i zadany poziom w zbiornikach")
        # ax2.legend(loc='center left')
        # ax2.grid()

        ax3 = plt.subplot(1, 1, 1)
        ax3.plot(time, z[0][num_tank], label='$h_1$')
        # h_model_i - NDArray z wynikami dla modeli 1 typu, ale różnych zbiorników
        for h_model_i, model_type in zip(h_model, model_type_tuple):
            ax3.plot(time, h_model_i[num_tank], linestyle='--', label=rf'model {model_type.upper()} $\hat{{h_1}}$')
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('h [cm]')
        # ax3.set_title("Rzeczywisty i modelowany poziom w 1. zbiorniku")
        ax3.legend(loc='center left')
        ax3.grid()

        # plt.subplots_adjust(hspace=0.5)

        if save_mode:
            plt.savefig(f"{plot_path}/SP_PV_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{active_noise}_variability_{param_name}{param_value}.png",
                        bbox_inches ='tight')
        plt.show()

        for h_i, h_model_i, model_type in zip(h, h_model, model_type_tuple):
            NRMSE = calc_NRMSE(h_i[num_tank], h_model_i[num_tank], 0)
            print(f"NRMSE {model_type.upper()}: {NRMSE:.4f}")

    else:

        result_df = pd.DataFrame(columns=['Opóźnienie [s]', 'Recall', 'FPR'])

        for model_type, attack_signal_i in zip(model_type_tuple, attack_signal):
            attack_res = pd.DataFrame()
            attack_res['true'] = attack_binary
            attack_res['pred'] = attack_signal_i[:, num_tank]
            attack_res.dropna(inplace=True)
            attack_res = attack_res.astype(int)

            recall = metrics.recall_score(attack_res['true'], attack_res['pred'])
            tn, fp, fn, tp = confusion_matrix(attack_res['true'], attack_res['pred']).ravel()
            fpr = fp / (fp + tn)

            indices = np.where((attack_binary[:-1] == 0) & (attack_binary[1:] == 1))[0]
            attack_time = indices[0]
            print(f"{attack_time=}")
            indices = np.where((attack_signal_i[:-1] == 0) & (attack_signal_i[1:] == 1))[0]
            print(indices)
            indices = indices[indices>attack_time]
            print(indices)
            if len(indices) == 0:
                attack_time_delay = None
            else:
                detected_attack_time = indices[0]
                attack_time_delay = detected_attack_time - attack_time
            print(attack_time_delay)

            result_df.loc[model_type.upper()] = [attack_time_delay, round(recall, 4), round(fpr, 4)]
            result_df.index.name = 'Model'

        result_df.to_excel(f"{result_path}/result_df_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{active_noise}_variability_{param_name}{param_value}.xlsx")
        print(result_df)

        fig2=plt.figure(figsize=(6, 6))
        # fig2.suptitle(f'Poziom wody w 2 zbiornikach {title_part} w trybie {title_recursion}')
        fig2.suptitle(f'Cyberatak - scenariusz {attack_scenario+1}.\nOkno czasowe o dł. {window_detection}')
        for i, (zi, hi, h_modeli, model_type, attack_signali) in enumerate(zip(z, h, h_model, model_type_tuple, attack_signal)):
            ax1 = plt.subplot(3, 1, i+1)
            ax1.plot(time, zi[num_tank], label=f'pomiar $h_{num_tank+1}$')
            ax1.plot(time, hi[num_tank], linestyle='-.', label=f'rzecz. $h_{num_tank+1}$')
            ax1.plot(time, h_modeli[num_tank], linestyle='--', label=rf'$\hat{{h_{num_tank+1}}}$')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel(f'$h_{num_tank+1} [cm]$')
            ax1.set_title(f"{model_dict[model_type]}")
            ax1.grid()
            # Add secondary y-axis to the first subplot
            ax1_secondary = ax1.twinx()
            ax1_secondary.fill_between(time, attack_signali[:, num_tank].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')
            if attack_scenario is not None:
                ax1_secondary.plot(time, attack_binary, color='red', linestyle='--', label='cyberatak')
            if variability==True:
                ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle='--', label='zmiana param')
            ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
            # ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
            # set y-axis to only show integer values
            ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1_secondary.legend(loc='best', bbox_to_anchor=(0.8, 0.1, 0.2, 0.8))
            ax1.legend(loc='best', bbox_to_anchor=(0, 0, 0.2, 1.0))

        plt.subplots_adjust(hspace=0.6)

        if save_mode:
            plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{active_noise}_variability_{param_name}{param_value}.png",
                        bbox_inches ='tight')
        plt.show()

if __name__ == "__main__":
    main_function()