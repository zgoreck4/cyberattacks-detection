import numpy as np
import matplotlib.pyplot as plt
import cycler
from pathlib import Path
from ..simulation import Simulation
import pickle
from ..models import ELM, RBFNN, MinMaxScalerLayer
import keras
from ..detection import CyberattackDetector
from matplotlib.ticker import MaxNLocator
import re
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
from datetime import datetime

def main_function() -> None:

    save_mode = True
    close_loop = True
    recursion_mode = True

    active_noise = True # wartość False wyłącza zakłócenia, wartość True włącza
    noise_sigma = 0.15 # 0.1
    if not active_noise:
        noise_sigma = 0

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
    Ti = 15
    Td = 0

    # cm
    step_dur = 3000/5 # 200

    SP_h = None
    q = None
    SP_h1 = np.array([h0[0], h0[0], 50, 50, 80, 80, 100, 100, 90, 90, 40])
    SP_h2 = np.array([h0[1], 55,   55, 70, 70, 95, 95, 105, 105, 60, 60])
    SP_h = np.vstack((SP_h1, SP_h2))
    SP_h = np.repeat(SP_h, step_dur, axis=1)

    n_sampl = np.shape(SP_h)[1]
    qd = np.round(np.random.randn(4,n_sampl)*noise_sigma*active_noise, 4)

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

    plot_path = Path(__file__).parent.parent / "plots/v4" 
    result_path = Path(__file__).parent.parent / "results/v3"
    model_path = Path(__file__).parent.parent / "saved_models"

    def calc_NRMSE(true, predict, y_name_idx):
        RMSE = metrics.root_mean_squared_error(true, predict)  
        return RMSE/(h_max[y_name_idx][0] - h_min[y_name_idx][0])

    # h = []
    y = []
    z = []
    q = []
    e = []
    h_model = []
    attack_signal = []

    model_type_tuple = ('lr', 'elm', 'rbf', 'gru', 'lstm', 
                        'lstm-mlp',)
    model_file_names = (f'model_validate_lr_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            f'model_validate_elm_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            f'model_validate_rbf_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            f'model_validate_gru_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            f'model_validate_lstm_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            f'model_validate_lstm-mlp_model_names_[]_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv',
                            )
    # model_file_names = (None, None, None, None, None, None)
    model_dict = {'lr':'Regresja liniowa', 'elm': 'Sieć ELM', 'rbf': 'Sieć RBF', 'lstm': 'Sieć LSTM', 'gru': 'Sieć GRU', 'lstm-mlp': 'Sieć LSTM-MLP'}
    saved_models_names1 = []
    result_df = pd.DataFrame(columns=['Model', 'Zbiornik', 'NRMSE', 'r2 score', 'saved_models_names1'])

    for model_type, model_file_name in zip(model_type_tuple, model_file_names):
        print(model_type)

        if model_type == 'lr':
            if model_file_name is None:
                model1 = pickle.load(open(f"{model_path}/lr_x1.sav", 'rb'))
                model2 = pickle.load(open(f"{model_path}/lr_x2.sav", 'rb'))
                model3 = pickle.load(open(f"{model_path}/lr_x3.sav", 'rb'))
                model4 = pickle.load(open(f"{model_path}/lr_x4.sav", 'rb'))
                model_list = [model1, model2, model3, model4]
        elif model_type == 'elm':
            if model_file_name is None:
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
            if model_file_name is None:
                model1 = RBFNN(None)
                model1.load_model(f"{model_path}/rbf_x1_v0.npz")
                model2 = RBFNN(None)
                model2.load_model(f"{model_path}/rbf_x2.npz")
                model3 = RBFNN(None)
                model3.load_model(f"{model_path}/rbf_x3.npz")
                model4 = RBFNN(None)
                model4.load_model(f"{model_path}/rbf_x4.npz")
                model_list = [model1, model2, model3, model4]
        elif model_type == 'lstm':
            if model_file_name is None:
                model1 = keras.models.load_model(f"{model_path}/lstm_x1.keras")
                model2 = keras.models.load_model(f"{model_path}/lstm_x2.keras")
                model3 = keras.models.load_model(f"{model_path}/lstm_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/lstm_x4.keras")
                print("Wczytano wszystkie modele :)")
                model_list = [model1, model2, model3, model4]
        elif model_type == 'gru':
            if model_file_name is None:            
                model1 = keras.models.load_model(f"{model_path}/gru_x1.keras")
                model2 = keras.models.load_model(f"{model_path}/gru_x2.keras")
                model3 = keras.models.load_model(f"{model_path}/gru_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/gru_x4.keras")
                print("Wczytano wszystkie modele :)")
                model_list = [model1, model2, model3, model4]
        elif model_type == 'lstm-mlp':
            if model_file_name is None:
                model1 = keras.models.load_model(f"{model_path}/lstm_mlp_x1_statespace.keras")
                model2 = keras.models.load_model(f"{model_path}/lstm_mlp_x2_statespace.keras")
                model3 = keras.models.load_model(f"{model_path}/lstm_mlp_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/lstm_mlp_x4.keras")
                print("Wczytano wszystkie modele :)")
                model_list = [model1, model2, model3, model4]
        else:
            model_list = None

        T_s = 1
        T = n_sampl // T_s # TODO: sprawdzić działanie jeżeli n_sampl nie dzieli się całkowicie przez T_s
        time = np.arange(0, T, T_s)
        T = max(time)
        if model_file_name is None:
            simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                    S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd
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
            print(np.shape(h))
            print(np.shape(q))
            print(np.shape(h_model1))
            h_model.append(h_model1)
            df = pd.DataFrame(np.concatenate((q, h, h_model1), axis=0), index= ['q_A', 'q_B', 'x1', 'x2', 'x3', 'x4', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']).T
            if save_mode:
                df.to_csv(f"{result_path}/model_validate_{model_type}_model_names_{str(saved_models_names1)}_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv", sep=';', index=False)
        
        else:
            df = pd.read_csv(f"{result_path}/{model_file_name}", sep=';')
            h = df[['x1', 'x2', 'x3', 'x4']].T.values
            q = df[['q_A', 'q_B']].T.values
            print(np.shape(q))
            h_model1 = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
            h_model.append(h_model1)

        for i, h_i in enumerate(h):
            NRMSE = calc_NRMSE(h_i, h_model1[i], i)
            r2 = metrics.r2_score(h_i, h_model1[i])
            result_df.loc[len(result_df)] = [model_type.upper(), i+1, NRMSE, r2, saved_models_names1]

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_mode:
        result_df.to_excel(f"{result_path}/results_SPh_var1_noise_{noise_sigma}_{current_datetime}.xlsx", index=False)
    
    h_model = np.array(h_model)

    print(np.shape(time))
    print(np.shape(q))
    print(q[0])

    # Set global font sizes for different elements
    plt.rcParams.update({
        'axes.titlesize': 9,    # Titles of subplots
        'axes.labelsize': 8,     # Labels for axes
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
        'xtick.labelsize': 8,    # X-axis tick labels
        'ytick.labelsize': 8,    # Y-axis tick labels
        'legend.fontsize': 7,     # Legend font size
        'figure.titlesize': 10    # Overall figure title size (if used)
    })

    fig=plt.figure(figsize=(6, 10))
    plt.subplot(3, 1, 1)
    plt.grid()
    plt.axhline(y=h_max[0], color='black', linestyle='--', label='$h_{max}$')
    plt.axhline(y=h_min[0], color='black', linestyle='--', label='$h_{min}$')
    plt.plot(time, h[0], label='$h_1$', linestyle='-')
    plt.plot(time, h[1], label='$h_2$', linestyle='-.')
    plt.plot(time, h[2], label='$h_3$', linestyle=(5, (10, 3)))
    plt.plot(time, h[3], label='$h_4$', linestyle=':')
    # plt.step(time, SP_h[0], label='$w_1$', linestyle='-.')
    # plt.step(time, SP_h[1], label='$w_2$', linestyle=(0, (5, 5)))
    plt.ylabel('h [cm]')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.grid()
    plt.axhline(y=qa_max, color='black', linestyle='--', label='$q_{Amax}$')
    plt.plot(time, q[0], label='$q_A$')
    plt.ylabel('$q_A [cm^3/s]$')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.grid()
    plt.axhline(y=qb_max, color='black', linestyle='--', label='$q_{Bmax}$')
    plt.plot(time, q[1], label='$q_A$')
    plt.xlabel('t [s]')
    plt.ylabel('$q_B [cm^3/s]$')
    plt.legend()

    plt.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"Dane testowe")

    print(f"{plot_path}/data_cl_{active_noise}_testowe_v5.png")
    plt.savefig(f"{plot_path}/data_cl_{active_noise}_testowe_v5.pdf", bbox_inches='tight')
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
            r2 = metrics.r2_score(h_i, h_model_model[i])
            print(f"r2 score {model_type.upper()}: {r2:.4f}")

    if recursion_mode:
        zoom_coord_list = [[3850, 4200, 97.5, 105], [3850, 4200, 89, 99], [3850, 4200, 100.5, 110.5], [3850, 4200, 86, 95]]
    else:
        zoom_coord_list = [[3850, 4200, 97.5, 103], [3850, 4200, 92, 97.5], [3850, 4200, 100.5, 109], [3850, 4200, 86, 95]]
    axins_v = [0.55, 0.015, 0.16, 0.34]
    plt.figure(figsize=(6, 10))
    plt.suptitle("Poziom rzeczywisty i przewidywany w stanie normalnym")
    legend_handles = []  # To store handles for figlegend
    legend_labels = []  # To store labels for figlegend
    line_styles = ['--', '-.', (5, (10, 3)), (0, (5, 5)), (0, (3, 5, 1, 5)), ':', (0, (3, 1, 1, 1, 1, 1))]
    for i, (h_i, zoom_coord) in enumerate(zip(h, zoom_coord_list)):
        ax1 = plt.subplot(4, 1, i+1)
        line1, = ax1.plot(time, h_i, label=f'pomiar $h_n$')
        if i == 0:
            legend_handles.extend([line1])
        for j, (h_model_model, model_type) in enumerate(zip(h_model, model_type_tuple)):
            line2, = ax1.plot(time, h_model_model[i, :], linestyle=line_styles[j], label=rf'model {model_type.upper()} $\hat{{h_n}}$')
            if i == 0:
                legend_handles.extend([line2])
        ax1.set_ylabel(f'$h_{i+1} [cm]$')
        ax1.set_title(f"{i+1}. zbiornik")
        ax1.grid()

        # add zoom
        x1, x2, y1, y2 = zoom_coord # fragment, który chcemy powiększyć
        axins = ax1.inset_axes(axins_v) # gdzie wstawić (x, y, wysokość, szerokość) - zakresy <0, 1>
        axins.plot(time, h_i)
        for j, (h_model_model, model_type) in enumerate(zip(h_model, model_type_tuple)):
            axins.plot(time, h_model_model[i, :], linestyle=line_styles[j])

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.tick_params(bottom = False)
        axins.set_yticklabels([])
        axins.tick_params(left = False)
        ax1.indicate_inset_zoom(axins, edgecolor="black")

        if i == 0:
            legend_labels.extend([h.get_label() for h in legend_handles if h is not None])  # Filter out None values

    # Add a single figure-wide legend
    plt.figlegend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=4)
    ax1.set_xlabel('t [s]')
    # Adjust layout for spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.99])  # Adjust for the title
    plt.subplots_adjust(hspace=0.4)  # Add vertical space between subplots

    if save_mode:
        plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_noise_{active_noise}_SPh_1_noise_{active_noise}_{current_datetime}.pdf",
                    bbox_inches ='tight')
    
    plt.show()

if __name__ == "__main__":
    main_function()