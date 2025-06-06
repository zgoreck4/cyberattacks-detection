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
from itertools import product
from tqdm import tqdm

# Set random seed for reproducibility
seed = 32
rng = np.random.default_rng(seed)

def rmse(y, y_hat, window):
    """
    Compute the Root Mean Squared Error (RMSE) over a sliding window.

    Parameters
    ----------
    y : pd.Series
        The true values.
    y_hat : pd.Series
        The predicted values.
    window : int
        The size of the rolling window.

    Returns
    -------
    pd.Series
        RMSE values over the sliding window.
    """
    se = ((y - y_hat) ** 2)
    return se.rolling(window, min_periods=1).mean() ** 0.5

def mae(y, y_hat, window):
    """
    Compute the Mean Absolute Error (MAE) over a sliding window.

    Parameters
    ----------
    y : pd.Series
        The true values.
    y_hat : pd.Series
        The predicted values.
    window : int
        The size of the rolling window.

    Returns
    -------
    pd.Series
        MAE values over the sliding window.
    """
    se = abs(y - y_hat)
    return se.rolling(window, min_periods=1).mean()

def main_function(
    save_mode: bool = False,
    close_loop: bool = True,
    simulate_from_file: bool = True,
    detection_from_file: bool = True, # If simulate_from_file=False, then detection_from_file will be False
    normal_trajectories_from_file: bool = True,
    attack_value: float = 0.05,
    tau_y_ca: int = 50,
    active_noise: bool = True,
    noise_sigma: float = 0.15,
    residual_calc_func: str = 'rmse',
    model_type='lstm-mlp',
    threshold_method='percentile',
    threshold_value=99,
    recursion_mode=True,
    window_detection=20,
    num_tank=0,
    attack_scenario=0,
    variability: bool = False,
    param_name: str = 'a',
    param_value=np.array([1.2, 1.51, 0.927, 0.882])
) -> None:
    """
    Main function to simulate and detect cyberattacks in a four-tank system.

    This function performs simulation of tank behavior under normal or attack conditions,
    applies trained models to estimate system outputs, computes residuals, and detects anomalies
    such as cyberattacks using a selected detection method.

    Parameters
    ----------
    save_mode : bool, optional
        If True, saves simulation and detection results to disk (default is False).
    close_loop : bool, optional
        If True, runs the simulation in closed-loop mode (default is True).
    simulate_from_file : bool, optional
        If True, loads precomputed simulation data from file (default is True).
    detection_from_file : bool, optional
        If True, loads detection results from file. Ignored if simulate_from_file is False (default is True).
    normal_trajectories_from_file : bool, optional
        If True, loads nominal (non-attack) system trajectories from file (default is True).
    attack_value : float, optional
        Amplitude of the injected cyberattack signal (default is 0.05).
    tau_y_ca : int, optional
        Time window (in samples) used in causal detection logic (default is 50).
    active_noise : bool, optional
        If True, Gaussian noise is added to sensor measurements during simulation (default is True).
    noise_sigma : float, optional
        Standard deviation of the measurement noise (default is 0.15).
    residual_calc_func : str, optional
        Method for residual calculation, either 'rmse' (root mean square error) or 'mae' (mean absolute error) (default is 'rmse').
    model_type : str, optional
        Type of model used for prediction (e.g., 'lstm-mlp', 'cnn', 'gru') (default is 'lstm-mlp').
    threshold_method : str, optional
        Method for computing anomaly detection threshold, such as 'percentile' or 'std' (default is 'percentile').
    threshold_value : float, optional
        Threshold level (percentile or factor depending on method) for detecting anomalies (default is 99).
    recursion_mode : bool, optional
        If True, applies recursive prediction over time instead of single-step predictions (default is True).
    window_detection : int, optional
        Size of the sliding window for aggregating residuals in the detection module (default is 20).
    num_tank : int, optional
        Index of the tank to analyze (0-based) (default is 0).
    attack_scenario : int, optional
        Index of the cyberattack scenario to simulate (default is 0).
    variability : bool, optional
        If True, simulates system variability by modifying one of the model parameters (default is False).
    param_name : str, optional
        Name of the parameter being varied (e.g., 'a', 'gamma') in case of variability (default is 'a').
    param_value : np.ndarray, optional
        Array of new parameter values used when variability is enabled (default is [1.2, 1.51, 0.927, 0.882]).

    Returns
    -------
    None
    """
    # If noise is not active, set noise_sigma to 0
    if not active_noise:
        noise_sigma = 0

    # If variability is disabled, set param_name and param_value to None
    if not variability:
        param_name = None
        param_value = None

    # Handle cases when detection cannot be loaded from file during attack simulation
    if (not simulate_from_file) and (detection_from_file):
        print("During cyberattack simulation, detection cannot be from file - it will be calculated during simulation.")
        detection_from_file = False

    # Set up recursive mode dictionary for better readability
    recursion_mode_dict = {True: 'rekurencyjnym', False: 'bez rekurencji'}


    # Define system parameters
    tau_u = 0
    tau_y = 0

    # Initial values for the system (heights and flows)
    h0 = [65.37515073890378, 64.98201463176996, 65.90206440432354, 65.8157923349714]
    h_max = [[136], [136], [130], [130]]
    h_min = [[20], [20], [20], [20]]

    # Maximum flow rates and other system constants
    qa_max = 3260000 / 3600
    qb_max = 4000000 / 3600
    q_min = 0
    gamma_a = 0.3
    gamma_b = 0.4
    S = np.array([60, 60, 60, 60])
    a = np.array([1.31, 1.51, 0.927, 0.882])  # Outlet cross-section areas
    c = np.array([1, 1, 1, 1])

    # Control parameters
    kp = 2
    Ti = 15
    Td = 0
    step_dur = 3000 / 5  # Time step duration

    # Define setpoints for tank heights under normal conditions
    SP_h1_normal = np.array([h0[0], h0[0], 50, 50, 80, 80, 100, 100, 90, 90, 40])
    SP_h2_normal = np.array([h0[1], 55,    55, 70, 70, 95, 95, 105, 105, 60, 60])
    SP_h_normal = np.vstack((SP_h1_normal, SP_h2_normal))
    SP_h_normal = np.repeat(SP_h_normal, step_dur, axis=1)

    # Number of samples in the normal trajectory
    n_sampl_normal = SP_h_normal.shape[1]

    # Setpoints for tank heights under attack conditions
    SP_h1 = np.array([h0[0], h0[0], 70, 70, 95, 95, 90])
    SP_h2 = np.array([h0[1], 50, 50, 90, 90, 105, 105])
    SP_h = np.vstack((SP_h1, SP_h2))
    SP_h = np.repeat(SP_h, step_dur, axis=1)

    # Number of samples in the attack scenario
    n_sampl = SP_h.shape[1]

    # Generate random noise for the simulation
    qd = np.round(rng.standard_normal(size=(4, n_sampl)) * noise_sigma * active_noise, 4)
    qd_normal = np.round(rng.standard_normal(size=(4, n_sampl_normal)) * noise_sigma * active_noise, 4)

    # Define paths for saving models, results, and plots
    model_path = Path(__file__).parent.parent / "saved_models"
    result_path = Path(__file__).parent.parent / "results/v3"
    plot_path = Path(__file__).parent.parent / "plots/v4"

    model_dict = {
        'lr': 'Regresja liniowa',
        'elm': 'Sieć ELM',
        'rbf': 'Sieć RBF',
        'lstm': 'Sieć LSTM',
        'gru': 'Sieć GRU',
        'lstm-mlp': 'Sieć LSTM-MLP'
    }

    print("Function executed with:")
    print(f"  save_mode = {save_mode}")
    print(f"  close_loop = {close_loop}")
    print(f"  attack_value = {attack_value}")
    print(f"  tau_y_ca = {tau_y_ca}")
    print(f"  simulate_from_file = {simulate_from_file}")
    print(f"  detection_from_file = {detection_from_file}")
    print(f"  residual_calc_func = '{residual_calc_func}'")
    print(f"  variability = {variability}")
    print(f"  param_name = {param_name}")
    print(f"  param_value = {param_value}")
    print(f"  active_noise = {active_noise}")
    print(f"  noise_sigma = {noise_sigma}")
    print(f"  model_type = '{model_type}'")
    print(f"  threshold_method = '{threshold_method}'")
    print(f"  threshold_value = {threshold_value}")
    print(f"  recursion_mode = {recursion_mode}")
    print(f"  window_detection = {window_detection}")
    print(f"  num_tank = {num_tank}")
    print(f"  attack_scenario = {attack_scenario}")

    kwargs = {'n_std': 3, 'percentile': 99}
    if threshold_method == 'percentile':
        kwargs['percentile'] = threshold_value
    elif threshold_method == 'z-score':
        kwargs['n_std'] = threshold_value

    # Initialize empty lists to store simulation results
    h = []
    y = []
    z = []
    q = []
    e = []
    h_model = []
    attack_signal = []
    residuals = []
    threshold = []

    # Set model file paths based on simulation and detection settings
    if simulate_from_file or detection_from_file:
        model_file_name = f"model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv"
    else:
        model_file_name = None
    if normal_trajectories_from_file:
        model_normal_file_name = f"model_{model_type}_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv"
    else:
        model_normal_file_name = None

    # Load or simulate model data
    if (model_normal_file_name is None) or (model_file_name is None):
        # Load models based on type (e.g., 'lr', 'elm', 'rbf', etc.)
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
            model1 = keras.models.load_model(f"{model_path}/lstm_mlp_x1_statespace.keras")
            model2 = keras.models.load_model(f"{model_path}/lstm_mlp_x2_statespace.keras")
            model3 = keras.models.load_model(f"{model_path}/lstm_mlp_x3.keras")
            model4 = keras.models.load_model(f"{model_path}/lstm_mlp_x4.keras")
            model_list = [model1, model2, model3, model4]
        else:
            model_list = None

    # Set threshold based on normal state simulation
    T_s = 1
    T = n_sampl_normal // T_s
    time = np.arange(0, T, T_s)
    T = max(time)
    if model_normal_file_name is None:
        print("SYMULACJA STANU NORMALNEGO")
        simulation_normal = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                        S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd_normal)
        h_normal, _, _, _, _, h_model_normal, _ = simulation_normal.run(h0,
                                                                        close_loop,
                                                                        model_list=model_list,
                                                                        recursion_mode=recursion_mode,
                                                                        SP_h=SP_h_normal,
                                                                        q=q,
                                                                        qa0=1630000 / 3600,
                                                                        qb0=2000000 / 3600,
                                                                        attack_scenario=None)
        df = pd.DataFrame(np.concatenate((h_normal, h_model_normal), axis=0),
                        index=['x1', 'x2', 'x3', 'x4', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']).T
        if save_mode:
            df.to_csv(f"{result_path}/model_{model_type}_normal_rec_{recursion_mode}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
    else:
        print(f"WCZYTYWANIE DANYCH Z PLIKÓW Z PRZEBIEGAMI ZMIENNYCH ZE STANU NORMALNEGO")
        df = pd.read_csv(f"{result_path}/{model_normal_file_name}", sep=';')
        h_normal = df[['x1', 'x2', 'x3', 'x4']].T.values
        h_model_normal = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values

    # Initialize the cyberattack detector
    cyberattack_detector = CyberattackDetector(window=window_detection, residual_calc_func=residual_calc_func)
    cyberattack_detector.calc_threshold(h_normal[:len(h_model_normal), :], h_model_normal, method=threshold_method, **kwargs)
    threshold = np.array(cyberattack_detector.threshold)

    # Continue simulation for attack scenarios
    T_s = 1
    T = n_sampl // T_s
    time = np.arange(0, T, T_s)
    T = max(time)

    attack_time = n_sampl // 2
    time_change = n_sampl // 4

    # Simulate or load attack scenario
    if model_file_name is None:
        print(f"SYMULACJA SCANARIUSZA {attack_scenario}")

        simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b,
                                S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd, cyberattack_detector=cyberattack_detector)

        h, y, z, q, e, h_model, attack_signal = simulation.run(h0,
                                                                    close_loop,
                                                                    model_list=model_list,
                                                                    recursion_mode=recursion_mode,
                                                                    SP_h=SP_h,
                                                                    q=q,
                                                                    qa0=1630000 / 3600,
                                                                    qb0=2000000 / 3600,
                                                                    attack_scenario=attack_scenario,
                                                                    num_tank=num_tank,
                                                                    attack_time=attack_time,
                                                                    attack_value=attack_value,
                                                                    tau_y_ca=tau_y_ca,
                                                                    variability=variability,
                                                                    param_name=param_name,
                                                                    param_value=param_value,
                                                                    time_change=time_change)
        df = pd.DataFrame(np.concatenate((h, y, z, q, e, h_model, attack_signal.T), axis=0),
                        index=['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4', 'z1', 'z2', 'q_A', 'q_B', 'e1', 'e2', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred', 'attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']).T
        for i in range(1, np.shape(h)[0] + 1):
            if residual_calc_func == 'rmse':
                df[f'res{i}'] = rmse(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
            elif residual_calc_func == 'mae':
                df[f'res{i}'] = mae(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
            else:
                print("Niepoprawna nazwa funkcji do wyliczania wskaźników diagnostycznych")
        try:
            residuals = df[['res1', 'res2', 'res3', 'res4']].T.values
        except:
            residuals = []

        # Save results if needed
        if save_mode:
            df.to_csv(f"{result_path}/model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
    else:
        print(f"WCZYTYWANIE DANYCH Z PLIKÓW ZE SCENARIUSZA {attack_scenario}")
        df = pd.read_csv(f"{result_path}/{model_file_name}", sep=';')
        h = df[['x1', 'x2', 'x3', 'x4']].T.values
        y = df[['y1', 'y2', 'y3', 'y4']].T.values
        z = df[['z1', 'z2']].T.values
        q = df[['q_A', 'q_B']].T.values
        e = df[['e1', 'e2']].values
        h_model = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
        if detection_from_file:
            print(f"DETEKCJA Z PLIKU ZE SCENARIUSZA {attack_scenario}")
            attack_signal = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
            residuals = df[['res1', 'res2', 'res3', 'res4']].T.values
        else:
            print("ZACIĄGAMY PRZEWIDYWANIA MODELI Z PLIKÓW, ALE WYLICZAMY WARTOŚCI GRANICZNE I WYKRYWAMY CYBERATAKI NA PODSTAWIE AKTUALNYCH DANYCH") 
            for i in range(1, np.shape(h)[0] + 1):
                if residual_calc_func == 'rmse':
                    df[f'res{i}'] = rmse(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                elif residual_calc_func == 'mae':
                    df[f'res{i}'] = mae(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                else:
                    print("Niepoprawna nazwa funkcji do wyliczania wskaźników diagnostycznych")
            df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']] = df[['res1', 'res2', 'res3', 'res4']] > cyberattack_detector.threshold
            attack_signal = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
            residuals = df[['res1', 'res2', 'res3', 'res4']].T.values
            if save_mode:
                df.to_csv(f"{result_path}/model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)

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

    # Define line styles for different model predictions
    line_styles = ['--', '-.', (5, (10, 3)), (0, (5, 5)), (0, (3, 5, 1, 5)), ':', (0, (3, 1, 1, 1, 1, 1))]

    # Create binary arrays for attack and change detection
    attack_binary = np.hstack((np.zeros(attack_time + 1), np.ones(T - attack_time)))
    change_binary = np.hstack((np.zeros(time_change + 1), np.ones(T - time_change)))

    def calc_NRMSE(true, predict, y_name_idx):
        """
        Calculate the Normalized Root Mean Squared Error (NRMSE).

        Parameters
        ----------
        true : np.ndarray
            The true values for the tank height.
        predict : np.ndarray
            The predicted values from the model.
        y_name_idx : int
            The index to access the maximum and minimum values for normalization.

        Returns
        -------
        float
            The NRMSE value.
        """
        RMSE = metrics.root_mean_squared_error(true, predict)
        return RMSE / (h_max[0][y_name_idx] - h_min[0][y_name_idx])

    # Create a DataFrame for storing attack detection results
    result_df = pd.DataFrame(columns=['Opóźnienie [s]', 'Recall', 'FPR', 'num_active_alarm'])

    # Evaluate model's performance in detecting the attack
    attack_signal_1tank = attack_signal[:, num_tank]
    attack_res = pd.DataFrame()
    attack_res['true'] = attack_binary
    attack_res['pred'] = attack_signal_1tank
    attack_res.dropna(inplace=True)
    attack_res = attack_res.astype(int)

    # Compute recall and false positive rate (FPR)
    recall = metrics.recall_score(attack_res['true'], attack_res['pred'])
    tn, fp, fn, tp = confusion_matrix(attack_res['true'], attack_res['pred']).ravel()
    fpr = fp / (fp + tn)

    # Find the attack detection time and the number of active alarms
    indices = np.where((attack_binary[:-1] == 0) & (attack_binary[1:] == 1))[0]
    attack_time = indices[0]
    indices = np.where((attack_signal_1tank[:-1] == 0) & (attack_signal_1tank[1:] == 1))[0]
    num_active_alarm = len(indices)
    indices = indices[indices > attack_time]
    if len(indices) == 0:
        attack_time_delay = None
    else:
        detected_attack_time = indices[0]
        attack_time_delay = detected_attack_time - attack_time

    # Store results in the DataFrame
    result_df.loc[model_type.upper()] = [attack_time_delay, round(recall, 4), round(fpr, 4), num_active_alarm]
    result_df.index.name = 'Model'

    # Add additional result columns
    result_df['recursion_mode'] = recursion_mode
    result_df['attack_scenario'] = attack_scenario
    result_df['num_tank'] = num_tank
    result_df['attack_value'] = attack_value
    result_df['tau_y_ca'] = tau_y_ca
    result_df['window_detection'] = window_detection
    result_df['threshold_method'] = threshold_method
    result_df['residual_calc_func'] = residual_calc_func
    result_df['n_std'] = kwargs['n_std']
    result_df['percentile'] = kwargs['percentile']
    result_df['noise_sigma'] = noise_sigma

    # Save results to Excel
    if save_mode:
        result_df.to_excel(f"{result_path}/result_df_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_variability_{param_name}{param_value}.xlsx")
    print(result_df)

    # Plot for attack detection results fot different models
    fig2 = plt.figure(figsize=(6, 6))
    if attack_scenario is not None:
        fig2.suptitle(f'Cyberatak na zbiornik {num_tank + 1}. - scenariusz {attack_scenario + 1}.\nModel {model_type.upper()}.')
    else:
        fig2.suptitle(f'Zmiana wartości {param_name} na {param_value}')

    # Plot results
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(time, z[num_tank], label=f'pomiar $h_{num_tank+1}$')
    ax1.plot(time, h[num_tank], linestyle='-.', label=f'rzecz. $h_{num_tank+1}$')
    ax1.plot(time, h_model[num_tank], linestyle='--', label=rf'$\hat{{h}}_{num_tank+1}$')

    ax1.set_ylabel(f'$h_{num_tank+1} [cm]$')
    ax1.set_title(f"Przebiegi zmiennej")
    ax1.grid()

    ax1_secondary = ax1.twinx()
    ax1_secondary.fill_between(time, attack_signal[:, num_tank].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')

    if attack_scenario is not None:
        ax1_secondary.plot(time, attack_binary, color='red', linestyle=(5, (10, 3)), label='cyberatak')

    if variability:
        ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle=(0, (5, 5)), label='zmiana param')

    ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
    ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Gather legend handles for the top subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_secondary.get_legend_handles_labels()

    # Combine and filter out duplicates
    all_handles_top = handles1 + handles2
    all_labels_top = labels1 + labels2
    unique_top = dict(zip(all_labels_top, all_handles_top))

    # Place the legend on the top subplot
    ax1.legend(unique_top.values(), unique_top.keys(), loc='best')


    # Plot residuals
    ax1 = plt.subplot(2, 1, 2)

    ax1.plot(time, residuals[num_tank, :], label=f'${residual_calc_func.upper()}_{num_tank + 1}$')

    plt.axhline(y=threshold[num_tank], color='tab:gray', linestyle='--', label=f'$h_{{max{num_tank + 1}}}$')

    ax1.set_ylabel(f'${residual_calc_func.upper()}_{num_tank + 1} [cm]$')
    ax1.set_title(f"Residua")
    ax1.grid()

    ax1_secondary = ax1.twinx()
    ax1_secondary.fill_between(time, attack_signal[:, num_tank].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')

    if attack_scenario is not None:
        ax1_secondary.plot(time, attack_binary, color='red', linestyle=(5, (10, 3)), linewidth=1, label='cyberatak')

    if variability:
        ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle=(0, (5, 5)), label='zmiana param')

    ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
    ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Gather legend handles for the top subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_secondary.get_legend_handles_labels()

    # Combine and filter out duplicates
    all_handles_top = handles1 + handles2
    all_labels_top = labels1 + labels2
    unique_top = dict(zip(all_labels_top, all_handles_top))

    # Place the legend on the top subplot
    ax1.legend(unique_top.values(), unique_top.keys(), loc='best')

    ax1.set_xlabel('t [s]')
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])  # Adjust for the title
    # plt.subplots_adjust(hspace=0.6)  # Add vertical space between subplots

    # Save the plot if save_mode is enabled
    if save_mode:
        plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}_variability_{param_name}{param_value}.pdf", bbox_inches='tight')
    plt.show()
                

if __name__ == "__main__":
    main_function()