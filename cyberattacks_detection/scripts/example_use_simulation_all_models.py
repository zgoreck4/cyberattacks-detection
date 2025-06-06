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
    model_type_tuple=(
        'lr', 'elm', 'rbf', 'gru', 'lstm', 'lstm-mlp',
    ),
    threshold_method_list=[('percentile', 99)],
    recursion_mode_list=[True],
    window_detection_list=[20],
    num_tank_list=[0],
    attack_scenario_list=[0, 1, 2, 3],
    variability: bool = False,
    param_name: str = 'a',
    param_value=np.array([1.2, 1.51, 0.927, 0.882])
) -> None:
    """
    Main function to simulate and detect cyberattacks in a system of four tanks.

    This function simulates tank levels under normal and attack scenarios, calculates residuals, 
    and uses models to predict values and detect anomalies based on defined thresholds.

    Parameters
    ----------
    save_mode : bool, optional
        Whether to save the results to files (default is False).
    close_loop : bool, optional
        Whether to use closed-loop simulation (default is True).
    simulate_from_file : bool, optional
        Whether to load simulation data from a file (default is True).
    detection_from_file : bool, optional
        Whether to load cyberatack detection from a file (default is True).
    normal_trajectories_from_file : bool, optional
        Whether to load normal trajectories from a file (default is True).
    attack_value : float, optional
        The magnitude of the cyberattack (default is 0.05).
    tau_y_ca : int, optional
        The time constant for the attack detection (default is 50).
    active_noise : bool, optional
        Whether to include noise in the simulation (default is True).
    noise_sigma : float, optional
        The standard deviation of the noise (default is 0.15).
    residual_calc_func : str, optional
        The function used to calculate the residuals ('rmse' or 'mae', default is 'rmse').
    model_type_tuple : tuple of str, optional
        The types of models to use for prediction (default includes various model types).
    threshold_method_list : list of tuples, optional
        The threshold methods to use (default includes 'percentile' method).
    recursion_mode_list : list of bool, optional
        Whether to use recursion mode (default is True).
    window_detection_list : list of int, optional
        The window size for detection (default is 20).
    num_tank_list : list of int, optional
        The tank numbers to simulate (default is the first tank).
    attack_scenario_list : list of int, optional
        The attack scenarios to simulate (default includes scenarios 0 to 3).
    variability : bool, optional
        Whether to include variability in the simulation (default is False).
    param_name : str, optional
        The parameter name to vary (default is 'a').
    param_value : np.ndarray, optional
        The parameter values to vary (default is a fixed array).

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

    # Your logic continues here...
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
    print(f"  model_type_tuple = {model_type_tuple}")
    print(f"  threshold_method_list = {threshold_method_list}")
    print(f"  recursion_mode_list = {recursion_mode_list}")
    print(f"  window_detection_list = {window_detection_list}")
    print(f"  num_tank_list = {num_tank_list}")
    print(f"  attack_scenario_list = {attack_scenario_list}")

    kwargs = {'n_std': 3, 'percentile': 99}

    for window_detection, num_tank, recursion_mode, (threshold_method, value), attack_scenario in tqdm(product(window_detection_list, num_tank_list, recursion_mode_list, threshold_method_list, attack_scenario_list)):
        """
        Loop over different configurations to simulate tank behavior and detect cyberattacks.

        Parameters
        ----------
        window_detection : int
            The detection window size for residual calculation.
        num_tank : int
            The index of the tank to simulate.
        recursion_mode : bool
            Whether to use recursion in the simulation.
        threshold_method : str
            The method for threshold calculation ('percentile' or 'z-score').
        value : float
            The value associated with the threshold method (percentile or z-score).
        attack_scenario : int
            The attack scenario to simulate.

        Returns
        -------
        None
        """
        # Update threshold method parameters based on the configuration
        print(f"\nthreshold_met={threshold_method}")
        print(f"value={value}")
        if threshold_method == 'percentile':
            kwargs['percentile'] = value
        elif threshold_method == 'z-score':
            kwargs['n_std'] = value
        threshold_method_dict = {'z-score': f'z-score (z={kwargs["n_std"]})', 'percentile': f'oparta na percent. ({kwargs["percentile"]}%)'}
        print(f"{recursion_mode=}")
        print(f"{window_detection=}")
        print(f"{num_tank=}")
        print(f"{attack_scenario=}")

        # Initialize empty lists to store simulation results
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

        # Iterate over model types to load corresponding models and run simulations
        for model_type in model_type_tuple:
            print(model_type)

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
            if (attack_scenario is not None and model_normal_file_name is None) or (model_file_name is None):
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
            if ((attack_scenario is not None) or (variability == True)):
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
                thresholdi = np.array(cyberattack_detector.threshold)

            else:
                cyberattack_detector = None

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

                hi, yi, zi, qi, ei, h_modeli, attack_signali = simulation.run(h0,
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
                df = pd.DataFrame(np.concatenate((hi, yi, zi, qi, ei, h_modeli, attack_signali.T), axis=0),
                                index=['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4', 'z1', 'z2', 'q_A', 'q_B', 'e1', 'e2', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred', 'attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']).T
                for i in range(1, np.shape(hi)[0] + 1):
                    if residual_calc_func == 'rmse':
                        df[f'res{i}'] = rmse(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                    elif residual_calc_func == 'mae':
                        df[f'res{i}'] = mae(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                    else:
                        print("Niepoprawna nazwa funkcji do wyliczania wskaźników diagnostycznych")
                try:
                    residualsi = df[['res1', 'res2', 'res3', 'res4']].T.values
                except:
                    residualsi = []

                # Save results if needed
                if save_mode:
                    df.to_csv(f"{result_path}/model_{model_type}_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}.csv", sep=';', index=False)
            else:
                print(f"WCZYTYWANIE DANYCH Z PLIKÓW ZE SCENARIUSZA {attack_scenario}")
                df = pd.read_csv(f"{result_path}/{model_file_name}", sep=';')
                hi = df[['x1', 'x2', 'x3', 'x4']].T.values
                yi = df[['y1', 'y2', 'y3', 'y4']].T.values
                zi = df[['z1', 'z2']].T.values
                qi = df[['q_A', 'q_B']].T.values
                ei = df[['e1', 'e2']].values
                h_modeli = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
                if detection_from_file:
                    print(f"DETEKCJA Z PLIKU ZE SCENARIUSZA {attack_scenario}")
                    attack_signali = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
                    residualsi = df[['res1', 'res2', 'res3', 'res4']].T.values
                else:
                    print("ZACIĄGAMY PRZEWIDYWANIA MODELI Z PLIKÓW, ALE WYLICZAMY WARTOŚCI GRANICZNE I WYKRYWAMY CYBERATAKI NA PODSTAWIE AKTUALNYCH DANYCH") 
                    for i in range(1, np.shape(hi)[0] + 1):
                        if residual_calc_func == 'rmse':
                            df[f'res{i}'] = rmse(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                        elif residual_calc_func == 'mae':
                            df[f'res{i}'] = mae(df[f'y{i}'], df[f'x{i}_pred'], window_detection)
                        else:
                            print("Niepoprawna nazwa funkcji do wyliczania wskaźników diagnostycznych")
                    df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']] = df[['res1', 'res2', 'res3', 'res4']] > cyberattack_detector.threshold
                    attack_signali = df[['attack_signal1', 'attack_signal2', 'attack_signal3', 'attack_signal4']].values
                    residualsi = df[['res1', 'res2', 'res3', 'res4']].T.values
                    if save_mode:
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

        # Visualization and results saving
        if (attack_scenario is None) and (variability == False):

            fig2 = plt.figure(figsize=(13, 2))
            fig2.suptitle("Stan normalny procesu")

            # Plot the real and predicted tank height for the selected tank
            ax3 = plt.subplot(1, 1, 1)
            ax3.plot(time, z[0][num_tank], label='$h_1$')
            for j, (h_model_i, model_type) in enumerate(zip(h_model, model_type_tuple)):
                ax3.plot(time, h_model_i[num_tank], linestyle=line_styles[j], label=rf'model {model_type.upper()} $\hat{{h}}_1$')
            ax3.set_xlabel('t [s]')
            ax3.set_ylabel('h [cm]')
            ax3.legend(loc='center left')
            ax3.grid()

            # Save the plot if save_mode is enabled
            if save_mode:
                plt.savefig(f"{plot_path}/SP_PV_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}_variability_{param_name}{param_value}.pdf", bbox_inches='tight')
            plt.show()

            # Calculate and print the NRMSE for each model
            for h_i, h_model_i, model_type in zip(h, h_model, model_type_tuple):
                NRMSE = calc_NRMSE(h_i[num_tank], h_model_i[num_tank], 0)
                print(f"NRMSE {model_type.upper()}: {NRMSE:.4f}")

        else:
            # Create a DataFrame for storing attack detection results
            result_df = pd.DataFrame(columns=['Opóźnienie [s]', 'Recall', 'FPR', 'num_active_alarm'])

            # Evaluate each model's performance in detecting the attack
            for model_type, attack_signal_i in zip(model_type_tuple, attack_signal):
                attack_signal_i_1tank = attack_signal_i[:, num_tank]
                attack_res = pd.DataFrame()
                attack_res['true'] = attack_binary
                attack_res['pred'] = attack_signal_i_1tank
                attack_res.dropna(inplace=True)
                attack_res = attack_res.astype(int)

                # Compute recall and false positive rate (FPR)
                recall = metrics.recall_score(attack_res['true'], attack_res['pred'])
                tn, fp, fn, tp = confusion_matrix(attack_res['true'], attack_res['pred']).ravel()
                fpr = fp / (fp + tn)

                # Find the attack detection time and the number of active alarms
                indices = np.where((attack_binary[:-1] == 0) & (attack_binary[1:] == 1))[0]
                attack_time = indices[0]
                indices = np.where((attack_signal_i_1tank[:-1] == 0) & (attack_signal_i_1tank[1:] == 1))[0]
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
            fig2 = plt.figure(figsize=(6, 10))
            if attack_scenario is not None:
                fig2.suptitle(f'Cyberatak na zbiornik {num_tank + 1}. - scenariusz {attack_scenario + 1}.\nPrzebiegi rzeczywiste, pomiarowe i modelowane dla różnych modeli.')
            else:
                fig2.suptitle(f'Zmiana wartości {param_name} na {param_value}')

            legend_handles = []  # To store handles for figlegend
            legend_labels = []  # To store labels for figlegend

            # Plot results for each model
            for i, (zi, hi, h_modeli, model_type, attack_signali) in enumerate(zip(z, h, h_model, model_type_tuple, attack_signal)):
                ax1 = plt.subplot(len(model_type_tuple), 1, i + 1)

                line1, = ax1.plot(time, zi[num_tank], label=f'pomiar $h_{num_tank+1}$')
                line2, = ax1.plot(time, hi[num_tank], linestyle='-.', label=f'rzecz. $h_{num_tank+1}$')
                line3, = ax1.plot(time, h_modeli[num_tank], linestyle='--', label=rf'$\hat{{h}}_{num_tank+1}$')

                ax1.set_ylabel(f'$h_{num_tank+1} [cm]$')
                ax1.set_title(f"{model_dict[model_type]}")
                ax1.grid()

                ax1_secondary = ax1.twinx()
                fill1 = ax1_secondary.fill_between(time, attack_signali[:, num_tank].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')

                if attack_scenario is not None:
                    line4, = ax1_secondary.plot(time, attack_binary, color='red', linestyle=(5, (10, 3)), label='cyberatak')
                else:
                    line4 = None  # Placeholder to avoid errors

                if variability:
                    line5, = ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle=(0, (5, 5)), label='zmiana param')
                else:
                    line5 = None  # Placeholder

                ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
                ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))

                # Collect legend handles & labels only from the first subplot
                if i == 0:
                    legend_handles.extend([line1, line2, line3, fill1, line4, line5])
                    legend_labels.extend([h.get_label() for h in legend_handles if h is not None])  # Filter out None values

            ax1.set_xlabel('t [s]')
            # Add a single figure-wide legend
            plt.figlegend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=5)
            plt.tight_layout(rect=[0, 0.06, 1, 0.99])  # Adjust for the title
            plt.subplots_adjust(hspace=0.6)  # Add vertical space between subplots

            # Save the plot if save_mode is enabled
            if save_mode:
                plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}_variability_{param_name}{param_value}.pdf", bbox_inches='tight')
            plt.show()

            # Plot residuals if any are available
            if len(residuals) != 0:

                fig3 = plt.figure(figsize=(6, 10))
                fig3.suptitle(f'Cyberatak - scenariusz {attack_scenario + 1}.\nWskaźniki diagnostyczne dla różnych modeli.')

                legend_handles = []  # To store handles for figlegend
                legend_labels = []  # To store labels for figlegend

                # Plot residuals for each model
                for i, (residual_1model, model_type, attack_signali, thresholdi) in enumerate(zip(residuals, model_type_tuple, attack_signal, threshold)):
                    ax1 = plt.subplot(len(model_type_tuple), 1, i + 1)

                    line1, = ax1.plot(time, residual_1model[num_tank, :], label=f'${residual_calc_func.upper()}_{num_tank + 1}$')

                    line2 = plt.axhline(y=thresholdi[num_tank], color='tab:gray', linestyle='--', label=f'$h_{{max{i + 1}}}$')

                    ax1.set_ylabel(f'${residual_calc_func.upper()}_{num_tank + 1} [cm]$')
                    ax1.set_title(f"{model_dict[model_type]}")
                    ax1.grid()

                    ax1_secondary = ax1.twinx()
                    fill1 = ax1_secondary.fill_between(time, attack_signali[:, num_tank].astype(float), color='r', alpha=0.2, label=f'wykryty cyberatak')

                    if attack_scenario is not None:
                        line4, = ax1_secondary.plot(time, attack_binary, color='red', linestyle=(5, (10, 3)), linewidth=1, label='cyberatak')
                    else:
                        line4 = None  # Placeholder to avoid errors

                    if variability:
                        line5, = ax1_secondary.plot(time, change_binary, color='tab:pink', linestyle=(0, (5, 5)), label='zmiana param')
                    else:
                        line5 = None  # Placeholder

                    ax1_secondary.set_ylabel('Sygnał binarny\ncyberataku')
                    ax1_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))

                    # Collect legend handles & labels only from the first subplot
                    if i == 0:
                        legend_handles.extend([line1, line2, fill1, line4, line5])
                        legend_labels.extend([h.get_label() for h in legend_handles if h is not None])  # Filter out None values

                ax1.set_xlabel('t [s]')
                # Add a single figure-wide legend
                plt.figlegend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=5)
                plt.tight_layout(rect=[0, 0.06, 1, 0.99])  # Adjust for the title
                plt.subplots_adjust(hspace=0.6)  # Add vertical space between subplots

                # Save the plot if save_mode is enabled
                if save_mode:
                    plt.savefig(f"{plot_path}/residuals_rec_{recursion_mode}_att{attack_scenario}_tank{num_tank}_value{attack_value}_tau_y{tau_y_ca}_window{window_detection}_met_{threshold_method}_res_calc_{residual_calc_func}_nstd{kwargs['n_std']}_perc{kwargs['percentile']}_noise_{noise_sigma}_seed{seed}_variability_{param_name}{param_value}.pdf", bbox_inches='tight')
                plt.show()
                

if __name__ == "__main__":
    main_function()