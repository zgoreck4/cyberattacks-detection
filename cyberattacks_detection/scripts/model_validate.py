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
    """
    Main function to simulate the system, detect cyberattacks, and plot results.
    
    This function simulates the behavior of a system with four tanks, calculates residuals between
    predicted and actual values, detects cyberattacks, and visualizes the results. It supports different
    model types (e.g., linear regression, ELM, RBF, GRU, LSTM, LSTM-MLP) for comparison.
    
    Returns
    -------
    None
    """
    # Define configuration flags
    save_mode = True
    close_loop = True
    recursion_mode = True

    # Active noise configuration
    active_noise = True  # Set to False to disable noise, True to enable noise
    noise_sigma = 0.15  # Standard deviation of noise
    if not active_noise:
        noise_sigma = 0  # If noise is not active, set noise sigma to 0

    # System parameters
    tau_u = 0
    tau_y = 0

    # Initial tank levels
    h0 = [65.37515073890378, 64.98201463176996, 65.90206440432354, 65.8157923349714]
    h_max = [[136], [136], [130], [130]]  # Maximum tank heights
    h_min = [[20], [20], [20], [20]]     # Minimum tank heights

    # Maximum flow rates and other constants
    qa_max = 3260000 / 3600
    qb_max = 4000000 / 3600
    q_min = 0
    gamma_a = 0.3
    gamma_b = 0.4
    S = np.array([60, 60, 60, 60])
    a = np.array([1.31, 1.51, 0.927, 0.882])  # Outlet cross-sectional areas
    c = np.array([1, 1, 1, 1])

    # Controller parameters
    kp = 2
    Ti = 15
    Td = 0
    step_dur = 3000 / 5  # Time step duration in seconds

    # Setpoints for tank heights under normal conditions
    SP_h1 = np.array([h0[0], h0[0], 50, 50, 80, 80, 100, 100, 90, 90, 40])
    SP_h2 = np.array([h0[1], 55,   55, 70, 70, 95, 95, 105, 105, 60, 60])
    SP_h = np.vstack((SP_h1, SP_h2))
    SP_h = np.repeat(SP_h, step_dur, axis=1)

    # Number of samples in the simulation
    n_sampl = np.shape(SP_h)[1]
    qd = np.round(np.random.randn(4, n_sampl) * noise_sigma * active_noise, 4)

    # Set global font sizes for different plot elements
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
        'legend.fontsize': 7,    # Legend font size
        'figure.titlesize': 12   # Overall figure title size
    })

    # Define paths for saving plots and results
    plot_path = Path(__file__).parent.parent / "plots/v4"
    result_path = Path(__file__).parent.parent / "results/v3"
    model_path = Path(__file__).parent.parent / "saved_models"

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
        return RMSE / (h_max[y_name_idx][0] - h_min[y_name_idx][0])

    # Initialize containers for simulation results
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
    model_dict = {'lr':'Regresja liniowa',
                  'elm': 'Sieć ELM',
                  'rbf': 'Sieć RBF',
                  'lstm': 'Sieć LSTM',
                  'gru': 'Sieć GRU',
                  'lstm-mlp': 'Sieć LSTM-MLP'}
    saved_models_names1 = []
    # DataFrame for storing results
    result_df = pd.DataFrame(columns=['Model', 'Tank', 'NRMSE', 'R2 Score', 'Saved Models'])

    # Iterate through each model type and corresponding file name
    for model_type, model_file_name in zip(model_type_tuple, model_file_names):
        print(model_type)

        # Load models based on the type (linear regression, ELM, RBF, etc.)
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
                print("All models loaded.")
                model_list = [model1, model2, model3, model4]
        elif model_type == 'gru':
            if model_file_name is None:
                model1 = keras.models.load_model(f"{model_path}/gru_x1.keras")
                model2 = keras.models.load_model(f"{model_path}/gru_x2.keras")
                model3 = keras.models.load_model(f"{model_path}/gru_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/gru_x4.keras")
                print("All models loaded.")
                model_list = [model1, model2, model3, model4]
        elif model_type == 'lstm-mlp':
            if model_file_name is None:
                model1 = keras.models.load_model(f"{model_path}/lstm_mlp_x1_statespace.keras")
                model2 = keras.models.load_model(f"{model_path}/lstm_mlp_x2_statespace.keras")
                model3 = keras.models.load_model(f"{model_path}/lstm_mlp_x3.keras")
                model4 = keras.models.load_model(f"{model_path}/lstm_mlp_x4.keras")
                print("All models loaded.")
                model_list = [model1, model2, model3, model4]
        else:
            model_list = None

        # Perform simulation and analysis
        T_s = 1
        T = n_sampl // T_s
        time = np.arange(0, T, T_s)
        T = max(time)
        if model_file_name is None:
            simulation = Simulation(h_max, h_min, qa_max, qb_max, gamma_a, gamma_b, S, a, c, T, T_s, kp, Ti, Td, tau_u, tau_y, qd)
            h, _, _, q, _, h_model1, _ = simulation.run(h0, close_loop, model_list=model_list, recursion_mode=recursion_mode, SP_h=SP_h, q=q, qa0=1630000 / 3600, qb0=2000000 / 3600, attack_scenario=None)
            h_model.append(h_model1)
            df = pd.DataFrame(np.concatenate((q, h, h_model1), axis=0), index=['q_A', 'q_B', 'x1', 'x2', 'x3', 'x4', 'x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']).T
            if save_mode:
                df.to_csv(f"{result_path}/model_validate_{model_type}_model_names_{str(saved_models_names1)}_SPh_var1_noise_{noise_sigma}_rec_{recursion_mode}.csv", sep=';', index=False)
        
        # Process model data from file if necessary
        else:
            df = pd.read_csv(f"{result_path}/{model_file_name}", sep=';')
            h = df[['x1', 'x2', 'x3', 'x4']].T.values
            q = df[['q_A', 'q_B']].T.values
            h_model1 = df[['x1_pred', 'x2_pred', 'x3_pred', 'x4_pred']].T.values
            h_model.append(h_model1)

        # Compute NRMSE and R2 for each model
        for i, h_i in enumerate(h):
            NRMSE = calc_NRMSE(h_i, h_model1[i], i)
            r2 = metrics.r2_score(h_i, h_model1[i])
            result_df.loc[len(result_df)] = [model_type.upper(), i + 1, NRMSE, r2, saved_models_names1]

    # Save the result DataFrame to Excel
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_mode:
        result_df.to_excel(f"{result_path}/results_SPh_var1_noise_{noise_sigma}_{current_datetime}.xlsx", index=False)

    h_model = np.array(h_model)
    
    # Plotting the results
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
        'legend.fontsize': 7,    # Legend font size
        'figure.titlesize': 10   # Overall figure title size
    })

    # Set up figure for plotting
    fig = plt.figure(figsize=(6, 10))

    # First subplot: Tank levels (h1, h2, h3, h4)
    plt.subplot(3, 1, 1)
    plt.grid()  # Add grid for better readability
    plt.axhline(y=h_max[0], color='black', linestyle='--', label='$h_{max}$')  # Max tank height line
    plt.axhline(y=h_min[0], color='black', linestyle='--', label='$h_{min}$')  # Min tank height line
    plt.plot(time, h[0], label='$h_1$', linestyle='-')  # Plot actual tank height for tank 1
    plt.plot(time, h[1], label='$h_2$', linestyle='-.')  # Plot actual tank height for tank 2
    plt.plot(time, h[2], label='$h_3$', linestyle=(5, (10, 3)))  # Plot actual tank height for tank 3
    plt.plot(time, h[3], label='$h_4$', linestyle=':')  # Plot actual tank height for tank 4
    plt.ylabel('h [cm]')  # Label for y-axis (tank height)
    plt.legend()  # Add legend for the plot

    # Second subplot: Flow rate for pump A (q_A)
    plt.subplot(3, 1, 2)
    plt.grid()  # Add grid for better readability
    plt.axhline(y=qa_max, color='black', linestyle='--', label='$q_{Amax}$')  # Max flow rate line for pump A
    plt.plot(time, q[0], label='$q_A$')  # Plot flow rate for pump A
    plt.ylabel('$q_A [cm^3/s]$')  # Label for y-axis (flow rate for pump A)
    plt.legend()  # Add legend for the plot

    # Third subplot: Flow rate for pump B (q_B)
    plt.subplot(3, 1, 3)
    plt.grid()  # Add grid for better readability
    plt.axhline(y=qb_max, color='black', linestyle='--', label='$q_{Bmax}$')  # Max flow rate line for pump B
    plt.plot(time, q[1], label='$q_B$')  # Plot flow rate for pump B
    plt.xlabel('t [s]')  # Label for x-axis (time)
    plt.ylabel('$q_B [cm^3/s]$')  # Label for y-axis (flow rate for pump B)
    plt.legend()  # Add legend for the plot

    # Adjust space between subplots and set title
    plt.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"Dane testowe")  # Overall title for the plot

    # Save the figure if save_mode is True
    if save_mode:
        plt.savefig(f"{plot_path}/data_cl_{active_noise}_testowe_v5.pdf", bbox_inches='tight')
    
    plt.show()  # Display the plot

    # Function to calculate Normalized Root Mean Squared Error (NRMSE)
    def calc_NRMSE(true, predict, y_name_idx):
        """
        Calculate the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.
        
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
        RMSE = metrics.root_mean_squared_error(true, predict)  # Calculate RMSE
        return RMSE / (h_max[y_name_idx][0] - h_min[y_name_idx][0])  # Normalize by max and min tank heights

    # Iterate through each tank and calculate NRMSE and R2 score for each model
    for i, h_i in enumerate(h):
        print("-" * 15)
        print(f"Zbiornik {i + 1}.")  # Print tank number
        for h_model_model, model_type in zip(h_model, model_type_tuple):
            NRMSE = calc_NRMSE(h_i, h_model_model[i], i)  # Calculate NRMSE for each model
            print(f"NRMSE {model_type.upper()}: {NRMSE:.4f}")  # Print NRMSE value
            r2 = metrics.r2_score(h_i, h_model_model[i])  # Calculate R2 score
            print(f"r2 score {model_type.upper()}: {r2:.4f}")  # Print R2 score

    # Set zoom coordinates for the zoomed-in plots
    if recursion_mode:
        zoom_coord_list = [[3850, 4200, 97.5, 105], [3850, 4200, 89, 99], [3850, 4200, 100.5, 110.5], [3850, 4200, 86, 95]]
    else:
        zoom_coord_list = [[3850, 4200, 97.5, 103], [3850, 4200, 92, 97.5], [3850, 4200, 100.5, 109], [3850, 4200, 86, 95]]
    
    # Create the zoomed-in plots
    axins_v = [0.55, 0.015, 0.16, 0.34]
    plt.figure(figsize=(6, 10))
    plt.suptitle("Poziom rzeczywisty i przewidywany w stanie normalnym")  # Title for the zoomed-in plots
    legend_handles = []  # To store handles for the legend
    legend_labels = []  # To store labels for the legend
    line_styles = ['--', '-.', (5, (10, 3)), (0, (5, 5)), (0, (3, 5, 1, 5)), ':', (0, (3, 1, 1, 1, 1, 1))]

    # Iterate through each tank and plot the zoomed-in data
    for i, (h_i, zoom_coord) in enumerate(zip(h, zoom_coord_list)):
        ax1 = plt.subplot(4, 1, i + 1)
        line1, = ax1.plot(time, h_i, label=f'pomiar $h_n$')  # Plot the actual tank height
        if i == 0:
            legend_handles.extend([line1])
        for j, (h_model_model, model_type) in enumerate(zip(h_model, model_type_tuple)):
            line2, = ax1.plot(time, h_model_model[i, :], linestyle=line_styles[j], label=rf'model {model_type.upper()} $\hat{{h_n}}$')  # Plot the predicted tank height
            if i == 0:
                legend_handles.extend([line2])
        ax1.set_ylabel(f'$h_{i + 1} [cm]$')  # Label for y-axis
        ax1.set_title(f"{i + 1}. zbiornik")  # Title for each subplot
        ax1.grid()  # Add grid for better readability

        # Add zoom feature for the plot
        x1, x2, y1, y2 = zoom_coord  # Get the zoom region coordinates
        axins = ax1.inset_axes(axins_v)  # Create inset axes for zoom
        axins.plot(time, h_i)  # Plot zoomed data
        for j, (h_model_model, model_type) in enumerate(zip(h_model, model_type_tuple)):
            axins.plot(time, h_model_model[i, :], linestyle=line_styles[j])  # Plot zoomed predicted data

        axins.set_xlim(x1, x2)  # Set x-axis limits for zoom
        axins.set_ylim(y1, y2)  # Set y-axis limits for zoom
        axins.set_xticklabels([])  # Remove x-axis labels for the zoomed area
        axins.tick_params(bottom=False)  # Remove x-axis ticks for the zoomed area
        axins.set_yticklabels([])  # Remove y-axis labels for the zoomed area
        axins.tick_params(left=False)  # Remove y-axis ticks for the zoomed area
        ax1.indicate_inset_zoom(axins, edgecolor="black")  # Highlight the zoomed area

        if i == 0:
            legend_labels.extend([h.get_label() for h in legend_handles if h is not None])  # Filter out None values for the legend

    # Add a single figure-wide legend
    plt.figlegend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=4)
    ax1.set_xlabel('t [s]')  # Label for x-axis (time)
    
    # Adjust layout for spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.99])  # Adjust for the title
    plt.subplots_adjust(hspace=0.4)  # Add vertical space between subplots

    # Save the figure if save_mode is True
    if save_mode:
        plt.savefig(f"{plot_path}/h_rec_{recursion_mode}_noise_{active_noise}_SPh_1_noise_{active_noise}_{current_datetime}.pdf", bbox_inches='tight')
    
    plt.show()  # Display the plot

if __name__ == "__main__":
    main_function()