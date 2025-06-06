from .scripts import main_function
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cyberattack detection simulation."
    )

    # Boolean flags (store_true → default False, store_false → default True)
    parser.add_argument("--save_mode", action="store_true", help="Enable saving results.")
    parser.add_argument("--open_loop", action="store_true", help="open-loop simulation.")
    parser.add_argument("--no_simulate_from_file", action="store_true", help="Live simulation (longer).")
    parser.add_argument("--no_detection_from_file", action="store_true", help="Live cyberattack detection.")
    parser.add_argument("--no_normal_trajectories_from_file", action="store_true", help="Live simulation of the normal state.")
    parser.add_argument("--no_active_noise", action="store_true", help="Disable noise in simulation.")
    parser.add_argument("--variability", action="store_true", help="Enable variability in simulation.")

    # Regular arguments
    parser.add_argument("--attack_value", type=float, default=0.05, help="Attack value (default: 0.05).")
    parser.add_argument("--tau_y_ca", type=int, default=50, help="Time constant for attack detection (default: 50).")
    parser.add_argument("--noise_sigma", type=float, default=0.15, help="Noise standard deviation (default: 0.15).")
    parser.add_argument(
        "--residual_calc_func",
        type=str,
        choices=["rmse", "mae"],  # Dodatkowo zapewnia walidację wejścia
        default="rmse",
        help="Residual calculation method. Options: 'rmse' (Root Mean Square Error), 'mae' (Mean Absolute Error). Default: 'rmse'."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lr", "rbf", "elm", "gru", "lstm", "lstm-mlp"],
        default="lstm-mlp",
        help=(
            "Type of model to use. Options: "
            "'lr' (Linear Regression), "
            "'rbf' (Radial Basis Function), "
            "'elm' (Ectreme Learning Machine), "
            "'gru' (Gated Recurrent Unit), "
            "'lstm' (Long Short-Term Memory), "
            "'lstm-mlp' (hybrid model LSTM and MLP), "
            "Default: 'lstm-mlp'."
            )
    )

    parser.add_argument(
        "--threshold_method",
        type=str,
        choices=["percentile", "z-score", "max"],
        default="percentile",
        help=(
            "Method used to compute detection threshold. Options: "
            "'percentile' (based on quantile threshold), "
            "'z-score' (based on standard deviation), "
            "'max' (based on max value). "
            "Default: 'percentile'."
        )
    )
    parser.add_argument("--threshold_value", type=int, default=99, help="Threshold value (default: 99).")
    parser.add_argument("--recursion_mode", type=bool, default=True, help="Enable recursive model prediction (default: True).")
    parser.add_argument("--window_detection", type=int, default=20, help="Window size for detection (default: 20).")
    parser.add_argument(
        "--num_tank",
        type=int,
        choices=[0, 1],  # Dodaj więcej, np. [0, 1, 2, 3], jeśli są 4 zbiorniki
        default=0,
        help="Tank number to simulate or monitor. Possible values: 0 or 1. Default: 0."
    )
    parser.add_argument(
        "--attack_scenario",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Attack scenario index. Possible values: 0, 1, 2, 3 (default: 0)."
    )
    parser.add_argument("--param_name", type=str, default="a", help="Parameter name to vary (default: 'a').")
    parser.add_argument(
        "--param_value",
        type=float,
        nargs=4,
        default=[1.2, 1.51, 0.927, 0.882],
        help="Parameter values to vary (4 floats, default: [1.2, 1.51, 0.927, 0.882])."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Map negated flags to actual values
    close_loop = not args.open_loop
    simulate_from_file = not args.no_simulate_from_file
    detection_from_file = not args.no_detection_from_file
    normal_trajectories_from_file = not args.no_normal_trajectories_from_file
    active_noise = not args.no_active_noise

    # Call the main function with parsed arguments
    main_function(
        save_mode=args.save_mode,
        close_loop=close_loop,
        simulate_from_file=simulate_from_file,
        detection_from_file=detection_from_file,
        normal_trajectories_from_file=normal_trajectories_from_file,
        attack_value=args.attack_value,
        tau_y_ca=args.tau_y_ca,
        active_noise=active_noise,
        noise_sigma=args.noise_sigma,
        residual_calc_func=args.residual_calc_func,
        model_type=args.model_type,
        threshold_method=args.threshold_method,
        threshold_value=args.threshold_value,
        recursion_mode=args.recursion_mode,
        window_detection=args.window_detection,
        num_tank=args.num_tank,
        attack_scenario=args.attack_scenario,
        variability=args.variability,
        param_name=args.param_name,
        param_value=np.array(args.param_value),
    )


if __name__ == "__main__":
    main()