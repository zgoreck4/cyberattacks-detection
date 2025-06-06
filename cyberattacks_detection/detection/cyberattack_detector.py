import numpy as np
from numpy.typing import NDArray

class CyberattackDetector:
    """
    Cyberattack detector based on residual errors between measured and predicted values.

    Parameters
    ----------
    threshold : float or None, optional
        Threshold for detecting attacks. If None, should be computed by `calc_threshold`.
    window : int, default=1
        Size of the rolling window for residual calculation.
    residual_calc_func : str, default="mae"
        Method for residual calculation: "mae" (mean absolute error) or "rmse" (root mean squared error).

    Attributes
    ----------
    attack_signal : list
        List storing detection results for each call to `detect`.
    """

    def __init__(self, threshold: float = None, window: int = 1, residual_calc_func: str = "mae"):
        self.threshold = threshold
        self.window = window
        self.residual_calc_func = residual_calc_func
        self.attack_signal = []

    def _rolling_mae(self, measured_values: NDArray, predicted_values: NDArray) -> NDArray:
        """
        Compute rolling Mean Absolute Error (MAE) over a window.

        Parameters
        ----------
        measured_values : NDArray
            Array of measured values with shape (n_samples, time_steps).
        predicted_values : NDArray
            Array of predicted values with shape (n_samples, time_steps).

        Returns
        -------
        NDArray
            Rolling MAE values for each sample.
        """
        error = measured_values - predicted_values
        cumsum_abs = np.cumsum(np.abs(error), axis=1)
        cumsum_abs[:, self.window:] = cumsum_abs[:, self.window:] - cumsum_abs[:, :-self.window]
        rolling_mae = cumsum_abs[:, self.window - 1:] / self.window
        return rolling_mae

    def _rolling_rmse(self, measured_values: NDArray, predicted_values: NDArray) -> NDArray:
        """
        Compute rolling Root Mean Squared Error (RMSE) over a window.

        Parameters
        ----------
        measured_values : NDArray
            Array of measured values with shape (n_samples, time_steps).
        predicted_values : NDArray
            Array of predicted values with shape (n_samples, time_steps).

        Returns
        -------
        NDArray
            Rolling RMSE values for each sample.
        """
        error = measured_values - predicted_values
        cumsum_sq = np.cumsum(error ** 2, axis=1)
        cumsum_sq[:, self.window:] = cumsum_sq[:, self.window:] - cumsum_sq[:, :-self.window]
        rolling_rmse = np.sqrt(cumsum_sq[:, self.window - 1:] / self.window)
        return rolling_rmse

    def calc_threshold(self, measured_values: NDArray, predicted_values: NDArray, method: str, **kwargs):
        """
        Calculate detection threshold based on residuals using a specified method.

        Parameters
        ----------
        measured_values : NDArray
            Ground truth values (shape: n_samples x time_steps).
        predicted_values : NDArray
            Predicted values (shape: n_samples x time_steps).
        method : str
            Thresholding method: 'z-score', 'percentile', or 'max'.
        **kwargs
            Additional parameters:
            - n_std : int, number of standard deviations for 'z-score' method (default=4).
            - percentile : int, percentile for 'percentile' method (default=99).
        """
        if self.residual_calc_func == 'mae':
            residuals = self._rolling_mae(measured_values, predicted_values)
        elif self.residual_calc_func == 'rmse':
            residuals = self._rolling_rmse(measured_values, predicted_values)
        else:
            raise ValueError(f"Unknown residual_calc_func: {self.residual_calc_func}")

        if method == 'z-score':
            mean = np.mean(residuals, axis=1)
            std = np.std(residuals, axis=1)
            n_std = kwargs.get("n_std", 4)
            self.threshold = mean + n_std * std
            print(f"n_std={n_std}")
            print(f"threshold={self.threshold}")
        elif method == "percentile":
            percentile = kwargs.get("percentile", 99)
            self.threshold = np.percentile(residuals, percentile, axis=1)
            print(f"percentile={percentile}")
            print(f"threshold={self.threshold}")
        elif method == "max":
            self.threshold = np.max(residuals, axis=1)
            print(f"threshold={self.threshold}")
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def detect(self, measured_values: NDArray, predicted_values: NDArray, k: int) -> NDArray:
        """
        Detect cyberattacks at time index `k` based on residual errors exceeding threshold.

        Parameters
        ----------
        measured_values : NDArray
            Measured values array with shape (n_samples, time_steps).
        predicted_values : NDArray
            Predicted values array with shape (n_samples, time_steps).
        k : int
            Current time index to check for attack detection.

        Returns
        -------
        NDArray
            Boolean array indicating detection for each sample (True if attack detected).
        """
        start_idx = max(k - self.window + 1, 0)
        if self.residual_calc_func == 'mae':
            residuals = np.abs(measured_values[:, start_idx: k + 1] - predicted_values[:, start_idx: k + 1])
            residuals[np.isnan(residuals)] = 0
            result = np.mean(residuals, axis=1)
        elif self.residual_calc_func == 'rmse':
            residuals = (measured_values[:, start_idx: k + 1] - predicted_values[:, start_idx: k + 1]) ** 2
            residuals[np.isnan(residuals)] = 0
            result = np.sqrt(np.mean(residuals, axis=1))
        else:
            raise ValueError(f"Unknown residual_calc_func: {self.residual_calc_func}")

        detections = result > self.threshold
        self.attack_signal.append(detections)

        return detections
