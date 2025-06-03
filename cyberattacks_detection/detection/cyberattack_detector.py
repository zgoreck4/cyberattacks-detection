import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class CyberattackDetector:

    def __init__(self, threshold: float=None, window: int = 1, residual_calc_func: str="mae"):
        self.threshold = threshold
        self.window = window
        self.residual_calc_func = residual_calc_func
        self.attack_signal = []

    def _rolling_mae(self, measured_values: NDArray, predicted_values: NDArray) -> NDArray:
        """
        Computes the mean absolute error (MAE) over a rolling window.
        
        Parameters:
            arr1 (np.ndarray): First input 1D NumPy array.
            arr2 (np.ndarray): Second input 1D NumPy array.
            
        Returns:
            np.ndarray: Array of RMSE values computed over each rolling window.
        """
        # Compute error between arrays
        error = measured_values - predicted_values
        # Compute rolling RMSE
        cumsum_sq = np.cumsum(abs(error), axis=1)
        cumsum_sq[:, self.window:] = cumsum_sq[:, self.window:] - cumsum_sq[:, :-self.window]
        rolling_mae = cumsum_sq[:, self.window - 1:] / self.window
        
        return rolling_mae


    def _rolling_rmse(self, measured_values: NDArray, predicted_values: NDArray) -> NDArray:
        """
        Computes the root mean square error (RMSE) over a rolling window.
        
        Parameters:
            arr1 (np.ndarray): First input 1D NumPy array.
            arr2 (np.ndarray): Second input 1D NumPy array.
            
        Returns:
            np.ndarray: Array of RMSE values computed over each rolling window.
        """
        # Compute error between arrays
        error = measured_values - predicted_values
        # Compute rolling RMSE
        cumsum_sq = np.cumsum(error ** 2, axis=1)
        cumsum_sq[:, self.window:] = cumsum_sq[:, self.window:] - cumsum_sq[:, :-self.window]
        rolling_rmse = np.sqrt(cumsum_sq[:, self.window - 1:] / self.window)
        
        return rolling_rmse

    def calc_threshold(self, measured_values: NDArray, predicted_values: NDArray, method: str, **kwargs):
        if self.residual_calc_func == 'mae':
            residuals = self._rolling_mae(measured_values, predicted_values)
        elif self.residual_calc_func == 'rmse':
            residuals = self._rolling_rmse(measured_values, predicted_values)
        if method == 'z-score':
            mean = np.mean(residuals, axis=1)
            std = np.std(residuals, axis=1)
            n_std = kwargs.get("n_std", 4)
            self.threshold = mean + n_std*std
            print(f"{n_std=}")
            print(f"{self.threshold=}")
        elif method == "percentile":
            percentile = kwargs.get("percentile", 99)            
            self.threshold = np.percentile(residuals, percentile, axis=1)
            print(f"{percentile=}")
            print(f"{self.threshold=}")
        elif method == "max":            
            self.threshold = np.max(residuals, axis=1)
            print(f"{self.threshold=}")
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def detect(self, measured_values: NDArray, predicted_values: NDArray, k: int) -> bool:
        start_idx = max(k-self.window+1, 0)
        if self.residual_calc_func == 'mae':
            residuals = np.abs(measured_values[:, start_idx: k+1] - predicted_values[:, start_idx: k+1])
            residuals[np.isnan(residuals)] = 0
            result = np.mean(residuals, axis=1)
        elif self.residual_calc_func == 'rmse':
            residuals = (measured_values[:, start_idx: k+1] - predicted_values[:, start_idx: k+1])**2
            residuals[np.isnan(residuals)] = 0
            result = np.sqrt(np.mean(residuals, axis=1))
        detections = result > self.threshold
        self.attack_signal.append(detections)

        return detections
