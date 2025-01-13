import numpy as np
from numpy.typing import NDArray

class CyberattackDetector:

    def __init__(self, threshold: float=None, window: int = 1):
        self.threshold = threshold
        self.window = window
        self.attack_signal = []

    def calc_threshold(self, measured_values: NDArray, predicted_values: NDArray, method: str, **kwargs):
        residuals = np.abs(measured_values - predicted_values)
        if method == 'z-score':
            mean = np.mean(residuals, axis=1)
            std = np.std(residuals, axis=1)
            n_std = kwargs.get("n_std", 4)
            self.threshold = mean + n_std*std
            print(f"{self.threshold=}")
        elif method == "percentile":
            percentile = kwargs.get("percentile", 99)            
            self.threshold = np.percentile(residuals, percentile, axis=1)
            print(f"{self.threshold=}")
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def detect(self, measured_values: NDArray, predicted_values: NDArray, k: int) -> bool:
        start_idx = max(k-self.window+1, 0)
        residuals = np.abs(measured_values[:, start_idx: k+1] - predicted_values[:, start_idx: k+1])
        residuals[np.isnan(residuals)] = 0
        result = np.mean(residuals, axis=1)
        detections = result > self.threshold
        if detections.any() and k<1500:
            print(measured_values[:, start_idx: k+1])
        self.attack_signal.append(detections)

        return detections
