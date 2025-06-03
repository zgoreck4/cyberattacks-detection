import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from pandas import DataFrame
from .BaseModel import BaseModel
from numpy.typing import NDArray
from typing import Union
from .utils import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

class RBFNN(BaseModel):
    def __init__(self, n_centers, alpha=0.01):
        super().__init__()
        self.n_centers = n_centers
        self.sigma = None
        self.centers = None
        self.weights = None
        self.alpha = alpha

    def _calc_activations(self, X):
        n_samples = X.shape[0]
        activations = np.ones((n_samples, self.n_centers+1))

        for i in range(n_samples):
            for j in range(self.n_centers):
                activations[i][j+1] = gaussian_func(X[i], self.centers[j], self.sigma[j])

        return activations
    
    def _GD(self, X, y):
        pred = self.predict(X)
        n_samples = X.shape[0]
        grad_sum = np.zeros((self.n_centers, 1))
        for j in range(self.n_centers):
            for i in range(n_samples):
                grad_gauss = grad_gaussian_func(X[i], self.centers[j], self.sigma[j])
                grad = (pred[i] - y[i]) @ (self.weights[j]*grad_gauss)
                # można zrb macierz ixj i zapisywać każdy wynik oddzielnie, a późnij policzyć sumę
                grad_sum[j] = grad_sum[j] + grad

        return grad_sum

    def fit(self,
            X: Union[NDArray, DataFrame],
            y: Union[NDArray, DataFrame],
            input_min_arr: NDArray,
            input_max_arr: NDArray,
            output_min_arr: NDArray,
            output_max_arr: NDArray,
            iterations: int=10,
            X_val: Union[NDArray, DataFrame] = None,
            y_val: Union[NDArray, DataFrame] = None,
            patience: int = 5,
            metric: callable = None, # only metric that a lower value is best     
            ClusterAlg: callable=KMeans,       
            **kwargs) -> int:
        if isinstance(X, DataFrame):
            self.feature_names_in_ = X.columns
        X = np.array(X)
        y = np.array(y)
        self.input_min_arr = input_min_arr
        self.input_max_arr = input_max_arr
        self.output_min_arr = output_min_arr
        self.output_max_arr = output_max_arr
        X = self._min_max_scale(X, self.input_min_arr, self.input_max_arr)
        y = self._min_max_scale(y, self.output_min_arr, self.output_max_arr)
        
        cluster_alg = ClusterAlg(n_clusters=self.n_centers)
        cluster_alg.fit(X)
        self.centers = cluster_alg.cluster_centers_

        center_distances = pdist(self.centers, metric='euclidean')
        sigma_init = np.mean(center_distances) / np.sqrt(2 * self.n_centers)
        self.sigma = np.full((self.n_centers, 1), sigma_init)

        best_metric_value = np.inf
        best_weights = None
        best_sigma = None
        for it in range(iterations):
            activations = self._calc_activations(X)
            # least squares method
            # weights = pseudo-inverse of activations @ y
            self.weights = np.linalg.pinv(activations) @ y
            grad = self._GD(X, y)
            self.sigma = self.sigma - self.alpha*grad

            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None and metric is not None:
                y_val_pred = self.predict(X_val)
                current_metric_value = metric(y_val, y_val_pred)

                # Check if there's an improvement
                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    best_weights = self.weights.copy()
                    best_sigma = self.sigma.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping check
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {it+1}. Best metric: {best_metric_value}")
                    break

        # Restore best weights and sigmas if early stopping occurred
        if best_weights is not None and best_sigma is not None:
            self.weights = best_weights
            self.sigma = best_sigma
            it = it - patience

        return it + 1

    def predict(self, X):
        X = np.array(X)
        X = self._min_max_scale(X, self.input_min_arr, self.input_max_arr)
        activations = self._calc_activations(X)
        y = activations @ self.weights
        return self._inverse_min_max_scale(y, self.output_min_arr, self.output_max_arr)

    def save_model(self, path):
        """Optional: Save the model if needed."""
        np.savez(path,
                 weights=self.weights,
                 sigma=self.sigma,
                 centers=self.centers,
                 feature_names_in_=self.feature_names_in_,
                 input_max_arr=self.input_max_arr, 
                 input_min_arr=self.input_min_arr,
                 output_max_arr=self.output_max_arr,
                 output_min_arr=self.output_min_arr)

    def load_model(self, path):
        """Optional: Load the model if needed."""
        model_data = np.load(path, allow_pickle=True)
        self.weights = model_data['weights']
        self.sigma = model_data['sigma']
        self.n_centers = np.shape(self.sigma)[0]
        self.centers = model_data['centers']
        self.feature_names_in_ = model_data['feature_names_in_']
        self.input_max_arr = model_data['input_max_arr']
        self.input_min_arr = model_data['input_min_arr']
        self.output_max_arr = model_data['output_max_arr']
        self.output_min_arr = model_data['output_min_arr']