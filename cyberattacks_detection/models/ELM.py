import numpy as np
from pandas import DataFrame
from .BaseModel import BaseModel
from numpy.typing import NDArray

class ELM(BaseModel):
    def __init__(self, num_features, num_hidden_neurons):
        super().__init__()
        self.num_hidden_neurons = num_hidden_neurons
        self.weights = np.random.randn(num_features, num_hidden_neurons)
        self.b = np.random.randn(1, num_hidden_neurons)

    def _activation_func(self, tempH):
        # sigmoid
        return 1/(1 + np.exp(- tempH))

    def fit(self,
            X: NDArray | DataFrame,
            y: NDArray,
            input_min_arr: NDArray,
            input_max_arr: NDArray,
            output_min_arr: NDArray,
            output_max_arr: NDArray,
            **kwargs) -> None:
        if isinstance(X, DataFrame):
            self.feature_names_in_ = X.columns
        self.input_min_arr = input_min_arr
        self.input_max_arr = input_max_arr
        self.output_min_arr = output_min_arr
        self.output_max_arr = output_max_arr
        X = self._min_max_scale(X, self.input_min_arr, self.input_max_arr)
        y = self._min_max_scale(y, self.output_min_arr, self.output_max_arr)

        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        X = self._min_max_scale(X, self.input_min_arr, self.input_max_arr)
        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH) 
        y = H @ self.beta
        return self._inverse_min_max_scale(y, self.output_min_arr, self.output_max_arr)

    def save_model(self, path):
        """Optional: Save the model if needed."""
        np.savez(path,
                 weights=self.weights,
                 b=self.b,
                 beta=self.beta,
                 feature_names_in_=self.feature_names_in_,
                 input_max_arr=self.input_max_arr, 
                 input_min_arr=self.input_min_arr,
                 output_max_arr=self.output_max_arr,
                 output_min_arr=self.output_min_arr)

    def load_model(self, path):
        """Optional: Load the model if needed."""
        model_data = np.load(path)
        self.weights = model_data['weights']
        self.b = model_data['b']
        self.num_hidden_neurons = np.shape(self.b)[1]
        self.beta = model_data['beta']
        self.feature_names_in_ = model_data['feature_names_in_']
        self.input_max_arr = model_data['input_max_arr']
        self.input_min_arr = model_data['input_min_arr']
        self.output_max_arr = model_data['output_max_arr']
        self.output_min_arr = model_data['output_min_arr']