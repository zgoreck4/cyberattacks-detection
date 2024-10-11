import numpy as np
from pandas import DataFrame

class ELM:
    def __init__(self, num_features, num_hidden_neurons):
        self.num_hidden_neurons = num_hidden_neurons
        self.weights = np.random.randn(num_features, num_hidden_neurons)
        self.b = np.random.randn(1, num_hidden_neurons)
        self.feature_names_in_ = None

    def _activation_func(self, tempH):
        # sigmoid
        return 1/(1 + np.exp(- tempH))

    def fit(self, X, y):
        if isinstance(X, DataFrame):
            self.feature_names_in_ = X.columns
        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH)    
        return H @ self.beta

    def save_model(self, path):
        """Optional: Save the model if needed."""
        np.savez(path, weights=self.weights, b=self.b, beta=self.beta)

    def load_model(self, path):
        """Optional: Load the model if needed."""
        model_data = np.load(path)
        self.weights = model_data['weights']
        self.b = model_data['b']
        self.beta = model_data['beta']