import numpy as np

class ELM:
    def __init__(self, num_inputs, num_hidden_neurons):
        self.num_hidden_neurons = num_hidden_neurons
        self.weights = np.random.randn(num_inputs, num_hidden_neurons)
        self.b = np.random.randn(1, num_hidden_neurons)

    def _activation_func(self, tempH):
        # sigmoid
        return 1/(1 + np.exp(- tempH))

    def fit(self, X, y):

        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH)
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        tempH = X @ self.weights + self.b
        H = self._activation_func(tempH)    
        return H @ self.beta
    
