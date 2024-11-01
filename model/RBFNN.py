import numpy as np
from utils import *
import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans

class RBFNN:
    def __init__(self, n_centers, alpha=0.01):
        self.n_centers = n_centers
        self.sigma = np.ones((self.n_centers, 1))
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

    def fit(self, X, y, iterations=10):
        kmeans = KMeans(n_clusters=self.n_centers)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        for it in range(iterations):

            activations = self._calc_activations(X)

            # least squares method
            # weights = pseudo-inverse of activations @ y
            self.weights = np.linalg.pinv(activations) @ y

            grad = self._GD(X, y)
            self.sigma = self.sigma - self.alpha*grad

    def predict(self, X):
        activations = self._calc_activations(X)
        return activations @ self.weights