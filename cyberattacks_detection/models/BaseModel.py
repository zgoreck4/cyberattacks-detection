from abc import ABC, abstractmethod
import numpy as np
from pandas import DataFrame
from numpy.typing import NDArray
from typing import Union


class BaseModel(ABC):
    """
    Abstract base class for predictive models. Provides shared scaling methods
    and enforces implementation of fit, predict, and save_model methods.
    """
    def __init__(self) -> None:
        self.input_min_arr: NDArray = None
        self.input_max_arr: NDArray = None
        self.output_min_arr: NDArray = None
        self.output_max_arr: NDArray = None
        self.feature_names_in_: list[str] = None

    def _min_max_scale(self, X: NDArray | DataFrame, x_min: NDArray, x_max: NDArray) -> NDArray | DataFrame:
        """
        Scale data to the range [0, 1].
        """
        return (X - x_min) / (x_max - x_min)

    def _inverse_min_max_scale(self, X: NDArray | DataFrame, x_min: NDArray, x_max: NDArray) -> NDArray | DataFrame:
        """
        Unscale data from the range [0, 1] to original range.
        """
        return X * (x_max - x_min) + x_min
    
    @abstractmethod
    def fit(self,
            X: NDArray | DataFrame,
            y: NDArray,
            input_min_arr: NDArray,
            input_max_arr: NDArray,
            **kwargs) -> None:
        """
        Fit the model to the data.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray | DataFrame) -> NDArray | DataFrame:
        """
        Predict using the model.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load the model from the specified path.
        """
        pass