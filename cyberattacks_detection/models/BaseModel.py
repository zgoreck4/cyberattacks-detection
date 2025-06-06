from abc import ABC, abstractmethod
import numpy as np
from pandas import DataFrame
from numpy.typing import NDArray
from typing import Union


class BaseModel(ABC):
    """
    Abstract base class for predictive models.

    This class provides the interface and utility methods for models
    that require min-max normalization and must implement fit, predict,
    save_model, and load_model methods.
    
    Attributes
    ----------
    input_min_arr : NDArray
        Minimum values for each input feature (for scaling).
    input_max_arr : NDArray
        Maximum values for each input feature (for scaling).
    output_min_arr : NDArray
        Minimum values for each output feature (for inverse scaling).
    output_max_arr : NDArray
        Maximum values for each output feature (for inverse scaling).
    feature_names_in_ : list of str
        Names of the input features.
    """

    def __init__(self) -> None:
        self.input_min_arr: NDArray = None
        self.input_max_arr: NDArray = None
        self.output_min_arr: NDArray = None
        self.output_max_arr: NDArray = None
        self.feature_names_in_: list[str] = None

    def _min_max_scale(
        self,
        X: Union[NDArray, DataFrame],
        x_min: NDArray,
        x_max: NDArray
    ) -> Union[NDArray, DataFrame]:
        """
        Normalize input data to the [0, 1] range using min-max scaling.

        Parameters
        ----------
        X : ndarray or DataFrame
            Input data to be scaled.
        x_min : ndarray
            Minimum values for each feature.
        x_max : ndarray
            Maximum values for each feature.

        Returns
        -------
        scaled_X : ndarray or DataFrame
            Scaled input data.
        """
        return (X - x_min) / (x_max - x_min)

    def _inverse_min_max_scale(
        self,
        X: Union[NDArray, DataFrame],
        x_min: NDArray,
        x_max: NDArray
    ) -> Union[NDArray, DataFrame]:
        """
        Undo min-max normalization and restore original scale.

        Parameters
        ----------
        X : ndarray or DataFrame
            Scaled input data to be unscaled.
        x_min : ndarray
            Original minimum values.
        x_max : ndarray
            Original maximum values.

        Returns
        -------
        original_X : ndarray or DataFrame
            Data restored to original scale.
        """
        return X * (x_max - x_min) + x_min

    @abstractmethod
    def fit(
        self,
        X: Union[NDArray, DataFrame],
        y: NDArray,
        input_min_arr: NDArray,
        input_max_arr: NDArray,
        **kwargs
    ) -> None:
        """
        Fit the model to training data.

        Parameters
        ----------
        X : ndarray or DataFrame
            Input features.
        y : ndarray
            Target values.
        input_min_arr : ndarray
            Minimum values of input features for scaling.
        input_max_arr : ndarray
            Maximum values of input features for scaling.
        kwargs : dict
            Additional keyword arguments for specific model implementations.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[NDArray, DataFrame]
    ) -> Union[NDArray, DataFrame]:
        """
        Predict target values for given input data.

        Parameters
        ----------
        X : ndarray or DataFrame
            Input data for prediction.

        Returns
        -------
        y_pred : ndarray or DataFrame
            Predicted values.
        """
        pass

    @abstractmethod
    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            Path to save the model to.
        """
        pass

    @abstractmethod
    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
            Path to the saved model.
        """
        pass
