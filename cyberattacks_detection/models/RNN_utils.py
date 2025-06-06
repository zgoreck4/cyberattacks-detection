import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Rescaling
import numpy as np
import re
from .utils import min_max_scale, reverse_min_max_scale

tf_version = tf.__version__

# Import register_keras_serializable based on TensorFlow version
if tf_version.startswith("2.10") or tf_version.startswith("2.9") or tf_version.startswith("2.8"):
    from tensorflow.keras.utils import register_keras_serializable
else:
    from keras.saving import register_keras_serializable  # For TensorFlow 2.11+

@register_keras_serializable()
class MinMaxScalerLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that applies Min-Max scaling to inputs.

    Parameters
    ----------
    min_vals : array-like
        Minimum values for each feature used for scaling.
    max_vals : array-like
        Maximum values for each feature used for scaling.

    Methods
    -------
    call(inputs)
        Scales inputs using min-max scaling formula.
    get_config()
        Returns the config of the layer for serialization.
    from_config(config)
        Creates a layer from its config.
    """
    def __init__(self, min_vals, max_vals, **kwargs):
        super(MinMaxScalerLayer, self).__init__(**kwargs)
        self.min_vals = tf.convert_to_tensor(min_vals, dtype=tf.float32)
        self.max_vals = tf.convert_to_tensor(max_vals, dtype=tf.float32)

    def call(self, inputs):
        """
        Apply min-max scaling to inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to be scaled.

        Returns
        -------
        tf.Tensor
            Scaled tensor with values normalized to [0, 1].
        """
        return (inputs - self.min_vals) / (self.max_vals - self.min_vals)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns
        -------
        dict
            Configuration dictionary containing min_vals and max_vals.
        """
        config = super(MinMaxScalerLayer, self).get_config()
        config.update({
            "min_vals": self.min_vals.numpy().tolist(),
            "max_vals": self.max_vals.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        MinMaxScalerLayer
            An instance of MinMaxScalerLayer.
        """
        min_vals = config.pop("min_vals")
        max_vals = config.pop("max_vals")
        return cls(min_vals=min_vals, max_vals=max_vals, **config)

def create_rnn_input(data, target_col, time_steps):
    """
    Convert time series DataFrame into input-output pairs for RNN training.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data.
    target_col : int
        Index of the column to be predicted.
    time_steps : int
        Number of time steps in each input sequence.

    Returns
    -------
    tuple of np.ndarray
        X: Input sequences of shape (num_samples, time_steps, num_features).
        y: Target values of shape (num_samples,).
    """
    X, y = [], []

    # Ensure the data is a NumPy array for slicing
    data = data.values

    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # sequence input
        y.append(data[i + time_steps, target_col])  # next step target

    return np.array(X), np.array(y)

def create_rnn(num_hidden_layers, units_per_layer, num_features, time_steps, min_vals, max_vals, recurrent_layer=LSTM, **kwargs):
    """
    Create a sequential RNN model with min-max scaling input layer.

    Parameters
    ----------
    num_hidden_layers : int
        Number of recurrent layers.
    units_per_layer : int
        Number of units per recurrent layer.
    num_features : int
        Number of input features.
    time_steps : int
        Number of time steps per input sequence.
    min_vals : array-like
        Minimum values for each feature for scaling.
    max_vals : array-like
        Maximum values for each feature for scaling.
    recurrent_layer : class, optional
        Type of recurrent layer to use (LSTM or GRU), by default LSTM
    **kwargs
        Additional keyword arguments for recurrent layers.

    Returns
    -------
    tf.keras.Sequential
        Compiled RNN model.
    """
    model = Sequential()
    model.add(MinMaxScalerLayer(min_vals=min_vals, max_vals=max_vals, input_shape=(time_steps, num_features)))

    for i in range(num_hidden_layers):
        return_sequences = i < num_hidden_layers - 1  # True except last layer
        model.add(recurrent_layer(units_per_layer, activation='tanh',
                                  return_sequences=return_sequences,
                                  input_shape=(time_steps, num_features)))
    model.add(Dense(1))  # output layer
    return model

def create_recurrent_and_mlp_model(num_mlp_hidden_layers, recurrent_units_per_layer, mlp_units_per_layer,
                                   num_features, time_steps, min_vals, max_vals,
                                   recurrent_layer=LSTM, activation_mlp='relu', **kwargs):
    """
    Create a sequential model combining a recurrent layer and MLP layers.

    Parameters
    ----------
    num_mlp_hidden_layers : int
        Number of MLP hidden layers after the recurrent layer.
    recurrent_units_per_layer : int
        Number of units in the recurrent layer.
    mlp_units_per_layer : int
        Number of units in each MLP hidden layer.
    num_features : int
        Number of input features.
    time_steps : int
        Number of time steps in input sequence.
    min_vals : array-like
        Minimum values for input scaling.
    max_vals : array-like
        Maximum values for input scaling.
    recurrent_layer : class, optional
        Type of recurrent layer (LSTM or GRU), by default LSTM.
    activation_mlp : str, optional
        Activation function for MLP layers, by default 'relu'.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    tf.keras.Sequential
        Combined recurrent and MLP model.
    """
    model = Sequential()
    model.add(MinMaxScalerLayer(min_vals=min_vals, max_vals=max_vals, input_shape=(time_steps, num_features)))
    model.add(recurrent_layer(recurrent_units_per_layer, activation='tanh',
                              return_sequences=False,
                              input_shape=(time_steps, num_features)))

    for _ in range(num_mlp_hidden_layers):
        model.add(Dense(mlp_units_per_layer, activation=activation_mlp))

    model.add(Dense(1))  # output layer
    return model

def predict_recursion(df, model, features, y_name, num_features, time_steps, min_val_y, max_val_y):
    """
    Perform recursive prediction over time series data using a trained model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features and targets.
    model : tf.keras.Model
        Trained Keras model used for prediction.
    features : list of str
        Feature column names to be used for prediction.
    y_name : str
        Name of the target column to be predicted.
    num_features : int
        Number of features in the input.
    time_steps : int
        Number of time steps per input sequence.
    min_val_y : float or np.ndarray
        Minimum value(s) for output scaling.
    max_val_y : float or np.ndarray
        Maximum value(s) for output scaling.

    Returns
    -------
    pd.Series
        Predicted target values as a pandas Series.
    """
    pattern = r'x(\d+)'
    idx = int(re.findall(pattern, y_name)[0]) - 1
    df_pred = df.copy()

    for i in range(len(df_pred) - time_steps):
        batch = df_pred.iloc[i:i + time_steps][features].values
        X = np.reshape(batch, (1, time_steps, num_features))
        y_pred_sc = model.predict(X, verbose=0)
        y_pred = reverse_min_max_scale(y_pred_sc, min_val_y, max_val_y)
        df_pred.at[i + time_steps, y_name] = y_pred

    return df_pred[y_name]