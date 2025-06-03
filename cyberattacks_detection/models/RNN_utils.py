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
    def __init__(self, min_vals, max_vals, **kwargs):
        super(MinMaxScalerLayer, self).__init__(**kwargs)
        self.min_vals = tf.convert_to_tensor(min_vals, dtype=tf.float32)
        self.max_vals = tf.convert_to_tensor(max_vals, dtype=tf.float32)

    def call(self, inputs):
        # Min-Max scaling formula
        return (inputs - self.min_vals) / (self.max_vals - self.min_vals)
    
    def get_config(self):
        config = super(MinMaxScalerLayer, self).get_config()
        config.update({
            "min_vals": self.min_vals.numpy().tolist(),
            "max_vals": self.max_vals.numpy().tolist()
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Convert lists back to tensors during deserialization
        min_vals = config.pop("min_vals")
        max_vals = config.pop("max_vals")
        return cls(min_vals=min_vals, max_vals=max_vals, **config)

# Prepare the data
def create_rnn_input(data, target_col, time_steps):
    """
    Converts a DataFrame to LSTM input format.
    
    Args:
        data (pd.DataFrame): Input time series data.
        time_steps (int): Number of time steps in each input sequence.

    Returns:
        X (np.ndarray): LSTM input of shape (num_samples, time_steps, num_features).
        y (np.ndarray): Targets of shape (num_samples, num_targets).
    """
    X, y = [], []
    
    # Ensure the data is a NumPy array for slicing
    data = data.values

    for i in range(len(data) - time_steps):
        # Input sequence
        X.append(data[i:i+time_steps])  # Take `time_steps` rows of all columns
        # Target: The next row after the sequence
        y.append(data[i+time_steps, target_col])  # Take the next row as the target

    return np.array(X), np.array(y)

# ze skalowaniem danych wejściowych

def create_rnn(num_hidden_layers, units_per_layer, num_features, time_steps, min_vals, max_vals, recurrent_layer=LSTM, **kwargs):
    model = Sequential()
    model.add(MinMaxScalerLayer(min_vals=min_vals, max_vals=max_vals,input_shape=(time_steps, num_features)))
    # Add LSTM layers
    for i in range(num_hidden_layers):
        return_sequences = i < num_hidden_layers - 1  # True for all but the last layer
        model.add(recurrent_layer(units_per_layer, activation='tanh', 
                        return_sequences=return_sequences, 
                        input_shape=(time_steps, num_features)))
    # Add dense output layer
    model.add(Dense(1))
    return model

# ze skalowaniem danych wejściowych

def create_recurrent_and_mlp_model(num_mlp_hidden_layers, recurrent_units_per_layer, mlp_units_per_layer, num_features, time_steps, min_vals, max_vals, recurrent_layer=LSTM, activation_mlp='relu', **kwargs):
    model = Sequential()
    model.add(MinMaxScalerLayer(min_vals=min_vals, max_vals=max_vals,input_shape=(time_steps, num_features)))
    # Add LSTM layers
    model.add(recurrent_layer(recurrent_units_per_layer, activation='tanh', 
                        return_sequences=False, 
                        input_shape=(time_steps, num_features)))
    for i in range(num_mlp_hidden_layers):
        model.add(Dense(mlp_units_per_layer, activation=activation_mlp))
    # Add dense output layer
    model.add(Dense(1))
    return model

def predict_recursion(df, model, features, y_name, num_features, time_steps, min_val_y, max_val_y):
    pattern = r'x(\d+)'
    idx = int(re.findall(pattern, y_name)[0]) - 1
    df_pred = df.copy()
    for i in range(len(df_pred) - time_steps):
        batch = df_pred.iloc[i:i+time_steps][features].values
        X = np.reshape(batch, (1, time_steps, num_features))
        # print(f"X: {X}")
        y_pred_sc = model.predict(X, verbose=0)
        y_pred = reverse_min_max_scale(y_pred_sc, min_val_y, max_val_y)
        # print(f"predykcja: {y_pred}")
        df_pred.at[i+time_steps, y_name] = y_pred
    return df_pred[y_name]