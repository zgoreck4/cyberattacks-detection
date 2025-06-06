from mlflow.pyfunc import PythonModel


class ModelWrapper(PythonModel):
    """
    MLflow wrapper for a custom model.

    This class allows integrating any custom model into the MLflow pyfunc
    interface, making it possible to log and load the model via MLflow.

    Parameters
    ----------
    model : object
        Trained model instance implementing a `predict` method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, context, X):
        """
        Run inference using the wrapped model.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            MLflow context (not used here).
        X : array-like
            Input data to predict on.

        Returns
        -------
        y_pred : array-like
            Predicted values from the model.
        """
        return self.model.predict(X)
