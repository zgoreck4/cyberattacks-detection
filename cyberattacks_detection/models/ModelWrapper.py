from mlflow.pyfunc import PythonModel

class ModelWrapper(PythonModel):

    def __init__(self, model):
        # Store your custom model as a class attribute
        self.model = model

    def predict(self, context, X):
        # Use your model to make predictions
        return self.model.predict(X)