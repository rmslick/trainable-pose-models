from .GenericMLModel import GenericMLModel

class RGBCNN(GenericMLModel):
    def load_data(self, path):
        """
        Load and preprocess data from the provided path.
        This method should be implemented based on the specific dataset format.
        """
        pass

    def preprocess_data(self, data):
        """
        Preprocess the data if necessary.
        This can include operations like normalization, augmentation, etc.
        """
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model on the provided data.
        """
        pass

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        pass

    def save_model(self, path):
        """
        Save the model to the specified path.
        """
        pass
    def load_model(self, path):
        """
        Load a model from the specified path.
        """
        pass
