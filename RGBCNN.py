from .GenericMLModel import GenericMLModel
from .Model import model
from tensorflow.keras.callbacks import Callback

class StreamingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Here you can send the metrics (in logs) to your frontend or any other destination.
        # For example:
        print(logs)
class RGBCNN(GenericMLModel):
    def __init__(self):
        self.model = model()
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
        regression_history = self.model.regression_model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=150, batch_size=32,callbacks=[streaming_callback])
        # Predict the initial poses using the trained regression model
        initial_train_poses =self.model.regression_model.predict(X_train)
        initial_val_poses =self.model.regression_model.predict(X_val)
        # Train the refinement model
        refinement_history = self.model.refinement_model.fit([X_train, initial_train_poses], y_train,
                                                 validation_data=([X_val, initial_val_poses], y_val),
                                                 epochs=50, batch_size=64,callbacks=[streaming_callback])

    def test(self,X_test,y_test_quat):
        initial_test_poses =self.model.regression_model.predict(X_test)
        y_pred_quat = self.model.refinement_model.predict([X_test, initial_test_poses])
        # Initialize lists to store errors
        trans_errors = []
        rot_errors = []

        # Compute errors for each prediction against the ground truth
        for gt_pose, pred_pose in zip(y_test_quat, y_pred_quat):
            trans_error, rot_error = pose_error(gt_pose, pred_pose)
            trans_errors.append(trans_error)
            rot_errors.append(rot_error)

        # Convert lists to numpy arrays for further analysis
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)

        # Report overall mean and standard deviation for both translation and rotation errors
        print(f"Mean Translation Error: {trans_errors.mean():.4f} units")
        print(f"Standard Deviation of Translation Error: {trans_errors.std():.4f} units")
        print(f"Mean Rotation Error: {rot_errors.mean():.4f} radians")
        print(f"Standard Deviation of Rotation Error: {rot_errors.std():.4f} radians")
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
