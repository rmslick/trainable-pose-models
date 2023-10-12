from .GenericMLModel import GenericMLModel
from .Model import model
from tensorflow.keras.callbacks import Callback

class StreamingCallback(Callback):
    def __init__(self, rgbcnn_instance):
        super().__init__()
        self.rgbcnn_instance = rgbcnn_instance

    def on_epoch_end(self, epoch, logs=None):
        print("I AM IN CALLBACK")
        logs = logs or {}
        # Add the current epoch number to the logs
        logs['current_epoch'] = epoch + 1  # epochs are 0-indexed, so add 1 for human readability
        self.rgbcnn_instance.training_epoch_count = logs['current_epoch']
        # Update training_epoch_count of the RGBCNN instance
        #self.rgbcnn_instance.training_epoch_count = logs['current_epoch']
        print(">>>>>>>",self.rgbcnn_instance.training_epoch_count,self.rgbcnn_instance.epochs_total)
        self.log(logs)

    def log(self, logs):
        with open("training_stats.txt", "w") as file:
            for key, value in logs.items():
                if str(key) == "current_epoch":
                    file.write(f"{key},{value}")
            file.write("\n")  # Separate epochs with a newline for clarity
class RGBCNN(GenericMLModel):
    def __init__(self,config):
        self.model = model()
        self.training_epoch_count = 0

        self.epochs_prediction = config["epochs_prediction"]
        self.batch_size_prediction = config["batch_size_prediction"]

        self.epochs_refinement = config["epochs_refinement"]
        self.batch_size_refinement = config["batch_size_refinement"]

        self.epochs_total = self.epochs_prediction + self.epochs_refinement

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
        streaming_callback = StreamingCallback(self)  # pass the RGBCNN instance
        regression_history = self.model.regression_model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=150, batch_size=32,callbacks=[streaming_callback])
        # Predict the initial poses using the trained regression model
        initial_train_poses =self.model.regression_model.predict(X_train)
        initial_val_poses =self.model.regression_model.predict(X_val)

        # Train the refinement model
        streaming_callback2 = StreamingCallback(self)  # pass the RGBCNN instance
        refinement_history = self.model.refinement_model.fit([X_train, initial_train_poses], y_train,
                                                 validation_data=([X_val, initial_val_poses], y_val),
                                                 epochs=50, batch_size=64,callbacks=[streaming_callback2])

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
