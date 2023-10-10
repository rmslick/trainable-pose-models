from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Concatenate
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Flatten
from keras.applications import ResNet50

class model
    def __init__(self):
        # Ensure TensorFlow is utilizing GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            print("No GPU found, model will be trained on CPU.")

        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()

        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            self.regression_model = self.create_resnet_6dof_model()
            self.regression_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
            self.refinement_model = self.pose_refinement_network()
            self.refinement_model.compile(optimizer='adam', loss='mse')
    def create_resnet_6dof_model(self,input_shape=(224, 224, 3)):
        # Load the ResNet50 model with weights pre-trained on ImageNet
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the layers of the ResNet model
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features using the ResNet50 model
        x = base_model.output

        # Add some custom layers on top
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)

        # Predict the 3 translation values
        translation = Dense(3, name="translation")(x)

        # Predict the 4 quaternion values for rotation
        quaternion = Dense(4, activation='tanh', name="quaternion")(x)

        # Combine the translation and rotation into a single output
        final_output = Concatenate(name="6DoF_output")([translation, quaternion])

        # Construct the full model
        model = Model(inputs=base_model.input, outputs=final_output)
    def pose_refinement_network(self,input_shape=(224, 224, 3)):
        # Initial pose input (7D: [tx, ty, tz, qx, qy, qz, qw])
        initial_pose_input = Input(shape=(7,), name="initial_pose")

        # RGB Image input
        image_input = Input(shape=input_shape, name="image_input")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        image_features = base_model(image_input)
        image_features = Flatten()(image_features)

        # Concatenate image features and initial pose
        x = Concatenate()([image_features, initial_pose_input])

        # Dense layers for refinement
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)

        # Output layer for refined pose
        refined_pose = Dense(7, name="refined_pose")(x)

        model = Model(inputs=[image_input, initial_pose_input], outputs=refined_pose)

        return model
    def predict(self,x):
        pose_init = self.regression_model.predict(x)
        pose_refined = self.refinement_model.predict(x,pose_init)
        return pose_refined
    
print("Starting...")
model()
