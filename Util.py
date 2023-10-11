import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .RGBCNN import *
def poses_to_quaternion_format(poses):
    """
    Convert a list of poses in the format:
    [[R11, R12, R13, tx],
     [R21, R22, R23, ty],
     [R31, R32, R33, tz]]
    into a list of poses in the format [tx, ty, tz, qx, qy, qz, qw].
    """
    quaternion_poses = []

    for pose in poses:
        pose_np = np.array(pose[0])  # Convert to numpy array.
        R = pose_np[:, :3]  # Extract the 3x3 rotation matrix.
        t = pose_np[:, 3]  # Extract the translation vector.

        # Convert rotation matrix to quaternion.
        rotation = Rotation.from_matrix(R)
        quaternion = rotation.as_quat()

        quaternion_pose = list(t) + list(quaternion)
        quaternion_poses.append(quaternion_pose)

    return quaternion_poses
def load_images_from_folder(folder_path, img_size=(224, 224)):
    """Load images from a folder and resize them."""

    images = []
    image_names = sorted(os.listdir(folder_path))

    for image_name in image_names:
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img)

    return images

def load_json(file_path):
    """Load a JSON file."""

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data
def process_gt_data(scene_gt, scene_camera):
    """Extract 6DoF pose and camera parameters from scene_gt and scene_camera."""

    poses = []
    camera_params = []

    for image_id, annotations in scene_gt.items():
        image_data = []
        for annotation in annotations:
            R = np.array(annotation["cam_R_m2c"]).reshape(3, 3)
            t = np.array(annotation["cam_t_m2c"]).reshape(3, 1)
            pose = np.hstack([R, t])
            image_data.append(pose)
        poses.append(image_data)

        cam_data = scene_camera[image_id]
        K = np.array(cam_data["cam_K"]).reshape(3, 3)
        camera_params.append(K)

    return poses, camera_params
from tensorflow.keras.preprocessing import image

def preprocess_images(images):
    """Normalize the images to [0, 1]."""
    return np.array(images, dtype=np.float32) / 255.0

'''
# Paths
IMAGE_FOLDER = "data/rgb"
SCENE_GT_PATH = "data/scene_gt.json"
SCENE_CAMERA_PATH = "data/scene_camera.json"

# Load images and JSON data
images = load_images_from_folder(IMAGE_FOLDER)
scene_gt = load_json(SCENE_GT_PATH)
scene_camera = load_json(SCENE_CAMERA_PATH)

# Process ground truth and camera parameters
poses, camera_params = process_gt_data(scene_gt, scene_camera)

# Preprocess images
images = preprocess_images(images)

quaternion_poses = poses_to_quaternion_format(poses)

from sklearn.model_selection import train_test_split

# Split the data into 70% training data, 15% validation data, and 15% test data

# First, separate out the test set (15% of the total data)
X_temp, X_test, y_temp, y_test = train_test_split(images, quaternion_poses, test_size=0.15, random_state=42)

# Now, split the remaining data (X_temp, y_temp) into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 of 85% is roughly 15%

# Convert lists to numpy arrays
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

rgbmodel = RGBCNN()
rgbmodel.train(X_train,y_train,X_val,y_val)
rgbmodel.test(X_test,y_test)
'''
