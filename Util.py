import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .RGBCNN import *
from tensorflow.keras.preprocessing import image
import numpy as np

def pose_error(gt_pose, pred_pose):
    """
    Calculate and print the translation and rotation error between a ground truth pose and a predicted pose.

    Parameters:
    - gt_pose: Ground truth pose as a 7D array [tx, ty, tz, qx, qy, qz, qw].
    - pred_pose: Predicted pose as a 7D array [tx, ty, tz, qx, qy, qz, qw].
    """

    # Translation error
    trans_error = np.linalg.norm(gt_pose[:3] - pred_pose[:3])

    # Rotation error
    q_gt = gt_pose[3:]
    q_pred = pred_pose[3:]
    dot_product = np.dot(q_gt, q_pred)

    # Clip to ensure dot_product is within the valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    rotation_error = 2 * np.arccos(np.abs(dot_product))

    #print(f"Translation Error: {trans_error:.4f} units")
    #print(f"Rotation Error: {rotation_error:.4f} radians")

    return trans_error, rotation_error
def get_ground_truth_pose_for_image(image_filename, json_path):
    """
    Retrieve the ground truth 6DoF pose for a given image from the LineMOD dataset.

    Parameters:
    - image_filename: The filename of the image, e.g., '000123.jpg'.
    - json_path: Path to the scene_gt.json file for the corresponding scene.

    Returns:
    - pose: A dictionary containing the ground truth pose (rotation and translation).
    """

    # Extract image ID from the filename
    image_id = int(image_filename.split('.')[0])

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract pose for the given image ID
    pose_data = data[str(image_id)][0]  # Assuming one primary object per image in LineMOD
    rotation_matrix = pose_data['cam_R_m2c']
    translation_vector = pose_data['cam_t_m2c']

    pose = {
        'rotation': rotation_matrix,
        'translation': translation_vector
    }

    return pose
import numpy as np
from scipy.spatial.transform import Rotation

def dict_to_7d_format(pose_dict):
    """
    Convert a dictionary with 'rotation' and 'translation' keys to a 7D array format:
    [tx, ty, tz, qx, qy, qz, qw].
    """
    R = np.array(pose_dict['rotation']).reshape(3, 3)  # Convert rotation list to a 3x3 matrix.

    # Convert the rotation matrix to a quaternion.
    rotation = Rotation.from_matrix(R)
    quaternion = rotation.as_quat()

    # Combine translation and quaternion into a single array.
    pose_7d = np.concatenate([pose_dict['translation'], quaternion])

    return pose_7d
import numpy as np

def pose_error(gt_pose, pred_pose):
    """
    Calculate and print the translation and rotation error between a ground truth pose and a predicted pose.

    Parameters:
    - gt_pose: Ground truth pose as a 7D array [tx, ty, tz, qx, qy, qz, qw].
    - pred_pose: Predicted pose as a 7D array [tx, ty, tz, qx, qy, qz, qw].
    """

    # Translation error
    trans_error = np.linalg.norm(gt_pose[:3] - pred_pose[:3])

    # Rotation error
    q_gt = gt_pose[3:]
    q_pred = pred_pose[3:]
    dot_product = np.dot(q_gt, q_pred)

    # Clip to ensure dot_product is within the valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    rotation_error = 2 * np.arccos(np.abs(dot_product))

    #print(f"Translation Error: {trans_error:.4f} units")
    #print(f"Rotation Error: {rotation_error:.4f} radians")

    return trans_error, rotation_error


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
    """Load images from a folder, resize them, and ignore .json files."""

    images = []
    image_names = sorted(os.listdir(folder_path))

    for image_name in image_names:
        if not image_name.endswith('.json'):
            img_path = os.path.join(folder_path, image_name)
            img = cv2.imread(img_path)

            if img is not None:  # Check if the file was a valid image format
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
