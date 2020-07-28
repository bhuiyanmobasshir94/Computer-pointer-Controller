from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandMarkDetection
from gaze_estimation import Model_GazeEstimation
from head_pose_estimation import Model_HeadPoseEstimation

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


model = os.path.join(BASE_DIR,'models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
device = "CPU"
extensions = None

fd= Model_FaceDetection(model, device, extensions)
fd.load_model()
import pdb;pdb.set_trace()