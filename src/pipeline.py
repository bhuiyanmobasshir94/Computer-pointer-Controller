import logging
import os
import sys
import time
from argparse import ArgumentParser

import cv2
from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandMarkDetection
from gaze_estimation import Model_GazeEstimation
from head_pose_estimation import Model_HeadPoseEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path_generator = lambda x: os.path.join(BASE_DIR, "models", x)
input_path_generator = lambda x: os.path.join(BASE_DIR, "bin", x)
output_path_generator = lambda x: os.path.join(BASE_DIR, "results", x)
log_path_generator = lambda x: os.path.join(BASE_DIR, "logs", x)
log_file_location = log_path_generator("App.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file_location), logging.StreamHandler()],
)

model_dict = None
input_path = None
output_path = None
device = None
cpu_extension = None
prob_threshold = None
flags = None
mouse_controller = None
feeder = None
video_writer = None
model_loading_total_time = None
model_inference_total_time = None


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-fdm",
        "--face_detection_model",
        required=True,
        type=str,
        help="Path to an xml and bin (without extension) file with a trained model.",
    )
    parser.add_argument(
        "-fldm",
        "--facial_landmarks_detection_model",
        required=True,
        type=str,
        help="Path to an xml and bin (without extension) file with a trained model.",
    )
    parser.add_argument(
        "-hpem",
        "--head_pose_estimation_model",
        required=True,
        type=str,
        help="Path to an xml and bin (without extension) file with a trained model.",
    )
    parser.add_argument(
        "-gem",
        "--gaze_estimation_model",
        required=True,
        type=str,
        help="Path to an xml and bin (without extension) file with a trained model.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to image or video file or camera feed usage direction."
        "For camera feed please pass 'CAM' keyword",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output directory name."
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help="MKLDNN (CPU)-targeted custom layers."
        "Absolute path to a shared library with the"
        "kernels impl.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for detections filtering" "(0.5 by default)",
    )
    parser.add_argument(
        "-f",
        "--flags",
        required=False,
        nargs="+",
        default=[],
        help="Specify flag with one or more model flags separated by space"
        "flags can be used fdm fldm hpem gem like -f fdm or -f fdm fldm etc",
    )
    return parser


def generate_model_dict(model_args, model_class):
    model_dict = {}
    logging.info("*********** Model Load Time Start ***************")
    start_loading = time.time()
    for arg, m_class in zip(model_args, model_class):
        try:
            start_time = time.time()
            model = model_path_generator(arg)
            model_dict[m_class.__name__] = m_class(model, device, cpu_extension)
            model_dict[m_class.__name__].load_model()
            end_time = time.time() - start_time
            logging.info(f"{m_class.__name__}: {1000 * end_time:.1f} ms.")
        except Exception as e:
            logging.error(f"Error while loading {m_class.__name__} ~ {e} ")
            sys.exit(1)
    end_loading = time.time() - start_loading
    logging.info("*********** Model Load Time End ***************")
    return model_dict, end_loading


def setup(args):
    global input_path, output_path, device, cpu_extension, prob_threshold, flags, mouse_controller, feeder, video_writer, model_dict, model_loading_total_time
    model_args = [
        args.face_detection_model,
        args.facial_landmarks_detection_model,
        args.head_pose_estimation_model,
        args.gaze_estimation_model,
    ]
    model_class = [
        Model_FaceDetection,
        Model_FacialLandMarkDetection,
        Model_HeadPoseEstimation,
        Model_GazeEstimation,
    ]
    input_path = input_path_generator(args.input) if args.input != "CAM" else None
    output_path = output_path_generator(args.output)
    device = args.device
    cpu_extension = args.cpu_extension
    prob_threshold = args.prob_threshold
    flags = args.flags
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    mouse_controller = MouseController("low", "fast")
    if input_path:
        if input_path.endswith(".jpg"):
            feeder = InputFeeder("image", input_path)
        else:
            feeder = InputFeeder("video", input_path)
    else:
        feeder = InputFeeder("cam")
    feeder.load_data()
    fps = feeder.fps()
    initial_w, initial_h, video_len = feeder.frame_initials_and_length()
    video_writer = cv2.VideoWriter(
        os.path.join(output_path, "output_video.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps / 10,
        (initial_w, initial_h),
        True,
    )
    model_dict, model_loading_total_time = generate_model_dict(model_args, model_class)
    return


def inference():
    inference_start_time = time.time()
    count = 0
    face_detection_infer_total = 0
    facial_landmark_detection_infer_total = 0
    headpose_estimation_infer_total = 0
    gaze_estimation_infer_total = 0

    face_detection = model_dict["Model_FaceDetection"]
    facial_landmark_detection = model_dict["Model_FacialLandMarkDetection"]
    headpose_estimation = model_dict["Model_HeadPoseEstimation"]
    gaze_estimation = model_dict["Model_GazeEstimation"]

    while True:
        try:
            flag, frame = next(feeder.next_batch())
        except StopIteration:
            break
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        start_time = time.time()
        b_boxes, output_frame = face_detection.predict(frame, flags, prob_threshold)
        face_detection_infer_total += time.time() - start_time

        for b_box in b_boxes:
            cropped_face = frame[b_box[1] : b_box[3], b_box[0] : b_box[2]]

            start_time = time.time()
            left_eye_coord, right_eye_coord, output_frame = facial_landmark_detection.predict(
                cropped_face, flags, prob_threshold, b_box, output_frame
            )
            facial_landmark_detection_infer_total += time.time() - start_time

            start_time = time.time()
            angle_list, output_frame = headpose_estimation.predict(
                cropped_face, flags, prob_threshold, output_frame
            )
            headpose_estimation_infer_total += time.time() - start_time

            start_time = time.time()
            gaze_vector, output_frame = gaze_estimation.predict(
                output_frame, flags, b_box, cropped_face, left_eye_coord, right_eye_coord, angle_list
            )
            gaze_estimation_infer_total += time.time() - start_time

            cv2.imshow("Computer Pointer Control", output_frame)
            video_writer.write(output_frame)
            mouse_controller.move(gaze_vector[0], gaze_vector[1])

        count += 1
        if key_pressed == 27:
            break

    model_inference_total_time = time.time() - inference_start_time
    fps = count / round(model_inference_total_time, 1)

    if count > 0:
        logging.info("*********** Model Inference Time Start ****************")
        logging.info(f"Model_FaceDetection: {1000*face_detection_infer_total/count:.1f} ms.")
        logging.info(f"Model_FacialLandMarkDetection: {1000*facial_landmark_detection_infer_total/count:.1f} ms.")
        logging.info(f"Model_HeadPoseEstimation: {1000*headpose_estimation_infer_total/count:.1f} ms.")
        logging.info(f"Model_GazeEstimation: {1000*gaze_estimation_infer_total/count:.1f} ms.")
        logging.info("*********** Model Inference Time End ***********")

    logging.info("*********** Summary ****************")
    logging.info(f"model_loading_total_time: {model_loading_total_time} s.")
    logging.info(f"model_inference_total_time: {model_inference_total_time} s.")
    logging.info(f"FPS: {fps}")
    logging.info("*********** Summary End ***********")

    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
        f.write(str(model_loading_total_time)+'\n')
        f.write(str(model_inference_total_time)+'\n')
        f.write(str(fps)+'\n')

    feeder.close()
    cv2.destroyAllWindows()


def pipeline():
    args = build_argparser().parse_args()
    setup(args)
    inference()

if __name__ == "__main__":
    pipeline()
