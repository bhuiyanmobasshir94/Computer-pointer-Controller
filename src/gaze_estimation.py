import numpy as np

import cv2
from model import Model


class Model_GazeEstimation(Model):
    def process_eye(self, output_frame, cropped_face, eye_coord):
        cropped_face_width = cropped_face.shape[1]
        cropped_face_height = cropped_face.shape[0]
        eye_x = eye_coord[0]
        eye_y = eye_coord[1]
        eye_shape = [1, 3, 60, 60]
        eye_width = eye_shape[3]
        eye_height = eye_shape[2]
        xmin = int(eye_x - eye_width // 2) if int(eye_x - eye_width // 2) >= 0 else 0
        xmax = (
            int(eye_x + eye_width // 2)
            if int(eye_x + eye_width // 2) <= cropped_face_width
            else cropped_face_width
        )
        ymin = int(eye_y - eye_height // 2) if int(eye_y - eye_height // 2) >= 0 else 0
        ymax = (
            int(eye_y + eye_height // 2)
            if int(eye_y + eye_height // 2) <= cropped_face_height
            else cropped_face_height
        )
        eye = cropped_face[ymin:ymax, xmin:xmax]
        input_img = cv2.resize(
            eye, (eye_shape[3], eye_shape[2]), interpolation=cv2.INTER_AREA,
        )
        input_img = np.moveaxis(input_img, -1, 0)
        return input_img

    def preprocess_input(
        self, output_frame, cropped_face, left_eye_coord, right_eye_coord
    ):
        left_eye = self.process_eye(output_frame, cropped_face, left_eye_coord)
        right_eye = self.process_eye(output_frame, cropped_face, right_eye_coord)
        return (
            left_eye,
            right_eye,
        )

    def draw_eye(self, output_frame, b_box, eye_coord, gaze_vector_x, gaze_vector_y):
        eye_x = eye_coord[0]
        eye_y = eye_coord[1]
        xmin, ymin, xmax, ymax = b_box
        center_x = int(xmin + eye_x)
        center_y = int(ymin + eye_y)
        cv2.arrowedLine(
            output_frame,
            (center_x, center_y),
            (center_x + int(gaze_vector_x * 100), center_y + int(-gaze_vector_y * 100)),
            (255, 255, 0),
            3,
        )

    def preprocess_outputs(
        self, outputs, output_frame, flags, b_box, left_eye_coord, right_eye_coord
    ):
        gaze_vector_x = outputs[0][0]
        gaze_vector_y = outputs[0][1]
        gaze_vector_z = outputs[0][2]

        if flags and "gem" in flags:
            cv2.putText(
                output_frame,
                f"X:{gaze_vector_x*100:.1f}",
                (20, 100),
                0,
                0.7,
                (255, 255, 0),
            )
            cv2.putText(
                output_frame,
                f"Y:{gaze_vector_y*100:.1f}",
                (20, 120),
                0,
                0.7,
                (255, 255, 0),
            )
            cv2.putText(
                output_frame, f"Z:{gaze_vector_z:.1f}", (20, 140), 0, 0.7, (255, 255, 0)
            )

            self.draw_eye(
                output_frame, b_box, left_eye_coord, gaze_vector_x, gaze_vector_y
            )
            self.draw_eye(
                output_frame, b_box, right_eye_coord, gaze_vector_x, gaze_vector_y
            )

        return [gaze_vector_x, gaze_vector_y, gaze_vector_z], output_frame

    def predict(
        self,
        output_frame,
        flags,
        b_box,
        cropped_face,
        left_eye_coord,
        right_eye_coord,
        angle_list,
    ):
        left_eye, right_eye = self.preprocess_input(
            output_frame, cropped_face, left_eye_coord, right_eye_coord
        )
        processed_input = {
            "left_eye_image": left_eye,
            "right_eye_image": right_eye,
            "head_pose_angles": angle_list,
        }
        async_infer = self.net.start_async(request_id=0, inputs=processed_input)
        if async_infer.wait() == 0:
            result = async_infer.outputs[self.output_name]
            gaze_vector, output_frame = self.preprocess_outputs(
                result, output_frame, flags, b_box, left_eye_coord, right_eye_coord
            )
            return gaze_vector, output_frame
