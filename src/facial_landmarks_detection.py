import cv2
from model import Model


class Model_FacialLandMarkDetection(Model):
    def predict(self, cropped_face, flags, prob_threshold, b_box, output_frame):
        processed_input = self.preprocess_input(cropped_face)
        async_infer = self.net.start_async(request_id=0, inputs=processed_input)
        if async_infer.wait() == 0:
            result = async_infer.outputs[self.output_name]
            left_eye_coord, right_eye_coord, image = self.preprocess_outputs(
                output_frame, flags, result, b_box
            )
            return left_eye_coord, right_eye_coord, image

    def preprocess_outputs(self, image, flags, outputs, cropped_face):
        height = cropped_face[3] - cropped_face[1]
        width = cropped_face[2] - cropped_face[0]
        landmarks = outputs.reshape(1, 10)[0]

        for i in range(2):
            x_coord = int(landmarks[i * 2] * width)
            y_coord = int(landmarks[i * 2 + 1] * height)
            if flags and "fldm" in flags:
                image = cv2.circle(
                    image,
                    (cropped_face[0] + x_coord, cropped_face[1] + y_coord),
                    30,
                    (0, 0, 255),
                    2,
                )
        left_eye_coord = [landmarks[0] * width, landmarks[1] * height]
        right_eye_coord = [landmarks[2] * width, landmarks[3] * height]
        return left_eye_coord, right_eye_coord, image
