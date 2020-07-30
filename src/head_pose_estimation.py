import cv2
from model import Model


class Model_HeadPoseEstimation(Model):
    def predict(self, cropped_face, flags, prob_threshold, output_frame):
        processed_input = self.preprocess_input(cropped_face)
        async_infer = self.net.start_async(request_id=0, inputs=processed_input)
        if async_infer.wait() == 0:
            result = async_infer.outputs
            angle_list, image = self.preprocess_outputs(output_frame, flags, result)
            return angle_list, image

    def preprocess_outputs(self, image, flags, outputs):
        yaw = outputs["angle_y_fc"][0][0]
        pitch = outputs["angle_p_fc"][0][0]
        roll = outputs["angle_r_fc"][0][0]

        if flags and "hpem" in flags:
            image = cv2.putText(
                image, f"yaw:{yaw:.1f}", (20, 20), 0, 0.6, (255, 255, 0)
            )
            image = cv2.putText(
                image, f"pitch:{pitch:.1f}", (20, 40), 0, 0.6, (255, 255, 0)
            )
            image = cv2.putText(
                image, f"roll:{roll:.1f}", (20, 60), 0, 0.6, (255, 255, 0)
            )

        return [yaw, pitch, roll], image
