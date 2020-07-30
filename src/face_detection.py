import cv2
from model import Model


class Model_FaceDetection(Model):
    def predict(self, image, flags, prob_threshold):
        processed_input = self.preprocess_input(image)
        async_infer = self.net.start_async(request_id=0, inputs=processed_input)
        if async_infer.wait() == 0:
            result = async_infer.outputs[self.output_name]
            coords = self.preprocess_outputs(result, prob_threshold)
            b_boxes, image = self.draw_outputs(coords, image, flags)
            return b_boxes, image

    def draw_outputs(self, coords, image, flags):
        width = image.shape[1]
        height = image.shape[0]
        b_boxes = []
        for coord in coords:
            xmin = int(coord[0] * width)
            ymin = int(coord[1] * height)
            xmax = int(coord[2] * width)
            ymax = int(coord[3] * height)
            b_boxes.append([xmin, ymin, xmax, ymax])
            if flags and 'fdm' in flags:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        return b_boxes, image
