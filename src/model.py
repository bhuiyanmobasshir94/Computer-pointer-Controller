import numpy as np

import cv2
from openvino.inference_engine import IECore


class Model:
    def __init__(self, model, device, extensions=None):
        self.model_weights = model + ".bin"
        self.model_structure = model + ".xml"
        self.device = device
        self.extensions = extensions

        try:
            self.core = IECore()
            if self.extensions and "CPU" in self.device:
                self.core.add_extension(self.extensions, self.device)

            self.model = self.core.read_network(
                model=self.model_structure, weights=self.model_weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not Initialise the network. Have you enterred the correct model path? Got this error ~ {e}"
            )

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        self.net = self.core.load_network(
            network=self.model, device_name=self.device, num_requests=1
        )

    def check_model(self):
        supported_layers = self.core.query_network(
            network=self.model, device_name=self.device
        )
        unsupported_layers = [
            l for l in self.model.layers.keys() if l not in supported_layers
        ]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        return

    def preprocess_outputs(self, outputs, threshold=0.6):
        coords = []
        for coord in outputs[0][0]:
            if coord[2] >= threshold:
                coords.append(coord[3:])
        return coords

    def preprocess_input(self, image):
        input_img = image
        input_img = cv2.resize(
            input_img,
            (self.input_shape[3], self.input_shape[2]),
            interpolation=cv2.INTER_AREA,
        )
        input_img = np.moveaxis(input_img, -1, 0)
        input_dict = {self.input_name: input_img}
        return input_dict
