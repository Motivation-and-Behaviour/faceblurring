import os
import cv2
import glob
import multiprocessing
import tqdm

from faceblurring.utils import *
from faceblurring.settings import *

import tensorflow as tf

from keras import backend as K

# import tensorflow.compat.v1.keras.backend as K
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

class FaceBlurer:
    def __init__(
        self,
        config_path=os.path.abspath("./weights/yolov3-face.cfg"),
        weights_path=os.path.abspath("./weights/yolov3-wider_16000.weights"),
        # model_path=os.path.abspath("./weights/YOLO_Face.h5"),
        # anchors_path=os.path.abspath("./cfg/yolo_anchors.txt"),
        # classes_path=os.path.abspath("./cfg/face_classes.txt")
    ):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def get_outputs_names(self):
        layers_names = self.net.getLayerNames()

        return [layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def process_frame(
        self, frame, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH, **kwargs
    ):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False
        )
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.get_outputs_names())

        # Remove the bounding boxes with low confidence
        post_process(frame, outs, conf_thresh, nms_thresh)

        return frame.astype(np.uint8)

    def process_file(
        self,
        file,
        replace=False,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
        **kwargs,
    ):
        # Read file
        img = cv2.imread(file)

        # Run file
        out_frame = self.process_frame(img, conf_thresh, nms_thresh)

        self.save_frame(file, out_frame, replace)

    def process_batch(
        self,
        folder,
        batch_size=32,
        replace=False,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
    ):
        files = glob.glob(os.path.join(folder, "*.jpg"))

        for files_batch in batch(files, batch_size):
            image_batch = [cv2.imread(img) for img in files_batch]
            blobs = cv2.dnn.blobFromImages(
                image_batch, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False
            )
            self.net.setInput(blobs)
            outs_multi = self.net.forward(self.get_outputs_names())
            for i, frame in enumerate(image_batch):
                post_process(
                    frame,
                    [outs_multi[0][i], outs_multi[1][i], outs_multi[2][i]],
                    conf_thresh,
                    nms_thresh,
                )
                self.save_frame(files_batch[i], frame, replace)

    def save_frame(self, file, frame, replace):
        if not replace:
            out_name = os.path.splitext(file)[0] + "_blurred.jpg"
        else:
            out_name = file

        cv2.imwrite(out_name, frame)

    def process_folder(
        self,
        folder,
        parallel=False,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
        replace=False,
    ):
        files = glob.glob(os.path.join(folder, "*.jpg"))

        if parallel:
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     executor.map(self.process_file, files)
            with multiprocessing.Pool() as pool:
                pool.map(self.process_file, files)
        else:
            t = tqdm.tqdm(files)
            for file in t:
                t.set_description(f"Working on file: {os.path.basename(file)}")
                t.refresh
                self.process_file(file)
