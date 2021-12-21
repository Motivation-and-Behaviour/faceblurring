import os
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np
from faceblurring.utils import *
from faceblurring.settings import *
from timeit import default_timer as timer

class FaceBlurer:
    def __init__(
        self,
        model_path=os.path.abspath("./weights/YOLO_Face.h5"),
        debug=DEBUG,
        conf_thresh = CONF_THRESH,
        nms_thresh = NMS_THRESH,
        img_dims = IMG_DIMS
    ):
        self.model = load_model(model_path, compile = False)
        self.debug = debug
        self.conf_thresh = conf_thresh,
        self.nms_thresh = nms_thresh
        self.net_h, self.net_w = img_dims, img_dims
        self.anchors = [[116,90, 156,198, 373,326], 
                        [30,61, 62,45, 59,119], 
                        [10,13, 16,30, 33,23]]

    def process_frame(self, frame, out_name):
        proc_start = timer()
        proc_image = preprocess_input(frame, self.net_h, self.net_w)
        proc_end = timer()
        # if self.debug: print(f"preprocess_input: {proc_end-proc_start}")

        pred_start = timer()
        yhat = self.model.predict_on_batch(proc_image)
        pred_end = timer()
        # if self.debug: print(f"model predict: {pred_end-pred_start}")

        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], self.anchors[i], self.conf_thresh, self.net_h, self.net_w)
        correct_yolo_boxes(boxes, frame.shape[0], frame.shape[1], self.net_h, self.net_w)
        do_nms(boxes, self.nms_thresh)
        out_boxes = list()
        out_conf = list()
        for box in boxes:
            if box.classes[0]> self.conf_thresh:
                out_boxes.append(box)
                out_conf.append(box.classes[0])

        post_start = timer()
        post_process(frame, out_boxes, out_conf, self.debug)
        post_end = timer()
        # if self.debug: print(f"postprocess: {post_end-post_start}")

        write_start = timer()
        cv2.imwrite(out_name, frame)
        write_end = timer()
        # if self.debug: print(f"write: {write_end-write_start}")     

        

        




    
    


    