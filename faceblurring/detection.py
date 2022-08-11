import cv2
import insightface
import numpy as np
import onnxruntime as ort
from colorama import Fore, Style, init
from insightface.app import FaceAnalysis
from tqdm import tqdm


def check_device():
    if ort.get_device() != "GPU":
        print(
            f"{Fore.YELLOW}WARNING: {Style.RESET_ALL}Could not find GPU. Falling back to CPU."
        )


def detect_faces(detector, images, batch_size):
    boxes = []
    confs = []

    for lb in np.arange(0, len(images), batch_size):
        imgs = [img for img in images[lb : lb + batch_size]]
        boxes_temp, confs_temp = detector.detect(imgs)

        boxes.extend(boxes_temp)
        confs.extend(confs_temp)

    return (boxes, confs)
