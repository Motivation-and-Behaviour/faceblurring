import cv2
import numpy as np
import torch
from colorama import Fore, Style, init
from facenet_pytorch import MTCNN
from tqdm import tqdm


def check_device():
    if torch.cuda.is_available():
        return "cuda:0"

    print(
        f"{Fore.YELLOW}WARNING: {Style.RESET_ALL}Could not find GPU. Falling back to CPU."
    )

    return "cpu"


def detect_faces(detector, images, batch_size):
    boxes = []
    confs = []

    for lb in np.arange(0, len(images), batch_size):
        imgs = [img for img in images[lb : lb + batch_size]]
        boxes_temp, confs_temp = detector.detect(imgs)

        boxes.extend(boxes_temp)
        confs.extend(confs_temp)

    return (boxes, confs)
