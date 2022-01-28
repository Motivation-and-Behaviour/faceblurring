import csv
import glob
import os
import tkinter as tk
from timeit import default_timer as timer
from tkinter import filedialog, simpledialog

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tqdm import tqdm

from faceblurring.settings import *
from faceblurring.utils import *


class FaceBlurer:
    def __init__(
        self,
        model_path=os.path.abspath("./weights/YOLO_Face.h5"),
        debug=DEBUG,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
        img_dims=IMG_DIMS,
    ):
        self.model = load_model(model_path, compile=False)
        self.debug = debug
        self.conf_thresh = (conf_thresh,)
        self.nms_thresh = nms_thresh
        self.net_h, self.net_w = img_dims, img_dims
        self.anchors = [
            [116, 90, 156, 198, 373, 326],
            [30, 61, 62, 45, 59, 119],
            [10, 13, 16, 30, 33, 23],
        ]

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
            boxes += decode_netout(
                yhat[i][0], self.anchors[i], self.conf_thresh, self.net_h, self.net_w
            )
        correct_yolo_boxes(
            boxes, frame.shape[0], frame.shape[1], self.net_h, self.net_w
        )
        do_nms(boxes, self.nms_thresh)
        out_boxes = list()
        out_conf = list()
        for box in boxes:
            if box.classes[0] > self.conf_thresh:
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


def get_inputs():
    root = tk.Tk()
    root.withdraw()

    print("[INFO] Getting parameters...")

    while True:
        part_id = simpledialog.askstring(
            "Participant ID", "Please provide the four digit ID number"
        )
        if len(part_id) == 4:
            break
        else:
            print("Participant ID must be a four digit code")

    # Input dir

    print("Please provide the location of the timelapse videos")
    input_dir = filedialog.askdirectory()

    print(
        f"""
    Participant ID: {part_id}
    Input Files:    {input_dir}

    Starting program...
    """
    )

    return part_id, input_dir


def get_video_files(input_dir):
    vid_files = glob.glob(os.path.join(input_dir, "*.AVI"))
    vid_files.sort()
    print(f"[INFO] Found {len(vid_files)} TLC files.")

    return vid_files


def create_csv(output_dir_images, img_id, csv_path):
    image_files = glob.glob(os.path.join(output_dir_images, "*.jpg"))
    image_files.sort()
    out_ids = ["{:05d}".format(id) for id in range(1, img_id)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Delete"])
        writer.writerows(zip(out_ids, [""] * len(out_ids)))

    return image_files


def create_timelapse_video(output_dir, image_files):
    print(f"[INFO] Creating timelapse video")
    start_time = timer()

    out = cv2.VideoWriter(
        os.path.join(output_dir, "timelapse.avi"),
        cv2.VideoWriter_fourcc(*"DIVX"),
        OUT_VID_FPS,
        (1920, 1080),
    )

    for image_file in tqdm(image_files):
        frame = cv2.imread(image_file)
        frame_num = image_file[-9:-4]
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )
        out.write(frame)

    out.release()
    end_time = timer()
    total_time = round(end_time - start_time, 3)
    print(f"[INFO] Created timelapse video in {total_time} seconds")


def check_csv_closed(csv_path):
    print("Please confirm that the csv file has been saved and closed.")
    while True:
        resp = input("Is the CSV file closed? (y/n)")
        if resp.lower() == "y":
            try:
                csv_file = open(csv_path, "r")
                break
            except IOError:
                print(
                    "File opening failed. Please check file has been closed in Excel."
                )

    csv_file.close()


def get_images_for_deletion(csv_path):
    check_csv_closed(csv_path)

    to_delete = list()

    # read the csv file back in
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[1]):
                to_delete.append(row[0])

    # First item will be the header
    to_delete = to_delete[1:]

    # Repad the numbers
    to_delete = [id.rjust(5, "0") for id in to_delete]

    print(f"[INFO] There are {len(to_delete)} files to be deleted.")

    return to_delete


def delete_images(csv_path, output_dir_images, part_id):
    to_delete = get_images_for_deletion(csv_path)

    deleted = 0
    for id in to_delete:
        try:
            os.remove(os.path.join(output_dir_images, f"{part_id}_{id}.jpg"))
            deleted += 1
        except:
            if DEBUG:
                print(f"Could not find file {part_id}_{id}.jpg")

    print(f"[INFO] Deleted {deleted} files.")


def tidy_up(vid_files, output_dir):
    print("Cleaning up...")

    # Original timelapse videos
    for vid_file in vid_files:
        os.remove(vid_file)
    print("\tRemoved original camera data")

    # Participant timelapse
    while True:
        try:
            os.remove(os.path.join(output_dir, "timelapse.avi"))
            print("\tRemoved blurred timelapse video")
            break
        except:
            print("Could not remove blurred timelapse video (it might still be open?)")
            input("Press enter to retry")


def is_tlc_video(frame):
    return (
        (np.all(frame[1056, 50] == [16, 16, 16]))  # Bottom left corner
        and (np.all(frame[1056, 1870] == [16, 16, 16]))  # Bottom right corner
        and (np.all(frame[1056, 777] == [240, 240, 240]))  # "T" in TLC
    )
