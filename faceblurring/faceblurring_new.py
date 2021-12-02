import glob
import os

import cv2

import faceblurring.faceblurer

################
# --- INPUTS ---
part_id = "1001"
input_dir = "C:/Users/MB/Desktop/DCIM/130TLC00"

# Step zero: Generate any outputs
output_dir = os.path.join(input_dir, "blurred")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Step one: create the faceblurring object
faceblurer = faceblurring.faceblurer.FaceBlurer()

# Step two: create the list of video files
vid_files = glob.glob(os.path.join(input_dir, "*.AVI"))
vid_files.sort()

# Step three: run through the videos and code the images
img_id = 1

for vid in vid_files:
    video = cv2.VideoCapture(vid)
    video.set(cv2.CAP_PROP_FPS, 1 / 60)  # I think this can be removed?

    while video.isOpened():
        success, frame = video.read()

        if success:
            out_name = os.path.join(output_dir, f"{part_id}_{img_id:05}.jpg")
            img_id += 1

        else:
            break

# Step four: create csv file of images

# Step five: create timelapse video

# Step six: delete from csv
