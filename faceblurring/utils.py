import csv
import glob
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog

import cv2
import numpy as np
from colorama import Fore
from scipy.special import expit


def get_video_files(input_dir):
    vid_files = glob.glob(os.path.join(input_dir, "*.AVI"))
    vid_files.sort()

    return vid_files


def is_tlc_video(frame):
    return (
        (np.all(frame[1056, 50] == [16, 16, 16]))  # Bottom left corner
        and (np.all(frame[1056, 1870] == [16, 16, 16]))  # Bottom right corner
        and (np.all(frame[1056, 777] == [240, 240, 240]))  # "T" in TLC
    )


def gen_step_frames(vid_fps, step_vid_length):
    t = 0
    while True:
        yield int(t * vid_fps)
        t += step_vid_length


def make_out_name(outdir, part_id, vid_name, img_id):
    return os.path.join(outdir, f"{part_id}_{vid_name}_{img_id:05}.jpg")


def blur_faces(frame, faces, debug=False):
    if not len(faces):
        return frame

    frame_h, frame_w, _ = frame.shape

    for face in faces:
        x1, y1, x2, y2 = resize_box(face["bbox"], frame_h, frame_w)
        conf = face["det_score"]

        if debug:
            # Include the rect and conf
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{:.2f}".format(conf)
            # Display the label at the top of the bounding box
            label_size, base_line = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 2, 1
            )
            top = max(y2, label_size[1])
            cv2.putText(
                frame,
                text,
                (x1, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Add the blurring in
        roi = frame[y1:y2, x1:x2]

        try:
            # Blur the coloured image
            blur = cv2.GaussianBlur(roi, (101, 101), 0)
            # Insert the blurred section back into image
            frame[y1:y2, x1:x2] = blur
        except:
            if debug:
                print(f"[ERROR] Blurring failed.")

    return frame


def resize_box(box, frame_h, frame_w):
    x1, y1, x2, y2 = (
        max(int(box[0]), 0),
        max(int(box[1]), 0),
        min(int(box[2]), frame_w),
        min(int(box[3]), frame_h),
    )

    return (x1, y1, x2, y2)


def create_csv(output_dir_images, img_id, csv_path):
    image_files = glob.glob(os.path.join(output_dir_images, "*.jpg"))
    image_files.sort()
    out_ids = ["{:05d}".format(id) for id in range(1, img_id)]
    image_files_base = [os.path.basename(filename) for filename in image_files]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Delete", "Filename"])
        writer.writerows(zip(out_ids, [""] * len(out_ids), image_files_base))


def delete_images(csv_path, output_dir_images, part_id, debug=False):
    to_delete = get_images_for_deletion(csv_path)

    deleted = 0
    for file in to_delete:
        try:
            os.remove(os.path.join(output_dir_images, file))
            deleted += 1
        except:
            if debug:
                print(f"Could not find file {file}")

    print(f"[INFO] Deleted {deleted} files.")


def get_images_for_deletion(csv_path):
    check_csv_closed(csv_path)

    to_delete = []

    # read the csv file back in
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[1]):
                to_delete.append(row[2])

    # First item will be the header
    to_delete = to_delete[1:]

    print(f"[INFO] There are {len(to_delete)} files to be deleted.")

    return to_delete


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


def print_instructions(output_dir, csv_path):
    print(f"{Fore.GREEN} INSTRUCTIONS\n\n")
    print(
        "Particpant to review the timelapse and indicate images to be deleted on the CSV file.\n"
    )
    print(f"Participant Timelapse: {os.path.join(output_dir, 'timelapse.avi')}")
    print(f"Particpant CSV: {csv_path}")
    try:
        os.startfile(output_dir)
    except AttributeError:
        subprocess.call(["open", output_dir])


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
