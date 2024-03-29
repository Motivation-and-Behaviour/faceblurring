import csv
import glob
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog

import cv2
import numpy as np
import onnxruntime as ort
import torch
from colorama import Fore, Style, init
from scipy.special import expit


def get_inputs():
    root = tk.Tk()
    root.withdraw()

    print("[INFO] Getting parameters...")

    while True:
        part_id = tk.simpledialog.askstring(
            "Participant ID", "Please provide the four digit ID number"
        )
        if len(part_id) == 4:
            break
        else:
            print("Participant ID must be a four digit code")

    # Input dir

    print("Please provide the location of the timelapse videos")
    input_dir = tk.filedialog.askdirectory()

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


def make_out_name(outdir, part_id, vid_name, frame_img_id):
    return os.path.join(outdir, f"{part_id}_{vid_name}_{frame_img_id:05}.jpg")


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


def check_device():
    if ort.get_device() != "GPU":
        print(
            f"{Fore.YELLOW}WARNING: {Style.RESET_ALL}Could not find GPU. Falling back to CPU."
        )
        return (480, 480), ["CPUExecutionProvider"]

    return (640, 640), ["CUDAExecutionProvider"]
