import logging
import os
from pathlib import Path
from timeit import default_timer as timer

import cv2
from tqdm import tqdm

import faceblurring.faceblurer as fb
from faceblurring.settings import *
from faceblurring.utils import gen_step_frames

if not DEBUG:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)


def main():
    print(
        """
    ###########################################
                KidVision Faceblurring
    ###########################################
    """
    )

    ################
    # --- INPUTS ---
    part_id, input_dir = fb.get_inputs()

    # Step zero: Generate any outputs
    output_dir = os.path.join(OUTPUT_DIR, f"Participant_{part_id}")
    output_dir_images = os.path.join(output_dir, "images")

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Step one: create the faceblurring object
    faceblurer = fb.FaceBlurer()

    # Step two: create the list of video files
    vid_files = fb.get_video_files(input_dir)

    # Step three: run through the videos and code the images
    img_id = 1
    start_time = timer()

    for vid in tqdm(vid_files, "Timelapse Files"):
        video = cv2.VideoCapture(vid)

        vid_name = Path(vid).stem

        vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        vid_frame_n = 0
        sv_frames = gen_step_frames(video.get(cv2.CAP_PROP_FPS), STEP_VID_LENGTH)

        with tqdm(
            total=vid_length, leave=False, desc=f"Frames (current file: {vid_name}",
        ) as frame_pbar:
            while video.isOpened():
                success, frame = video.read()

                if success:
                    if vid_frame_n == 0:
                        # Only run the check on the first frame
                        tlc_vid = fb.is_tlc_video(frame)
                        current_sv_frame = next(sv_frames)
                        if not tlc_vid:
                            print("[INFO] Step video detected\n")

                    if (tlc_vid) or (not tlc_vid and current_sv_frame == vid_frame_n):
                        out_name = os.path.join(
                            output_dir_images, f"{part_id}_{vid_name}_{img_id:05}.jpg"
                        )
                        img_id += 1
                        faceblurer.process_frame(frame, out_name)
                        frame_pbar.update(1)
                        if vid_frame_n >= current_sv_frame:
                            current_sv_frame = next(sv_frames)

                        vid_frame_n += 1
                    else:
                        frame_pbar.update(1)

                else:
                    break

        video.release()

    end_time = timer()
    total_time = round(end_time - start_time, 3)

    print(
        f"[INFO] Created {img_id-1} images in {total_time} seconds ({round((img_id-1)/total_time,1)} fps)."
    )

    # Step four: create csv file of images
    csv_path = os.path.join(output_dir, f"Image_Log_{part_id}.csv")
    image_files = fb.create_csv(output_dir_images, img_id, csv_path)

    # Step five: create timelapse video
    fb.create_timelapse_video(output_dir, image_files)

    # Step six: delete from csv
    fb.delete_images(csv_path, output_dir_images, part_id)

    # Step seven: delete the original files
    if not DEBUG:
        fb.tidy_up(vid_files, output_dir)


if __name__ == "__main__":
    main()
