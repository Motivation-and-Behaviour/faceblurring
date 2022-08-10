import logging
import os
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
from colorama import Fore, init
from PIL import Image
from tqdm import tqdm

from faceblurring.detection import *
from faceblurring.settings import *
from faceblurring.utils import *

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def main():
    init(autoreset=True)
    print(
        """
    ###########################################
                KidVision Faceblurring
    ###########################################
    """
    )

    ################
    # --- INPUTS ---
    part_id, input_dir = get_inputs()
    check_device()

    # Step zero: Generate any outputs
    output_dir = os.path.join(OUTPUT_DIR, f"Participant_{part_id}")
    output_dir_images = os.path.join(output_dir, "images")

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Step one: create the face detector
    detector = FaceAnalysis(
        allowed_modules=["detection"], providers=["CUDAExecutionProvider"]
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))

    # Step two: create the list of video files
    vid_files = get_video_files(input_dir)

    # Create the output timelapse
    out_tlc = cv2.VideoWriter(
        os.path.join(output_dir, "timelapse.avi"),
        cv2.VideoWriter_fourcc(*"DIVX"),
        OUT_VID_FPS,
        (1920, 1080),
    )

    # Step three: run through the videos and code the images
    img_id = 1
    start_time = timer()

    for vid in tqdm(vid_files, "Timelapse Files"):
        video = cv2.VideoCapture(vid)

        vid_name = Path(vid).stem

        vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        vid_frame_n = 0
        sv_frames = gen_step_frames(video.get(cv2.CAP_PROP_FPS), STEP_VID_LENGTH)

        frames, out_names = [], []

        with tqdm(
            total=vid_length, leave=False, desc=f"Loading frames (file: {vid_name})",
        ) as frame_pbar:
            while video.isOpened():
                success, frame = video.read()

                if success:
                    if vid_frame_n == 0:
                        # Only run the check on the first frame
                        tlc_vid = is_tlc_video(frame)
                        current_sv_frame = next(sv_frames)
                        if not tlc_vid:
                            frame_pbar.set_description(
                                f"Loading frames (file: {vid_name} is a step video)"
                            )

                    if (tlc_vid) or (not tlc_vid and current_sv_frame == vid_frame_n):
                        out_names.append(
                            make_out_name(output_dir_images, part_id, vid_name, img_id)
                        )

                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frames.append(frame)

                        img_id += 1
                        if vid_frame_n >= current_sv_frame:
                            current_sv_frame = next(sv_frames)

                    else:
                        frame_pbar.update(1)

                    vid_frame_n += 1

                else:
                    break

        video.release()

        # Detect faces in the resized images
        faces = []
        for frame in tqdm(frames, leave=False,):
            faces.append(detector.get(frame))

        for i, frame in enumerate(
            tqdm(frames, leave=False, desc=f"Blurring faces (file: {vid_name})")
        ):
            # Blur the frame
            blurred_frame = blur_faces(frame, faces[i], DEBUG)

            # Save the frame to disk
            cv2.imwrite(out_names[i], blurred_frame)

            # Add the frame to the TLC Video
            frame_num = out_names[i][-9:-4]
            cv2.putText(
                blurred_frame,
                f"Frame: {frame_num}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_4,
            )
            out_tlc.write(blurred_frame)

    out_tlc.release()
    end_time = timer()
    total_time = round(end_time - start_time, 3)

    print(
        f"[INFO] Created {img_id-1} images in {total_time} seconds ({round((img_id-1)/total_time,1)} fps)."
    )

    # Step four: create csv file of images
    csv_path = os.path.join(output_dir, f"Image_Log_{part_id}.csv")
    image_files = create_csv(output_dir_images, img_id, csv_path)

    print_instructions(output_dir, csv_path)

    # Step six: delete from csv
    delete_images(csv_path, output_dir_images, part_id, DEBUG)

    # Step seven: delete the original files
    if not DEBUG:
        tidy_up(vid_files, output_dir)

    input("Finished! Press any key to close...")


if __name__ == "__main__":
    main()
