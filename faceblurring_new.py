import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from timeit import default_timer as timer

import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from tqdm import tqdm

from faceblurring import settings, utils


def process_image(detector, frame):
    blurred_frame = utils.blur_faces(frame, detector.get(frame), settings.DEBUG)
    return blurred_frame


def process_and_save_frame(detector, frame, img_id, output_dir, part_id, vid_name):
    processed_frame = process_image(detector, frame)
    output_path = utils.make_out_name(output_dir, part_id, vid_name, img_id)
    cv2.imwrite(output_path, processed_frame)

    cv2.putText(
        processed_frame,
        f"Frame: {img_id:05}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )

    return (img_id, processed_frame)


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
    part_id, input_dir = utils.get_inputs()
    output_dir = os.path.join(
        os.path.expanduser("~"),
        "Desktop",
        settings.OUTPUT_FOLDER,
        f"Participant_{part_id}",
    )
    output_dir_images = os.path.join(output_dir, "images")

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Step one: create the face detector
    detector = FaceAnalysis(
        allowed_modules=["detection"], providers=["CUDAExecutionProvider"]
    )
    detector.prepare(ctx_id=0, det_size=(480, 480))

    # Step two: create the list of video files
    vid_files = utils.get_video_files(input_dir)

    # Create the output timelapse
    out_tlc = cv2.VideoWriter(
        os.path.join(output_dir, "timelapse.avi"),
        cv2.VideoWriter_fourcc(*"DIVX"),
        settings.OUT_VID_FPS,
        (1920, 1080),
    )

    img_id = 1
    written_frames = 0
    start_time = timer()

    for vid in vid_files:
        video = cv2.VideoCapture(vid)
        vid_name = Path(vid).stem
        vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_frame_n = 0
        sv_frames = utils.gen_step_frames(
            video.get(cv2.CAP_PROP_FPS), settings.STEP_VID_LENGTH
        )

        frames = []
        with tqdm(
            total=vid_length,
            leave=False,
            desc=f"Loading frames (file: {vid_name})",
        ) as frame_pbar:
            while video.isOpened():
                success, frame = video.read()

                if success:
                    if vid_frame_n == 0:
                        # Only run the check on the first frame
                        tlc_vid = utils.is_tlc_video(frame)
                        current_sv_frame = next(sv_frames)
                        if not tlc_vid:
                            frame_pbar.set_description(
                                f"Loading frames (file: {vid_name} is a step video)"
                            )

                    if (tlc_vid) or (not tlc_vid and current_sv_frame == vid_frame_n):
                        frames.append((img_id, frame))

                        img_id += 1
                        if vid_frame_n >= current_sv_frame:
                            current_sv_frame = next(sv_frames)

                    frame_pbar.update(1)

                    vid_frame_n += 1

                else:
                    break

        video.release()

        processed_frames = []

        with ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(
                    process_and_save_frame,
                    detector,
                    frame,
                    img_id,
                    output_dir,
                    part_id,
                    vid_name,
                ): img_id
                for img_id, frame in frames
            }

            for future in concurrent.futures.as_completed(future_to_frame):
                img_id, processed_frame = future.result()
                processed_frames.append((img_id, processed_frame))

        processed_frames.sort(key=lambda x: x[0])
        for img_id, processed_frame in processed_frames:
            out_tlc.write(processed_frame)

    out_tlc.release()
    end_time = timer()
    total_time = round(end_time - start_time, 3)

    print(
        f"[INFO] Created {img_id-1} images in {total_time} seconds ({round((img_id-1)/total_time,1)} fps)."
    )


if __name__ == "__main__":
    main()
