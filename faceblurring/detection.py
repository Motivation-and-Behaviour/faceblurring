import asyncio
import concurrent.futures
from pathlib import Path
from timeit import default_timer as timer

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from colorama import Fore, Style, init
from insightface.app import FaceAnalysis
from tqdm import tqdm

from . import settings, utils


def resize_box(box, frame_h, frame_w):
    x1, y1, x2, y2 = (
        max(int(box[0]), 0),
        max(int(box[1]), 0),
        min(int(box[2]), frame_w),
        min(int(box[3]), frame_h),
    )

    return (x1, y1, x2, y2)


def process_frame(frame, detector):
    faces = detector.get(frame)
    if not len(faces):
        return frame

    frame_h, frame_w, _ = frame.shape
    for face in faces:
        x1, y1, x2, y2 = resize_box(face["bbox"], frame_h, frame_w)
        conf = face["det_score"]
        roi = frame[y1:y2, x1:x2]

        # Blur the coloured image
        blur = cv2.GaussianBlur(roi, (101, 101), 0)
        # Insert the blurred section back into image
        frame[y1:y2, x1:x2] = blur

    return frame


def save_frame(frame, output_dir, part_id, vid_name, img_id, out_tlc):
    out_name = utils.make_out_name(output_dir, part_id, vid_name, img_id)
    cv2.imwrite(out_name, frame)

    cv2.putText(
        frame,
        f"Frame: {img_id:05}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )
    out_tlc.write(frame)


def add_tlc_frame(out_tlc, blurred_frame, img_id):
    cv2.putText(
        blurred_frame,
        f"Frame: {img_id:05}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_4,
    )
    out_tlc.write(blurred_frame)


async def process_and_save(
    semaphore, executor, frame, detector, output_dir, part_id, vid_name, img_id, out_tlc
):
    async with semaphore:
        processed_frame = await asyncio.get_event_loop().run_in_executor(
            executor, process_frame, frame, detector
        )
        await asyncio.get_event_loop().run_in_executor(
            executor,
            save_frame,
            processed_frame,
            output_dir,
            part_id,
            vid_name,
            img_id,
            out_tlc,
        )


async def run_processing(semaphore, vid_files, detector, output_dir, part_id, out_tlc):
    img_id = 1
    start_time = timer()

    for vid in tqdm(vid_files, "Timelapse Files"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            video = cv2.VideoCapture(vid)

            vid_name = Path(vid).stem

            vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            vid_frame_n = 0
            sv_frames = utils.gen_step_frames(
                video.get(cv2.CAP_PROP_FPS), settings.STEP_VID_LENGTH
            )
            tasks = []
            with tqdm(
                total=vid_length, leave=False, desc=f"Processing {vid_name}"
            ) as frame_pbar:
                while video.isOpened():
                    success, frame = video.read()

                    if success:
                        if vid_frame_n == 0:
                            # Run stepvideo check
                            tlc_vid = utils.is_tlc_video(frame)
                            current_sv_frame = next(sv_frames)
                            if not tlc_vid:
                                frame_pbar.set_description(
                                    f"Processing {vid_name} (step video)"
                                )

                        if (tlc_vid) or (
                            not tlc_vid and current_sv_frame == vid_frame_n
                        ):
                            task = asyncio.create_task(
                                process_and_save(
                                    semaphore,
                                    executor,
                                    frame,
                                    detector,
                                    output_dir,
                                    part_id,
                                    vid_name,
                                    img_id,
                                    out_tlc,
                                )
                            )
                            tasks.append(task)

                            img_id += 1

                            if vid_frame_n >= current_sv_frame:
                                current_sv_frame = next(sv_frames)

                        else:
                            frame_pbar.update(1)

                        vid_frame_n += 1

                    else:
                        break

            await asyncio.gather(*tasks)

            video.release()

    end_time = timer()
    total_time = round(end_time - start_time, 3)

    print(
        f"[INFO] Created {img_id-1} images in {total_time} seconds ({round((img_id-1)/total_time,1)} fps)."
    )


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
