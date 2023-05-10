import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

import cv2


def process_and_save_frame(frame_number, frame, output_dir, model_path):
    session = ort.InferenceSession(model_path)
    processed_frame = process_image(frame, frame_number, session)

    # Save the frame as an image
    image_path = os.path.join(output_dir, f"frame_{frame_number}.png")
    cv2.imwrite(image_path, processed_frame)

    return (frame_number, processed_frame)


def process_video(input_video_path, output_dir, output_video_path, model_path):
    os.makedirs(output_dir, exist_ok=True)

    capture = cv2.VideoCapture(input_video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Read all frames
    frames = []
    for _ in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # Process and save frames using a ThreadPoolExecutor
    processed_frames = []
    with ThreadPoolExecutor() as executor:
        future_to_frame = {
            executor.submit(
                process_and_save_frame, frame_number, frame, output_dir, model_path
            ): frame_number
            for frame_number, frame in enumerate(frames)
        }

        for future in concurrent.futures.as_completed(future_to_frame):
            frame_number, processed_frame = future.result()
            processed_frames.append((frame_number, processed_frame))

    # Sort the processed frames by frame number and write them to the output video
    processed_frames.sort(key=lambda x: x[0])
    for frame_number, processed_frame in processed_frames:
        video_writer.write(processed_frame)

    # Release the VideoWriter
    video_writer.release()


if __name__ == "__main__":
    input_video_path = "input_video.avi"
    output_dir = "output_frames"
    output_video_path = "output_video.avi"
    model_path = "path_to_your_onnx_model.onnx"
    process_video(input_video_path, output_dir, output_video_path, model_path)
