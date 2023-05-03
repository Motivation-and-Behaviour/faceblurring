import asyncio
import concurrent.futures
import os
import time

import cv2

# Bounded semaphore to limit concurrent tasks
semaphore = asyncio.BoundedSemaphore(10)


def process_image(frame_number, frame):
    # Add the frame number as text
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_frame(frame_number, processed_frame):
    # Save the processed frame as an image
    cv2.imwrite(
        os.path.join("out_folder", f"frame_{frame_number}.png"), processed_frame
    )


async def process_and_save_frame(executor, frame_number, frame):
    async with semaphore:
        # Process the frame and save it concurrently
        processed_frame = await asyncio.get_event_loop().run_in_executor(
            executor, process_image, frame_number, frame
        )
        await asyncio.get_event_loop().run_in_executor(
            executor, save_frame, frame_number, processed_frame
        )


async def main(video_path):
    # Create a ThreadPoolExecutor for running the synchronous functions in separate threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Read the video file
        cap = cv2.VideoCapture(video_path)

        frame_number = 0
        tasks = []
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if we've reached the end of the video
            if not ret:
                break

            # Process and save the frame asynchronously
            task = asyncio.create_task(
                process_and_save_frame(executor, frame_number, frame)
            )
            tasks.append(task)

            frame_number += 1

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Release the video capture
        cap.release()


def main_seq(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        frames.append(frame)

    cap.release()

    for i, frame in enumerate(frames):
        # Process the frame
        processed_frame = process_image(i, frame)

        # Save the processed frame as an image
        save_frame(i, processed_frame)


video_path = "/Users/tasanders/GitHub/faceblurring/data/TLC00001.AVI"

start_time = time.time()
asyncio.run(main(video_path))
asycn_time = time.time() - start_time
print(f"Async time: {asycn_time}")


start_time = time.time()
main_seq(video_path)
seq_time = time.time() - start_time
print(f"Seq time: {seq_time}")
