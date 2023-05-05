import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Queue

from nicegui import app, ui

pool = ProcessPoolExecutor()

videos = ["video1", "video2", "video3", "video4", "video5"]


def heavy_computation(q: Queue, video) -> str:
    """Some heavy computation that updates the progress bar through the queue."""
    n = 50
    for i in range(n):
        # Perform some heavy computation
        time.sleep(0.1)

        # Update the progress bar through the queue
        update = (videos.index(video), i / (n - 1))
        q.put_nowait(update)
        print(update)
    return "Done!"


def labelled_progress_bar(video):
    ui.label(video)
    return ui.linear_progress(value=0).props("instant-feedback stripe rounded")


def parse_queue(queue):
    if not queue.empty():
        video_idx, progress = queue.get()
        progressbars[video_idx].set_value(progress)
        if progress == 1:
            progressbars[video_idx].props("color=positive")


@ui.page("/")
def main_page():
    async def start_computation():
        for video in videos:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(pool, heavy_computation, queue, video)
            ui.notify(result)

    # Create a queue to communicate with the heavy computation process
    queue = Manager().Queue()
    # Update the progress bar on the main process
    ui.timer(0.1, callback=lambda: parse_queue(queue))

    # Create the UI
    global progressbars
    progressbars = [labelled_progress_bar(video) for video in videos]
    ui.button("compute", on_click=start_computation)


# stop the pool when the app is closed; will not cancel any running tasks
app.on_shutdown(pool.shutdown)

ui.run()
