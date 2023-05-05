import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Queue

from nicegui import app, ui

from .. import config

pool = ProcessPoolExecutor()


@ui.refreshable
def make_progressbars():
    global progressbars
    progressbars = []

    for video in config.video_files:
        ui.label(os.path.basename(video))
        progressbars.append(
            ui.linear_progress(value=0, size="25px", show_value=False).props(
                "instant-feedback stripe rounded"
            )
        )


def heavy_computation(q: Queue, video) -> str:
    """Some heavy computation that updates the progress bar through the queue."""
    n = 50
    for i in range(n):
        # Perform some heavy computation
        time.sleep(0.01)

        # Update the progress bar through the queue
        update = (config.video_files.index(video), i / (n - 1))
        q.put_nowait(update)
    return "Done!"


def parse_queue(queue):
    if not queue.empty():
        video_idx, progress = queue.get()
        progressbars[video_idx].set_value(progress)
        if progress == 1:
            progressbars[video_idx].props("color=positive")


def finish_processing():
    config.ui_tab_review.enable()
    config.active_tab = "Review"
    config.ui_tab_panel.set_value(config.active_tab)


def content():
    async def start_computation():
        compute_button.set_visibility(False)
        for video in config.video_files:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(pool, heavy_computation, queue, video)
            ui.notify(result)
        cont_button.set_visibility(True)

    # Create a queue to communicate with the heavy computation process
    queue = Manager().Queue()
    # Update the progress bar on the main process
    ui.timer(0.01, callback=lambda: parse_queue(queue))

    # Create the UI
    make_progressbars()
    compute_button = ui.button("compute", on_click=start_computation)
    cont_button = ui.button("Continue", on_click=finish_processing)
    cont_button.set_visibility(False)

    # stop the pool when the app is closed; will not cancel any running tasks
    app.on_shutdown(pool.shutdown)
