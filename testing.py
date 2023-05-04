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
        q.put_nowait(i / n)
    return "Done!"


@ui.page("/")
def main_page():
    async def start_computation():
        for i, video in enumerate(videos):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                pool, heavy_computation, queue[i], video
            )
            ui.notify(result)

    # Create a queue to communicate with the heavy computation process
    queue = [Manager().Queue() for _ in range(len(videos))]
    # Update the progress bar on the main process
    [
        ui.timer(
            0.1,
            callback=lambda: progressbars[i].set_value(
                queue[i].get() if not queue[i].empty() else progressbars[i].value
            ),
        )
        for i in range(len(videos))
    ]

    # Create the UI
    ui.button("compute", on_click=start_computation)
    progressbars = [
        ui.linear_progress(value=0).props("instant-feedback stripe")
        for _ in range(len(videos))
    ]


# stop the pool when the app is closed; will not cancel any running tasks
app.on_shutdown(pool.shutdown)

ui.run()
