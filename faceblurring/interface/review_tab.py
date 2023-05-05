import asyncio
import time

from nicegui import ui

from .. import config


async def run_chat():
    start_button.set_visibility(False)
    ui.chat_message(
        f"Hi there, {config.participant_name}!",
        name="KidVision Robot",
        avatar="https://robohash.org/Square Eyes",
    )
    await asyncio.sleep(2)
    ui.chat_message(
        "I'm here to help you get started with KidVision!",
        name="KidVision Robot",
        avatar="https://robohash.org/Square Eyes",
    )


def content():
    global start_button
    start_button = ui.button("Start", on_click=run_chat)
