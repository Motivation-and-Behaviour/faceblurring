import os
from functools import partial

from nicegui import ui

from .. import config
from .local_file_picker import local_file_picker, pick_file

default_outputdir = [os.path.join(os.path.expanduser("~"), "Desktop", "KidVision Data")]


def content():
    global participant_id, participant_name, parent_name, input_path, output_path
    ui.markdown("### Participant Details")
    participant_id = ui.input(
        label="Participant ID",
        placeholder="Enter ID",
        validation={
            "Must be a four digit number": lambda value: value.isnumeric()
            and int(value) > 1000
            and int(value) < 9999,
        },
    )
    participant_name = ui.input(label="First Name", placeholder="Enter Name")
    parent_name = ui.input(label="Parent Name", placeholder="Enter Name")
    ui.markdown("### Data")
    ui.markdown("**Camera Data**")
    input_path = ui.label("")
    ui.button("Choose folder", on_click=partial(pick_file, input_path)).props(
        "icon=folder flat size=sm"
    )
    ui.markdown("**Output Folder**")
    output_path = ui.label(default_outputdir)
    ui.button("Change folder", on_click=partial(pick_file, output_path)).props(
        "icon=folder flat size=sm"
    )
    with ui.expansion("Other Settings", icon="settings").classes("w-full"):
        ui.markdown("**Processing**")
        ui.switch("Delete originals", value=True).tooltip("Required for ethics")
        ui.number("Output Timelapse FPS", value=15).tooltip("Changes playback speed")
        ui.markdown("<br>**Model**")
        ui.number("Detection Threshold", min=0, max=1, step=0.1, value=0.5)
        ui.markdown("<br>**Camera**")
        ui.number("Step video length", value=3).tooltip("Length in seconds")
        ui.number("Step video interval", value=0.5).tooltip("Frequency in minutes")

    ui.button("Continue", on_click=save_inputs)


def save_inputs():
    print(participant_id.value)
    print(participant_name.value)
    print(parent_name.value)
    print(input_path.text)
    print(output_path.text)
    config.active_tab = "Processing"
    config.tab_panel.set_value(config.active_tab)
    config.ui_tab_processing.enable()
