import os
from functools import partial

from nicegui import ui

from .. import config, utils
from . import validate
from .local_file_picker import local_file_picker, pick_file
from .processing_tab import make_progressbars

default_outputdir = [os.path.join(os.path.expanduser("~"), "Desktop", "KidVision Data")]


def validate_inputs():
    # Validate participant ID
    if not len(pid_input.value) == 4 or not pid_input.value.isnumeric():
        validate.raise_error_dialog("Please enter a valid participant ID.")
        return
    else:
        config.participant_id = pid_input.value

    # Validate names
    if not len(pname_input.value) or not len(parent_input.value):
        validate.raise_error_dialog("Please enter valid names.")
        return
    else:
        config.participant_name = pname_input.value
        config.parent_name = parent_input.value

    # Validate input directory
    input_dir_valid, config.input_dir = validate.validate_inputdir(input_path)
    if not input_dir_valid:
        return

    # Validate output directory
    output_dir_valid, output_dir = validate.validate_outputdir(output_path)
    if not output_dir_valid:
        return
    else:
        config.output_dir = os.path.join(
            output_dir, f"Particpant_{config.participant_id}"
        )

    # Validate advanced settings
    adv_settings_valid = validate.validate_settings(adv_settings)
    if not adv_settings_valid:
        return
    else:
        config.DELETE_ORIGINALS = adv_settings["del_org"]
        config.DETECTION_THRESH = adv_settings["det_thresh"]
        config.OUT_VID_FPS = adv_settings["fps"]
        config.STEP_VID_LENGTH = adv_settings["step_len"]
        config.STEP_VID_INTERVAL = adv_settings["step_int"]

    config.video_files = utils.get_video_files(config.input_dir)
    config.ui_tab_processing.enable()
    config.active_tab = "Processing"
    config.ui_tab_panel.set_value(config.active_tab)
    make_progressbars.refresh()


def content():
    global pid_input, pname_input, parent_input, input_path, output_path
    ui.markdown("### Participant Details")
    pid_input = ui.input(
        label="Participant ID",
        placeholder="Enter ID",
        validation={
            "Must be a four digit number": lambda value: value.isnumeric()
            and int(value) > 1000
            and int(value) < 9999,
        },
    )
    pname_input = ui.input(label="First Name", placeholder="Enter Name")
    parent_input = ui.input(label="Parent Name", placeholder="Enter Name")
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
    with ui.expansion("Advanced Settings", icon="settings").classes("w-full"):
        global adv_settings
        adv_settings = {}
        ui.markdown("**Processing**")
        adv_settings["del_org"] = ui.switch("Delete originals", value=True).tooltip(
            "Required for ethics"
        )
        adv_settings["fps"] = ui.number("Output Timelapse FPS", value=15).tooltip(
            "Changes playback speed"
        )
        ui.markdown("<br>**Model**")
        adv_settings["det_thresh"] = ui.number(
            "Detection Threshold", min=0, max=1, step=0.1, value=0.5
        )
        ui.markdown("<br>**Camera**")
        adv_settings["step_len"] = ui.number("Step video length", value=3).tooltip(
            "Length in seconds"
        )
        adv_settings["step_int"] = ui.number("Step video interval", value=0.5).tooltip(
            "Frequency in minutes"
        )

    ui.button("Continue", on_click=validate_inputs)
