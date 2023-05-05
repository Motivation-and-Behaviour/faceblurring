import os

from nicegui import ui

from .. import config, utils


def raise_error_dialog(message):
    with ui.dialog() as dialog, ui.card():
        ui.icon("error", color="negative").classes("text-5xl")
        ui.label(message)
        ui.button("Close", on_click=dialog.close)
    dialog.open()


def validate_inputdir(input_path):
    if isinstance(input_path.text, list):
        input_dir = input_path.text[0]
    else:
        input_dir = input_path.text

    valid = len(utils.get_video_files(input_dir))

    if not valid:
        raise_error_dialog("There are no video files in the input directory.")

    return valid, input_dir


def validate_outputdir(output_path):
    if isinstance(output_path.text, list):
        output_dir = output_path.text[0]
    else:
        output_dir = output_path.text

    if not os.path.isdir(output_dir):
        raise_error_dialog("Not a valid output directory")
        return False, output_dir

    return True, output_dir


def validate_settings(advanced_settings: dict):
    return all(
        isinstance(x, (int, float, complex)) and not isinstance(x, bool)
        for x in list(advanced_settings.values())[1:]
    )
