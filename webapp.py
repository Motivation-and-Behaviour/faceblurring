import os

from nicegui import ui

from local_file_picker import local_file_picker

default_outputdir = os.path.join(os.path.expanduser("~"), "Desktop", "KidVision Data")


def save_inputs():
    print(participant_id.value)
    print(participant_name.value)
    print(parent_name.value)
    print(input_path.text)
    print(output_path.text)


async def pick_file(element) -> None:
    result = await local_file_picker("~", multiple=False)
    ui.notify(f"You chose {result}")
    element.text = result


with ui.card():
    ui.markdown("### Particpant Details")
    participant_id = ui.input(label="Participant ID", placeholder="Enter ID")
    participant_name = ui.input(label="First Name", placeholder="Enter Name")
    parent_name = ui.input(label="Parent Name", placeholder="Enter Name")
    ui.markdown("### Data")
    ui.markdown("**Camera Data**")
    input_path = ui.label("")
    ui.button("Choose folder", on_click=pick_file(input_path)).props(
        "icon=folder flat size=sm"
    )
    ui.markdown("**Output Folder**")
    output_path = ui.label(default_outputdir)
    out_button = ui.button("Change folder").props("icon=folder flat size=sm")

ui.button("Save", on_click=save_inputs)


ui.run()
