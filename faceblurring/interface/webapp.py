from nicegui import ui

from .. import config
from . import setup_tab


def run_interface(native=False):
    config.init()

    ui.colors(
        primary="#3c1053", secondary="#3d3935", accent="#c8e3db", negative="#ed0c00"
    )

    with ui.header() as header:
        with ui.tabs() as tabs:
            ui.tab("Setup", icon="home")
            config.ui_tab_processing = ui.tab("Processing", icon="blur_on")
            config.ui_tab_review = ui.tab("Review", icon="face")
            config.ui_tab_done = ui.tab("Summary", icon="done")

            config.ui_tab_processing.disable()
            config.ui_tab_review.disable()
            config.ui_tab_done.disable()

    with ui.tab_panels(tabs, value=config.active_tab) as config.tab_panel:
        with ui.tab_panel("Setup"):
            setup_tab.content()
            # ui.button("test", on_click=ui_tab_processing.enable)
        with ui.tab_panel("Processing"):
            pass
        with ui.tab_panel("Review"):
            pass
        with ui.tab_panel("Summary"):
            pass

    ui.run(title="KidVision Faceblurring", native=native)
