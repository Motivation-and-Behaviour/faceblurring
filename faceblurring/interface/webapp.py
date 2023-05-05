from nicegui import app, ui

from .. import config
from . import done_tab, processing_tab, review_tab, setup_tab


def run_interface(native=False):
    config.init()
    app.add_static_files("/assets", "assets")

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
            # config.ui_tab_review.disable()
            config.ui_tab_done.disable()

    with ui.tab_panels(tabs, value=config.active_tab) as config.ui_tab_panel:
        with ui.tab_panel("Setup").style("min-width:600px"):
            setup_tab.content()
        with ui.tab_panel("Processing").style("min-width:600px"):
            processing_tab.content()
        with ui.tab_panel("Review").style("min-width:600px"):
            review_tab.content()
        with ui.tab_panel("Summary").style("min-width:600px"):
            done_tab.content()

    ui.run(title="KidVision Faceblurring", native=native)
