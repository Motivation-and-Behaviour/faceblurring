def init():
    global active_tab
    active_tab = "Setup"

    # UI Elements
    global ui_tab_panel, ui_tab_processing, ui_tab_review, ui_tab_done

    # Files and file paths
    global input_dir, output_dir, video_files
    video_files = []

    # Participant characteristics
    global participant_id, participant_name, parent_name
    # TODO: Remove this
    participant_name = "Test name"

    # Advanced settings
    global DETECTION_THRESH, OUT_VID_FPS, STEP_VID_LENGTH, STEP_VID_INTERVAL, DELETE_ORIGINALS
