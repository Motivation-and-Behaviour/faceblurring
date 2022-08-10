######## PARAMS ########
CONF_THRESH = 0.6
NMS_THRESH = 0.5
IMG_DIMS = 416  # Must be multiple of 32
BATCH_SIZE = {"cpu": 8, "cuda:0": 32}
DIM_FACTOR = {"cpu": 2, "cuda:0": 1.5}
DEBUG = True
OUTPUT_DIR = "C:/Users/MB/Desktop/KidVision Data"
OUT_VID_FPS = 15.0
STEP_VID_LENGTH = 3
STEP_VID_INTERVAL = 0.5
