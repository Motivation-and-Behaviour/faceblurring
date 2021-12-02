import cv2
import faceblurring.faceblurer
import glob
import os

################
# --- INPUTS ---
participant_id = "1001"
input_dir = "C:/Users/MB/Desktop/DCIM/130TLC00"

# Step one: create the faceblurring object
faceblurer = faceblurring.faceblurer.FaceBlurer()

# Step two: create the list of video files
vid_files = glob.glob(os.path.join(input_dir, "*.AVI"))
vid_files.sort()

# Step three: run through the videos and code the images
# for video in video files... 
# video = cv2.VideoCapture(video_path)
# video.set(cv2.CAP_PROP_FPS, 1/60)

# frame_num = 0

# while video.isOpened():
#     success, frame = video.read()
    
#     if success:
#         cv2.imwrite(f"C:/Users/MB/Desktop/DCIM/130TLC00/testout/frame_{frame_num}.jpg", frame)
#         frame_num +=1
        
#     else:
#         break

# Step four: create csv file of images

# Step five: create timelapse video

# Step six: delete from csv