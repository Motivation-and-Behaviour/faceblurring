import glob
import os
import cv2
import faceblurring.faceblurer
from faceblurring.settings import *
from timeit import default_timer as timer
import csv


################
# --- INPUTS ---
part_id = "1001"
input_dir = "C:/Users/MB/Desktop/DCIM/130TLC00"

# Step zero: Generate any outputs
output_dir = os.path.join(OUTPUT_DIR, f"Participant_{part_id}")
output_dir_images = os.path.join(output_dir,"images")

if not os.path.exists(output_dir_images):
    os.makedirs(output_dir_images)

# Step one: create the faceblurring object
faceblurer = faceblurring.faceblurer.FaceBlurer()

# Step two: create the list of video files
vid_files = glob.glob(os.path.join(input_dir, "*.AVI"))
vid_files.sort()
print(f"[INFO] Found {len(vid_files)} TLC files.")

# Step three: run through the videos and code the images
img_id = 1
start_time = timer()

for vid in vid_files:
    video = cv2.VideoCapture(vid)
    video.set(cv2.CAP_PROP_FPS, 1 / 60)  # I think this can be removed?

    while video.isOpened():
        success, frame = video.read()

        if success:
            out_name = os.path.join(output_dir_images, f"{part_id}_{img_id:05}.jpg")
            img_id += 1
            faceblurer.process_frame(frame, out_name)

        else:
            break
    
    video.release()

end_time = timer()
total_time = round(end_time - start_time, 3)

print(f"[INFO] Created {img_id-1} images in {total_time} ({round((img_id-1)/total_time,1)} fps).")

# Step four: create csv file of images
image_files = glob.glob(os.path.join(output_dir_images, "*.jpg"))
image_files.sort()
out_ids = ["{:05d}".format(id) for id in range(1,img_id)]

csv_path = os.path.join(output_dir, f"Image_Log_{part_id}.csv")

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Delete"])
    writer.writerows(zip(out_ids, [""]*len(out_ids)))


# Step five: create timelapse video
start_time = timer()

out = cv2.VideoWriter(os.path.join(output_dir, "timelapse.avi"),cv2.VideoWriter_fourcc(*'DIVX'), OUT_VID_FPS, (1920,1080))

for image_file in image_files:
    frame = cv2.imread(image_file)
    frame_num = image_file[-9:-4]
    cv2.putText(frame, f"Frame: {frame_num}", 
                (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255),2, cv2.LINE_4)
    out.write(frame)

out.release()
end_time = timer()
total_time = round(end_time - start_time, 3)
print(f"[INFO] Created timelapse video in {total_time}")

# Step six: delete from csv
print("Please confirm that the csv file has been saved and closed.")
while True:
    resp = input("Is the CSV file closed?")
    if len(resp) > 0:
        break

to_delete = list()

# read the csv file back in
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row[1]):
            to_delete.append(row[0])

# First item will be the header
to_delete = to_delete[1:]

# Repad the numbers
to_delete = [id.rjust(5,"0") for id in to_delete]

print(f"[INFO] There are {len(to_delete)} files to be deleted.")

deleted = 0
for id in to_delete:
    try:
        os.remove(os.path.join(output_dir_images, f"{part_id}_{id}.jpg"))
        deleted += 1
    except:
        if DEBUG: print(f"Could not find file {part_id}_{id}.jpg")

print(f"[INFO] Deleted {deleted} files.")

# Step seven: delete the original files
if not DEBUG:
    print("Cleaning up...")
    # Original timelapse videos
    for vid_file in vid_files:
        os.remove(vid_file)
    print("\tRemoved original camera data")
    # Participant timelapse
    os.remove(os.path.join(output_dir, "timelapse.avi"))
    print("\tRemoved blurred timelapse video")
