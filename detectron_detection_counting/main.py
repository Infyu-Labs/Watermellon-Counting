import cv2
import os
#from dotenv import load_dotenv
#load_dotenv()

#import env
from ObjectCounter import ObjectCounter

VIDEO="/content/drive/MyDrive/watermelon_counting/watermelon.mp4"
vidObj = cv2.VideoCapture(VIDEO)
if not vidObj.isOpened():
    print("invalid")     
success, image = vidObj.read()
f_height, f_width, _ = image.shape

droi=[(0, 0), (100, 0), (100, 360), (0, 360)]
mcdf = 1
mctf = 3
detection_interval = 10
tracker = "kcf"
show_counts = True
OUTPUT_VIDEO_PATH = "/content/drive/MyDrive/watermelon_counting/output/detectron2.avi"
counting_lines = [{'label': 'A', 'line': [(100, 0), (100, 360)]}]
hud_color = (255, 0, 0)

object_counter = ObjectCounter(image, tracker, droi, mcdf, mctf, detection_interval, counting_lines, show_counts, hud_color)

output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'MJPG'), 30, (f_width, f_height))
output_frame = None
frame_count = round(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
frame_processed = 0

try:
    while success:
        object_counter.count(image)
        output_frame = object_counter.visualize()
        output_video.write(output_frame)
        frame_processed += 1
        success, image = vidObj.read()
finally:    
    vidObj.release()
    output_video.release()