import os
import subprocess

# Run detections on all files in the inputVideos directory
for fileName in os.listdir("inputVideos/"):
	lastDotIndex = fileName.rfind(".")
	# print(fileName[:lastDotIndex])
	# print("python3 yolo_video.py --input inputVideos/" + fileName + " --output outputVideos/" + \
	# 	fileName[:lastDotIndex] + ".avi --yolo yolo-coco --use-gpu 1")
	cmd = "python3 yolo_video.py --input inputVideos/" + fileName + " --output outputVideos/" + \
		fileName[:lastDotIndex] + ".avi --yolo yolo-coco --use-gpu 1"
	print("Running command:\n" + cmd)
	subprocess.run(cmd, shell=True)
