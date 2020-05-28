import argparse
import os

# PURPOSE: Parsing the command line input and extracting the user entered values
# PARAMETERS: N/A
# RETURN:
# - Labels of COCO dataset
# - Path to the weight file
# - Path to configuration file
# - Path to the input video
# - Path to the output video
# - Confidence value
# - Threshold value
def parseCommandLineArguments():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
		help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-u", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")

	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	
	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	
	inputVideoPath = args["input"]
	outputVideoPath = args["output"]
	confidence = args["confidence"]
	threshold = args["threshold"]
	USE_GPU = args["use_gpu"]

	return LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence, threshold, USE_GPU
