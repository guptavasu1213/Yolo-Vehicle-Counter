**Yolo Vehicle Counter**

YoLo is a CNN architecture which specialize in object detection. I am using tutorial from https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ to develop my own modification. This project aims to count every vehicle(motorcycle, bus, car, cycle, truck)  detected in the input image/video. If you want to try this program, you must download the pretrained-model, datasets, and classes from the link above. This is how to run the program 

python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco
