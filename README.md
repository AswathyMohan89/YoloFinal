# YoloFinal

Implementation of Yolo V3 . I followed the tutorial : "https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/"
I was able to convert the rectangular anchor box to a circular box. It is detecting and labelling the object, but am not able to extract one circle from that. I tried updating IOU , but I was not able to succeed in that.

The Yolo weights can be downloaded from: "https://pjreddie.com/media/files/yolov3.weights".

There is 3 main python files: detect.py, darknet.py and util.py

The detect.py is the main code, that need to be executed. The command is : `python detect.py --images dog-cycle-car.png --det det`
