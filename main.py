#Import packages
import numpy as np
#import argparse
import cv2

import imutils
import time
import os
from collections import Counter


#from imutils.video import VideoStream
from imutils.video import FPS

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())
import pygame

pygame.mixer.init()
file_path = os.path.join(os.getcwd(), 'audio/siren.wav')
pygame.mixer.music.load(file_path)


protext=r"models/MobileNetSSD_deploy.prototxt.txt"
model=r"models/MobileNetSSD_deploy.caffemodel"

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
REQ_CLASSES=["bird","cat","cow","dog","horse","sheep"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#Load model
print("Loading Model...")
net=cv2.dnn.readNetFromCaffe(protext, model)
print("Starting camera feed...")

vs=cv2.VideoCapture(0, cv2.CAP_DSHOW)
#vs=VideoStream(src=0).start()
time.sleep(2)
fps=FPS().start()

#Set confidence threshold
conf_thresh=0.2

#Animal detection counter
count=[]
flag=0

#Read frame by frame
while vs:
    success, frame = vs.read()
    if not success:
      break
    frame = imutils.resize(frame, width=500)
   #Take the frame dimentions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300),
    127.5)
    net.setInput(blob)
    detections=net.forward()
        #frame detection flag
    det=0
    for i in np.arange(0, detections.shape[2]):
        #Probability associated with predictions
        confidence = detections[0,0,i,2]
        if confidence > conf_thresh:
            #Extract the class labels and dimentions of bounding box
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #draw the box with labels
            if CLASSES[idx] in REQ_CLASSES:
                det=1
                label="{}: {:.2f}%".format("Animal",confidence*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (36,255,12), 2)
                if (startY-15) > 15:
                    y = (startY - 15)
                else:
                    y = (startY+15)
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

            #Show the frames
            cv2.imshow("Frame", frame)
            count.append(det)

            #Alerts only if at least 18 frames sucessfully detects animals in the last 36 frames (appx 2 sec)
            #if flag==1 and len(count) > c+36:
                #flag=0
            if flag==1 and len(count) > c +(11*18):
                flag=0
            if Counter(count[len(count)-36:])[1] > 15 and flag==0:
                print(f"Animal Intrusion Alert...!!! {len(count)}")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():pygame.time.Clock().tick(10)
                flag=1
                c=len(count)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        fps.update()

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approximate FPS: {:.2f}".format(fps.fps()))
vs.release()
cv2.destroyAllWindows()
#vs.stop()