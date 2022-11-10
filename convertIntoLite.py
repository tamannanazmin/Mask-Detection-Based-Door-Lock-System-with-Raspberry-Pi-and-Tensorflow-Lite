# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
#Servo
from gpiozero import Servo
myGPIO=21
servo=Servo(myGPIO)
import tensorflow as tf
#import serial
import time 

from tflite_runtime.interpreter import Interpreter
#arduino = serial.Serial('COM3', 9600)
#arduino = serial.Serial('/dev/ttyUSB0',9600)                                
lowConfidence = 0.75

def detectAndPredictMask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > lowConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
       faces = np.array(faces, dtype="float32")
       interpreter.set_tensor(input_details[0]['index'],faces)
       interpreter.invoke()
          # Retrieve detection results
       #boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
       #classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
       #scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

      # preds = maskNet.predict(faces, batch_size=32) 
       preds=interpreter.get_tensor(output_details[0]['index'])
       
    return (locs, preds)
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model_path= 'model.tflite'
#label_path="/home/pi/Downloads/label.txt"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#top_k_results=2
#get input size
#input_shape=input_details[0]['shape']
#size=input_shape[:2] if len(input_shape)==3 else input_shape[1:3]

#maskNet = load_model("mask_detector.model")
#interpreter = tf.lite.Interpreter(model_path=path)
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    (locs, preds) = detectAndPredictMask(frame, faceNet, interpreter)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        if label =="Mask":
            print("ACCESS GRANTED")
            #arduino.write(b'H')
            servo.max()
           
        else:
            print("ACCESS DENIED")
            #arduino.write(b'L')
            servo.min()
            
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.imshow("press q to quit", frame)  
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()