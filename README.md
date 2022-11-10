# Project Title: 
#### Face Mask Detection Based Door Lock System with Raspberry Pi and Tensorflow Lite

# Project Description: 
For building face mask based door system, I used machine learning model using Tensorflow Lite in python language. After building model I used OpenCV to detect whether a person is wearing a mask or not in real time. This system contains Three devices. They are: servo, camera(webcam), and Raspberry Pi. I used Raspberry Pi to control servo, and camera. If someone appears in front of the entrance wearing a mask properly, covering both their mouth and nose, then they will be let in. But if someone appears without a mask then they will be denied entry. This versatile system could be used with variety of entrances with different locking system. 

# How to run the project

#### Step 1
You have to run 'train_mask.py' file to create a model named 'model.tflite' or you can use my previous trained model which I have provided in GitHub.
For training 'train_mask.py' file you need a dataset having two folders: 1. withMask 2. Withoutmask. I also provided the dataset in GitHub.

#### Step 2
You have to run ‘convertIntoLite.py’ file. For running this file you need 'res10_300x300_ssd_iter_140000.caffemodel',  'deploy.prototxt' and 'model.tflite' file which I have provided in the Github and you must connect your webcam with your Raspberry Pi.
If you run everything successfully then your camera will be opened and you can test if it works with the mask or not.

#### Step 3
Finally, you will add Serial Command to the facemask detection algorithm that will order the Raspberry Pi to send commands to the servo based on the state of detection. 
Remember you have to connect your Raspberry Pi with servo before running the code. 

# Language: 
Python 
# Library:
tensorflow, keras, imutils, cv2, numpy, time, os, serial
# Hardware:
Raspberry Pi, servo, and webcam.

