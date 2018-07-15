#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:47:56 2017

@author: mohit jain

The haar cascade can be downloaded from https://github.com/opencv/opencv/tree/master/data/haarcascades 
and ftp://mozart.dis.ulpgc.es/pub/Software/HaarClassifiers/FaceFeaturesDetectors.zip
"""
import cv2

#load cascades for detecting features
faceFront_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eyeDetect_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileDetect_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
upperbodyDetect_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
noseDetect_cascade = cv2.CascadeClassifier('Nariz.xml')
headAndSoulder_cascade = cv2.CascadeClassifier('HS.xml')

#make a function to detect face, eyes, upperbody and smile
def detectFeatures(gray, frame):
    frontFace = faceFront_cascade.detectMultiScale(gray, 1.3, 5) #adjust parameters based on your conditions
    
    #for face, eyes and smile
    for (x_topLeft, y_topLeft, width, height) in frontFace:
        cv2.rectangle(frame, (x_topLeft, y_topLeft), (x_topLeft+width, y_topLeft+height), (125, 125, 125), 1)
        #for eye, nose and smile detection select an area of interest within the face
        regionOfInterset_gray = gray[y_topLeft:y_topLeft+height, x_topLeft:x_topLeft+width]
        regionOfInterset_color = frame[y_topLeft:y_topLeft+height, x_topLeft:x_topLeft+width]
        eyes = eyeDetect_cascade.detectMultiScale(regionOfInterset_gray, 1.1, 3) #adjust parameters based on your conditions
        for (eyeX_topLeft, eyeY_topLeft, eyeWidth, eyeHeight) in eyes:
            cv2.rectangle(regionOfInterset_color, (eyeX_topLeft, eyeY_topLeft), (eyeX_topLeft+eyeWidth, eyeY_topLeft+eyeHeight), (100, 0, 0), 1)
        
        #smile detection
        smile = smileDetect_cascade.detectMultiScale(gray, 2.7, 30) #adjust parameters based on your conditions
         #print smile found
        if(len(smile) > 0):
            print("Smiling face found =", len(smile))
        for (smileX_topLeft, smileY_topLeft, smileWidth, smileHeight) in smile:
            cv2.rectangle(regionOfInterset_color, (smileX_topLeft, smileY_topLeft), (smileX_topLeft+smileWidth, smileY_topLeft+smileHeight), (255, 255, 0), 1)
            
        #nose detect
        nose = noseDetect_cascade.detectMultiScale(regionOfInterset_gray, 1.1, 22) #adjust parameters based on your conditions
        for (noseX_topLeft, noseY_topLeft, noseWidth, noseHeight) in nose:
            cv2.rectangle(regionOfInterset_color, (noseX_topLeft, noseY_topLeft), (noseX_topLeft+noseWidth, noseY_topLeft+noseHeight), (255, 255, 255), 1)
            
    headAndSoulder = headAndSoulder_cascade.detectMultiScale(gray, 1.9, 10) #adjust parameters based on your conditions
    for (hsX_topLeft, hsY_topLeft, hsWidth, hsHeight) in headAndSoulder:
        cv2.rectangle(frame, (hsX_topLeft, hsY_topLeft), (hsX_topLeft+width, hsY_topLeft+height), (255, 0, 255), 1)
            
    return frame

#Face recognition with a USB webcam

getVideo = cv2.VideoCapture(0) #only one camera is connected

#start the loop and press q to stop

while(True):
    #capture each frame
    _, frame = getVideo.read()
    
    #convert to gray for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    startDetetcting = detectFeatures(gray, frame)
    
    #Display the frame and press q to stop (make sure frame window is selected)
    cv2.imshow('camera frame', startDetetcting)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

#this will close the camera window    
getVideo.release()
cv2.destroyAllWindows()    


