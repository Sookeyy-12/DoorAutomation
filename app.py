import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

# Function to load encodings from a file
def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


encodeDict = load_encodings('encodings.pickle')

# Set threshold for face matching
face_match_threshold = 0.4

# Capture images from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matchIndex = None
        min_distance = float('inf')
        for className, encoding_list in encodeDict.items():
            for encoding in encoding_list:
                distance = face_recognition.face_distance([encoding], encode_face)
                if distance < min_distance:
                    min_distance = distance
                    matchIndex = className

        if matchIndex is not None and min_distance <= face_match_threshold:
            name = matchIndex.upper().lower()
        else:
            name = "unknown"

        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break