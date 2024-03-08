# %%
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

# %%
dataset_path = 'dataset/students/KRS Door automation images'

# %%
# Function to save encodings to a file
def save_encodings(encodeDict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(encodeDict, f)

# Function to load encodings from a file
def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# %%
images = []
classNames = []
encodeDict = {}

# %%
# Check if saved encodings file exists, if not load the encodings from file
if os.path.exists('encodings.pickle'):
    encodeDict = load_encodings('encodings.pickle')
else:
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            # Get the list of image files in the subdirectory
            image_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                curImg = cv2.imread(img_path)
                if curImg is not None:
                    images.append(curImg)
                    # Append the class name (person's name) to the classNames list
                    classNames.append(person_dir)

                    # Convert image to RGB (face_recognition library requires RGB format)
                    rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                    # Find face encodings
                    face_encodings = face_recognition.face_encodings(rgb_img)
                    if len(face_encodings) > 0:
                        # If the person already exists in the dictionary, append the encoding
                        if person_dir in encodeDict:
                            encodeDict[person_dir].append(face_encodings[0])
                        # Otherwise, create a new entry in the dictionary
                        else:
                            encodeDict[person_dir] = [face_encodings[0]]
                    else:
                        print(f"No face found in the image: {img_path}")
                else:
                    print(f"Unable to read image: {img_path}")

    # Save encodings to a file
    save_encodings(encodeDict, 'encodings.pickle')

# %%
# encodeDict

# %%
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

# %%
# Release the webcam
cap.release()
cv2.destroyAllWindows()

# %%
# %pip freeze > requirements.txt

# %%
# # Always retrain the encodings
# mylist = os.listdir(dataset_path)
# for cl in mylist:
#     if cl.endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join(dataset_path, cl)
#         curImg = cv2.imread(img_path)
#         if curImg is not None:
#             images.append(curImg)
#             className = os.path.splitext(cl)[0]
#             classNames.append(className)

#             rgb_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
#             face_encodings = face_recognition.face_encodings(rgb_img)
#             if len(face_encodings) > 0:
#                 if className in encodeDict:
#                     encodeDict[className].append(face_encodings[0])
#                 else:
#                     encodeDict[className] = [face_encodings[0]]
#             else:
#                 print(f"No face found in the image: {cl}")

# # Save encodings to a file
# save_encodings(encodeDict, 'encodings.pickle')

# %%
# # Function to encode faces in the images
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         # Convert image to RGB (face_recognition library requires RGB format)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # Find face encodings
#         face_encodings = face_recognition.face_encodings(img)
#         if len(face_encodings) > 0:
#             encodeList.append(face_encodings[0])
#         else:
#             print("No face found in the image")
#     return encodeList

# %%
# # Encode faces in the loaded images
# encoded_face_train = findEncodings(images)

# %%
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             time = now.strftime('%I:%M:%S:%p')
#             date = now.strftime('%d-%B-%Y')
#             f.writelines(f'n{name}, {time}, {date}')

# %%
# train_elon_encodings = face_recognition.face_encodings(imgelon)[0]

# %%
# imgelon =face_recognition.load_image_file('dataset\\genuine\\WIN_20240307_21_52_24_Pro.jpg')
# imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
# #----------Finding face Location for drawing bounding boxes-------
# face = face_recognition.face_locations(imgelon_rgb)[0]
# copy = imgelon.copy()
# #-------------------Drawing the Rectangle-------------------------
# cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
# cv2.imshow('copy', copy)
# cv2.imshow('elon',imgelon)
# cv2.waitKey(0)

# %%
# imgelon_bgr = face_recognition.load_image_file('dataset\\genuine\\WIN_20240307_21_52_24_Pro.jpg')
# imgelon_rgb = cv2.cvtColor(imgelon_bgr,cv2.COLOR_BGR2RGB)
# cv2.imshow('bgr', imgelon_bgr)
# cv2.imshow('rgb', imgelon_rgb)
# cv2.waitKey(0)

# %%
# # lets test an image
# test = face_recognition.load_image_file('dataset\\genuine\\aman gupta.png')
# test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
# test_encode = face_recognition.face_encodings(test)[0]
# print(face_recognition.compare_faces([train_elon_encodings],test_encode))


