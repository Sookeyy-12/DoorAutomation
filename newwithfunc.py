from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pickle
import face_recognition
import numpy as np
import os

confidence = 0.9
face_match_threshold = 0.4

model = YOLO("models/new_model.pt")
classNames = ["fake", "real"]

def load_encodings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def save_encodings(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


encodeDict = load_encodings('encodings.pickle') if os.path.exists('encodings.pickle') else {}

def add_face_encoding(name, img):
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
    if encoded_faces:
        if name not in encodeDict:
            encodeDict[name] = []
        encodeDict[name].append(encoded_faces[0])
        save_encodings('encodings.pickle', encodeDict)
        print(f"Added encoding for {name}.")
    else:
        print("No face found in the provided image.")

def delete_face_encoding(name):
    if name in encodeDict:
        del encodeDict[name]
        save_encodings('encodings.pickle', encodeDict)
        print(f"Deleted encoding for {name}.")
    else:
        print(f"No encoding found for {name}.")

def recognize_faces(img):
    results = model(img, stream=True)
    is_real = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100))/100
            if conf > confidence:
                cls = int(box.cls[0])
                if classNames[cls] == "real":
                    is_real = True

    if is_real:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
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
                print(name)
            else:
                name = "unknown"

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        print("Spoofing attempt")

def main():
    print("Select an operation:")
    print("1: Recognize faces")
    print("2: Add a new face")
    print("3: Delete a face encoding")
    choice = input("Enter the number of the operation you want to perform: ")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    prev_frame_time = 0
    new_frame_time = 0

    if choice == '1':
        print("Face recognition mode selected.")
        while True:
            new_frame_time = time.time()
            success, img = cap.read()
            recognize_faces(img)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    elif choice == '2':
        print("Add face mode selected.")
        name = input("Enter the name of the person to add: ")
        while True:
            success, img = cap.read()
            add_face_encoding(name, img)
            time.sleep(2)  
            break

    elif choice == '3':
        print("Delete face mode selected.")
        name = input("Enter the name of the person to delete: ")
        delete_face_encoding(name)

    else:
        print("Invalid option selected. Exiting.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
