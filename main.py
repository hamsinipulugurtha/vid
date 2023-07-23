from fastapi import FastAPI, Request, File
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
import numpy as np
from matplotlib.colors import CenteredNorm
import datetime
import pickle

count = 0
frame_counter = 0
attendance_dict = {}  # Dictionary to store attendance data
timestamps=[]
# Desired square frame size
square_size = 500

with open('model.pkl', 'rb') as f:
    known_faces, known_names = pickle.load(f)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(100,0,100))
        f_gray=gray[y:y+h,x:x+w]
        f_color=img[y:y+h,x:x+w]
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if len(face_locations) == 0:
            continue
        for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encoding with the known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
        if len(matches) > 0:
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                print(name," ",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                #attendance_dict[name] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.imshow("Image",img)

    if cv2.waitKey(20) & 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()