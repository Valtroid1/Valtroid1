import face_recognition
import cv2
import numpy as np
import csv
import os   
import pygame
from datetime import datetime

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

janessa_image = face_recognition.load_image_file("images/janessa.jpg")
janessa_encoding = face_recognition.face_encodings(janessa_image)[0]
 
ridz_image = face_recognition.load_image_file("images/ridz.jpg")
ridz_encoding = face_recognition.face_encodings(ridz_image)[0]

joanna_image = face_recognition.load_image_file("images/joanna.jpg")
joanna_encoding = face_recognition.face_encodings(joanna_image)[0]

celest_image = face_recognition.load_image_file("images/celest.jpg")
celest_encoding = face_recognition.face_encodings(celest_image)[0]
 
known_face_encoding = [
janessa_encoding,
ridz_encoding,
joanna_encoding,
celest_encoding
]
 
known_faces_names = [
"janessa",
"ridz",
"joanna",
"celest"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
# Initialize pygame for sound alerts
pygame.init()

# Load the sound file (replace 'alert.wav' with your desired sound file)
alert_sound = pygame.mixer.Sound('siren.wav')

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Initialize variables for FPS calculation
fps_start_time = cv2.getTickCount()
fps_counter = 0
fps = 0

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if s:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index] and (1 - face_distance[best_match_index]) * 100 >= 50:
                name = known_faces_names[best_match_index]

                accuracy = (1 - face_distance[best_match_index]) * 100  # Calculate accuracy percentage
                accuracy_str = f"{accuracy:.2f}% \n"
                print(f"{name} - Accuracy: {accuracy_str}")

                # Play the alert sound when a face is detected
                alert_sound.play()

            else:
                name = "Unknown"
                print("Match Incorrect")

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                bottomLeftCornerOfText2 = (10,200)
                bottomLeftCornerOfText3 = (10,120)
                fontScale              = 1.5
                fontScale1             = 0.9
                fontColor              = (255,0,0)
                fontColor2             = (0,0,255)
                thickness              = 3
                lineType               = 2
                accuracy_percentage = f"Accuracy: {int((1 - face_distance[best_match_index]) * 100)}%"
                

                cv2.putText(frame, accuracy_percentage, 
                    bottomLeftCornerOfText2, 
                    font, 
                    fontScale,
                    fontColor2,
                    thickness,
                    lineType)

            face_names.append(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (x, y - 10)  # Adjust position for text
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 2
            cv2.putText(frame, name, bottomLeftCornerOfText, font, fontScale, fontColor, thickness)

            if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    # Calculate FPS
    fps_counter += 1
    if cv2.getTickCount() - fps_start_time > 1.0:
        fps = fps_counter / ((cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency())
        fps_counter = 0
        fps_start_time = cv2.getTickCount()

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display ethics
    cv2.putText(frame, f"By using this product, you allow the developers to use your data for research.",
                    bottomLeftCornerOfText3, 
                    font, 
                    fontScale1,
                    fontColor2,
                    thickness,
                    lineType)

    cv2.imshow("facial recognition system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()