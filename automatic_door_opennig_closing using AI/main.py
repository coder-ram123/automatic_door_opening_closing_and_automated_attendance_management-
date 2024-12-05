import heapq
import os
import time
import pickle
import cv2
import cvzone
import numpy as np
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db
import winsound
# from pyfirmata import Arduino, SERVO


# winsound
def play_alert_sound():
    winsound.Beep(500, 200)  #frequency (500) and duration (200)


# Arduino setup
# PORT = "COM6"  # Arduino's COM port
# pin = 10
# board = Arduino(PORT)
# board.digital[pin].mode = SERVO

# Define calculate_distance function
def calculate_distance(features1, features2):
    # distance calculation logic
    pass

# rotate the servo motor by a specified angle
# def rotateServo(pin, angle):
#     board.digital[pin].write(angle)
#
# # Rotate the servo motor by 90 degrees initially
# rotateServo(pin, 220)

# Timestamp to track the starting time
start_time = time.time()

# Define your 'threshold' variable here if it's not already defined
threshold = 0.5  # Replace with your threshold value

# Function to perform A* search for face matching
def search_database(features, database):
    # Initialize an empty priority queue
    pq = []

    # Initialize the starting distance as 0
    starting_distance = 0

    # Heuristic function that estimates the distance to the nearest face
    def heuristic(face_id):
        face_features = database[face_id]
        return calculate_distance(features, face_features)

    # Add the starting state to the priority queue
    heapq.heappush(pq, (starting_distance, list(database.keys())[0]))

    while pq:
        # Get the current state with the lowest estimated distance
        current_distance, current_face_id = heapq.heappop(pq)

        # Check if the current state is the goal state
        if np.array_equal(features, database[current_face_id]):
            return current_face_id  # Found a match

        # Expand the current state by generating its neighbors
        for face_id, face_features in database.items():
            # Calculate the actual distance to the neighbor
            neighbor_distance = calculate_distance(features, face_features)

            # Calculate the estimated distance from the neighbor to the goal state
            neighbor_heuristic = heuristic(face_id)

            # Calculate the total cost of reaching the neighbor
            total_cost = current_distance + neighbor_distance

            # Add the neighbor to the priority queue with its total cost
            heapq.heappush(pq, (total_cost, face_id))

    #  no matching face was found
    return None


def is_living_person(detected_features, database):
    for face_id, face_features in database.items():
        distance = calculate_distance(detected_features, face_features)
        if distance < threshold:
            return True
    return False

# def rotateServo(pin, angle):
#     board.digital[pin].write(angle)
#
# def doorAutomate(val):
#     if val == 0:
#         rotateServo(pin, 220)  # Adjust the angles as per servo motor
#     elif val == 1:
#         rotateServo(pin, 40)


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://automaticdooropenclose-default-rtdb.firebaseio.com/",
    'storageBucket': "automaticdooropenclose.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('resources/background.png')

# importing the mode images into a list
folderModePath = 'resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

print("Loading Encode File...")
file = open('EncodingFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face recognition code
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Convert face encodings to NumPy arrays
    encodeListKnown = np.array(encodeListKnown)
    encodeCurFrame = np.array(encodeCurFrame)

    # # Check if the image is a photo based on characteristics
    is_photo = False

    if len(faceCurFrame) == 0:
        # If no faces are detected, it might be a photo
        is_photo = True

    if not is_photo:
        # If faces are detected, calculate the confidence
        face_recognition_confidence = 1.0 - np.min(np.linalg.norm(encodeListKnown - encodeCurFrame, axis=1))

        if face_recognition_confidence > 0.8:
            # If confidence is high, it might be a static photo
            is_photo = True

    if is_photo:
        # Take actions for photos
        print("Photo detected")
        play_alert_sound()


    else:
        # It's a real person
        print("Real person detected")
        # doorAutomate(1)  # Open the door

        elapsed_time = time.time() - start_time

        # If 10 seconds have passed, reverse the motor
        # if elapsed_time > 10:
        #     # Reverse the motor direction by rotating it
        #     rotateServo(pin, 220)  # Assuming reverse angle is 210
        #     break# Exit the loop after reversing the motor

    imgBackground[162:162 + 480, 55:55 + 640] = img  # Overwrite with the webcam image

    # Check if modeType is within the range of the mode images
    if modeType < len(imgModeList):
        imgBackground[44:44 + imgModeList[modeType].shape[0], 808:808 + imgModeList[modeType].shape[1]] = imgModeList[
            modeType]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

            id = studentIds[matchIndex]
            if counter == 0:
                counter = 1
                modeType = 1

    if counter != 0:
        if counter == 1:
            # Get the data
            studentInfo = db.reference(f'Students/{id}').get()
            print(studentInfo)


            # Get image from storage
            # blob = bucket.get_blob(f'images/{id}.png')
            # array = np.frombuffer(blob.download_as_string(), np.uint8)
            # imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

            #    update data
            ref = db.reference(f'Students/{id}')
            studentInfo['total_in_time'] += 1
            ref.child('total_in_time').set(studentInfo['total_in_time'])

        cv2.putText(imgBackground, str(studentInfo['total_in_time']), (861, 125),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(id), (1006, 493),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['joining_year']), (1125, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (414 - w) // 2
        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

        # imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

        counter += 1

    cv2.imshow("Face Recognition", imgBackground)
    cv2.waitKey(1)


