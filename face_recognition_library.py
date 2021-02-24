import face_recognition
import cv2
import numpy as np
import glob
import os
from pathlib import Path
from tensorflow_infer import inference
import math


def getFaceLocation_SSD(img):
    face_locations = []
    output_info = inference(img, show_result=False, draw_result=False, return_result=False, target_shape=(260,260))
    for op in output_info:
        (lb, conf, x1, y1, x2, y2) = op
        
        # scale to small (1:4)
        top = int(y1 / 4)
        right = int(x2 / 4)
        bottom = int(y2 / 4)
        left = int(x1/ 4) 
        face_locations.append((top, right, bottom, left))
    return face_locations
    
def dis2point(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    
known_face_encodings = []
known_face_names = []

'''
print('create face encocdings...')
known_people_image_file = glob.glob('known/*.*')
for f in known_people_image_file:
    image = face_recognition.load_image_file(f)

    #find face location
    face_locations = getFaceLocation_SSD(image)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_image = small_image[:, :, ::-1]
    
    if len(face_locations) > 0:
        name = Path(f).stem
        # Find all the faces and face encodings in the current frame of video
        face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

        print('name: ', name)
'''

print('create face encocdings...')
face_data_dir = 'face-data'
names = os.listdir(face_data_dir)
for name in names:
    face_images = glob.glob(face_data_dir + '/' + name + '/*.jpg')
    print('name: ', name, ' --picture: ', face_images)
    for f in face_images:
        image = face_recognition.load_image_file(f)
        #find face location
        face_locations = getFaceLocation_SSD(image)
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_image = small_image[:, :, ::-1]
        
        if len(face_locations) > 0:
            # Find all the faces and face encodings in the current frame of video
            face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

print('Total people: ',len(names))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def recognize_with_location(image, face_location):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_image = small_image[:, :, ::-1]

    face_locations = []
    face_locations.append(face_location)

    face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)

    face_encoding = face_encodings[0]


    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # print(matches)

    # # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    min_dis = np.min(face_distances)
    # print('Distance: ', min_dis, end="\t")
    if min_dis <= 0.5:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    
    # Display the results
    (top, right, bottom, left) = face_location
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Draw a box around the face
    # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    # cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return name, min_dis

def recognize(image):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_image = small_image[:, :, ::-1]

    face_locations = getFaceLocation_SSD(image)
    face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)

    # print('location: ', face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        print(matches)

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    return image



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
    # image = cv2.imread('h.jpg')
        _,image = cap.read()
        # image = recognize(image)
        face_locations = getFaceLocation_SSD(image)

        name = 'unknown'
        if len(face_locations) > 0:
            image, name = recognize_with_location(image, face_locations[0])

        print('name', name)
        # print('recognize: ', name)
        cv2.imshow('im', image)
        if cv2.waitKey(1) == 27:
            break



