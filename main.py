import io
import tempfile
import threading

import numpy as np
import cv2
import face_recognition
import time
from pathlib import Path
from pytesseract import *
from difflib import SequenceMatcher
import os
import pandas as pd
import tkinter as tk
import enchant
from PIL import Image, ImageTk
import pyttsx3
from imutils.object_detection import non_max_suppression

def faceRecognitionTest():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        face_image = face_recognition.load_image_file("face_images/Kristof.jpg")
        cv2.imwrite('face_images/temp.jpg', frame)
        cv2.imshow("preview", frame)

        image = face_recognition.load_image_file("face_images/temp.jpg")

        if (len(face_recognition.face_encodings(face_image)) > 0):
            face_encoding = face_recognition.face_encodings(face_image)[0]
        else:
            continue

        if (len(face_recognition.face_encodings(image)) > 0):
            unknown_encoding = face_recognition.face_encodings(image)[0]
        else:
            continue

        res = face_recognition.compare_faces([face_encoding], unknown_encoding)
        print(res)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


def detectFace(model, frame, threshold=0.75):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    model.setInput(blob)
    detections = model.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, 4)
    return frame, faceBoxes


def calculate_overlap_area(face_box1, face_box2):
    x1 = max(face_box1[0], face_box2[0])
    y1 = max(face_box1[1], face_box2[1])
    x2 = min(face_box1[2], face_box2[2])
    y2 = min(face_box1[3], face_box2[3])

    width = max(0, x2 - x1 + 1)
    height = max(0, y2 - y1 + 1)

    overlap_area = (width * height) / ((face_box1[2] - face_box1[0] + 1) * (face_box1[3] - face_box1[1] + 1))
    return overlap_area


# MTCNN
"""def detect_face(image, face_size_prop = None):
    faceProportion=None
    shape = image.shape
    image_size = shape[0] * shape[1]

    if face_size_prop is None:
        face_detector = MTCNN()
    else:
        faceProportion = int(2 * math.sqrt(image_size * float(face_size_prop)))
        face_detector = MTCNN(min_face_size=faceProportion)

    faces = face_detector.detect_faces(image)
    image = boxFace(image, faces, (255, 0, 0))
    return image, faces


def boxFace(frame, boxes, color):

    # for each face, draw a rectangle based on coordinates
    for box in boxes:
        x, y, width, height = box['box']
        x = round(x)
        y = round(y)
        width = round(width)
        height = round(height)

        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        return frame"""

slo_dict = enchant.Dict('sl_SI')

configFace = "gad/opencv_face_detector.pbtxt"
modelFace = "gad/opencv_face_detector_uint8.pb"
faceNeuralNet = cv2.dnn.readNet(modelFace, configFace)

configAge = "gad/age_deploy.prototxt"
modelAge = "gad/age_net.caffemodel"
ageNeuralNet = cv2.dnn.readNet(modelAge, configAge)

configGender = "gad/deploy_gender.prototxt"
modelGender = "gad/gender_net.caffemodel"
genderNeuralNet = cv2.dnn.readNet(modelGender, configGender)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-22)', '(23-34)', '(35-45)', '(46-59)', '(60-100)']
genderList = ['Male', 'Female']

# person_image = face_recognition.load_image_file("face_images/Kristof.jpg")
# person_encoding = face_recognition.face_encodings(person_image)[0]

known_faces = []
known_faces_names = []

isTextDetectionOK = True
num_no_faces = 0

engine = pyttsx3.init(driverName='sapi5')
voices = engine.getProperty('voices')
for voice in voices:
    if voice.name == "Maja eBralec":
        voice_id = voice.id
engine.setProperty('voice', voice_id)


for p in Path('.').glob('face_images/*.jpg'):
    person_image = face_recognition.load_image_file("face_images/" + p.name)
    person_encoding = face_recognition.face_encodings(person_image)
    if len(person_encoding) > 0:
        known_faces.append(person_encoding[0])
        known_faces_names.append(p.name[0:len(p.name) - 4])


def ageGender(frame, face, faceBox, genderAgeData):
    # print("AGE GENDER")
    new_gender_age_data = [0, 0, 0, 0]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNeuralNet.setInput(blob)
    genderPreds = genderNeuralNet.forward()
    i = genderPreds[0].argmax()
    genderConfidence = genderPreds[0][i]

    if genderConfidence > genderAgeData[1]:
        new_gender_age_data[0] = i
        new_gender_age_data[1] = genderConfidence
    else:
        new_gender_age_data[0] = genderAgeData[0]
        new_gender_age_data[1] = genderAgeData[1]

    gender = genderList[genderAgeData[0]]

    ageNeuralNet.setInput(blob)
    agePreds = ageNeuralNet.forward()
    j = agePreds[0].argmax()
    ageConfidence = agePreds[0][j]

    if ageConfidence > genderAgeData[3]:
        new_gender_age_data[2] = j
        new_gender_age_data[3] = ageConfidence
    else:
        new_gender_age_data[2] = genderAgeData[2]
        new_gender_age_data[3] = genderAgeData[3]

    age = ageList[new_gender_age_data[2]]

    cv2.putText(frame, f'{gender}({round(new_gender_age_data[1], 2)}), {age}({round(new_gender_age_data[3], 2)})', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    age = age[1:len(age)-1]
    age = age.split("-")
    prompt = ""
    if gender == "Male":
        prompt = f"Moški med {age[0]} in {age[1]} let"
    else:
        prompt = f"Ženska med {age[0]} in {age[1]} let"

    if age[1] == "3":
        prompt = prompt + "a"

    return frame, prompt, new_gender_age_data


def faceRecognition(frame, face, faceBox, genderAgeData):
    # print("FACE RECOGNITION")
    face_smaller = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)

    rgb_face = cv2.cvtColor(face_smaller, cv2.COLOR_BGR2RGB)

    face_locations = [
        (0 + 2, rgb_face.shape[1] - 2, rgb_face.shape[0] - 2, 0 + 2)
    ]
    encoding_test_faces = face_recognition.face_encodings(rgb_face, face_locations)

    if len(encoding_test_faces) > 0:
        encoding_test_face = encoding_test_faces[0]

        i = 0
        while i < len(known_faces):
            if known_faces[i].size == 0:
                del known_faces[i]
                del known_faces_names[i]
            else:
                i += 1

        match = face_recognition.compare_faces(known_faces, encoding_test_face)

        for i in range(0, len(match)):
            if match[i]:
                cv2.putText(frame, known_faces_names[i], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                return frame, True, known_faces_names[i], genderAgeData

    frame, textAgeGender, gender_age_data = ageGender(frame, face, faceBox, genderAgeData)
    return frame, False, textAgeGender, gender_age_data


def addNewEncoding(face_id):
    # print("ADD ENCODING")
    known_faces.append(np.array([]))
    index = len(known_faces) - 1
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    known_faces_names.append(str(face_id))
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def changeEncoding(face_id):
    # print("CHANGE ENCODING")
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    index = known_faces_names.index(str(face_id))
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def saveFaceImage(frame, face_positions, time_at_end, prev_face_positions):
    # print("SAVE FACE IMAGE")
    for face_id, face_data in face_positions.items():
        face = frame[max(0, face_data[0][1] - 10): min(face_data[0][3] + 10, frame.shape[0] - 1), max(0, face_data[0][0] - 10): min(face_data[0][2] + 10, frame.shape[1] - 1)]
        if face_data[1][len(face_data[1])-1]:
            continue
        if time_at_end - face_data[3] > 4:
            if os.path.exists('face_images/' + str(face_id) + '.jpg'):
                continue
            cv2.imwrite('face_images/'+str(face_id)+'.jpg', face)
            # thread1 = threading.Thread(target=addNewEncoding, args=(face_id,))
            # thread1.start()
            known_faces.append(np.array([]))
            index = len(known_faces) - 1
            new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
            known_faces_names.append(str(face_id))
            new_person_encoding = face_recognition.face_encodings(new_person_image)
            if len(new_person_encoding) > 0:
                known_faces[index] = new_person_encoding[0]

        elif time_at_end - face_data[3] > 1.5:
            if prev_face_positions == {} or face_data[4][1] < 0.9 or face_id not in prev_face_positions: #or face_id > len(prev_face_positions)-1:
                continue
            # print("PREV FACE[face_id]:", prev_face_positions[face_id])
            if prev_face_positions[face_id][4][1] != face_data[4][1]:
                cv2.imwrite('face_images/'+str(face_id)+'.jpg', face)
                # thread1 = threading.Thread(target=addNewEncoding, args=(face_id,))
                # thread2 = threading.Thread(target=changeEncoding, args=(face_id,))
                if str(face_id) not in known_faces_names:
                    # print("NEW FACE ADDED")
                    # thread1.start()

                    known_faces.append(np.array([]))
                    index = len(known_faces) - 1
                    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
                    known_faces_names.append(str(face_id))
                    new_person_encoding = face_recognition.face_encodings(new_person_image)
                    if len(new_person_encoding) > 0:
                        known_faces[index] = new_person_encoding[0]
                else:
                    # print("CHANGED FACE ENCOD")
                    # thread2.start()
                    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
                    new_person_encoding = face_recognition.face_encodings(new_person_image)
                    index = known_faces_names.index(str(face_id))
                    if len(new_person_encoding) > 0:
                        known_faces[index] = new_person_encoding[0]

    return face_positions


previous_thread = None

def auditoryPrompt(face_positions, frame_shape, time_at_end):
    # print("AUDITORY PROMPT")
    global previous_thread
    global num_no_faces
    all_faces_none = True
    for face_id, face_data in face_positions.items():
        if face_data[0] is None:
            continue
        all_faces_none = False
        if time_at_end - face_data[3] > 4:
            name_prompt = face_id

            if face_data[5] > 0:
                name_prompt = "Spet je tu "

                if face_id.isdigit():
                    gender = genderList[face_data[4][0]]
                    age = ageList[face_data[4][2]]
                    age = age[1:len(age) - 1]
                    age = age.split("-")
                    if gender == "Male":
                        name_prompt += f"moški med {age[0]} in {age[1]} let"
                    else:
                        name_prompt += f"ženska med {age[0]} in {age[1]} let"

                    if age[1] == "3":
                        name_prompt += "a"

                    name_prompt += " in"

                else:
                    name_prompt += face_id + " in"

            elif face_id.isdigit():
                gender = genderList[face_data[4][0]]
                age = ageList[face_data[4][2]]
                age = age[1:len(age) - 1]
                age = age.split("-")
                if gender == "Male":
                    name_prompt = f"Moški med {age[0]} in {age[1]} let"
                else:
                    name_prompt = f"Ženska med {age[0]} in {age[1]} let"

                if age[1] == "3":
                    name_prompt += "a"

            if (face_data[0][0] / frame_shape[1] + face_data[0][2] / frame_shape[1]) / 2 < 0.35:
                name_prompt = name_prompt + " je na levi."
            elif (face_data[0][0] / frame_shape[1] + face_data[0][2] / frame_shape[1]) / 2 > 0.65:
                name_prompt = name_prompt + " je na desni."
            else:
                name_prompt = name_prompt + " je naravnost pred vami."

            # print("before speech:", name_prompt)
            if previous_thread and previous_thread.is_alive():
                face_positions[str(face_id)][3] = face_data[3] + 1
            else:
                face_positions[str(face_id)][5] += 1
                face_positions[str(face_id)][3] = face_data[3] + 45

            print(name_prompt)
            new_thread = None
            if not previous_thread or not previous_thread.is_alive():
                new_thread = threading.Thread(target=speech, args=(name_prompt,))
                new_thread.start()

            if new_thread:
                previous_thread = new_thread
            # print("after speech:", name_prompt)

    if all_faces_none:
        num_no_faces += 1
    else:
        num_no_faces = 0
    return face_positions


def faceMain(frame_org, prev_face_positions, face_id_counter):
    # print("FACE MAIN")
    frame = cv2.cvtColor(frame_org.copy(), cv2.COLOR_BGR2RGB)
    current_face_positions = {}  # DICT: faceBox, []array(length: 2) was face recognized in prev frames, text for display, time, array for max conf ageGender (gender index, gender max conf, age index, age max conf), readout #times

    result, faces = detectFace(faceNeuralNet, frame.copy())

    # if not faces:
    # print("No face detected")
    # print("Prev: ", prev_face_positions)

    for faceBox in faces:
        face = frame[max(0, faceBox[1] - 10): min(faceBox[3] + 10, frame.shape[0] - 1), max(0, faceBox[0] - 10): min(faceBox[2] + 10, frame.shape[1] - 1)]
        if faceBox[0] + face.shape[1] > frame.shape[1] or faceBox[0] < 0 or faceBox[1] <= 0 or faceBox[1] + face.shape[0] > frame.shape[0]:
            continue
        if face.shape[0] < 1 or face.shape[1] < 1:
            continue

        # print(face.shape)
        if len(prev_face_positions) > 0:
            found_matching_face = False
            for prev_id, prev_face_box in prev_face_positions.items():
                if prev_face_box[0] is not None:
                    overlap_area = calculate_overlap_area(faceBox, prev_face_box[0])
                else:
                    overlap_area = 0

                # cv2.imshow("face", face)
                if overlap_area > 0.5:
                    if len(prev_face_box[1]) < 2:
                        frame, isFaceDetected, faceText, gender_age_data = faceRecognition(frame, face, faceBox, prev_face_box[4])
                        if isFaceDetected and not prev_face_box[1][0]:
                            current_face_positions[faceText] = [faceBox, [prev_face_box[1][0], isFaceDetected], faceText, prev_face_box[3], gender_age_data, prev_face_box[5]]
                        else:
                            current_face_positions[str(prev_id)] = [faceBox, [prev_face_box[1][0], isFaceDetected], faceText, prev_face_box[3], gender_age_data, prev_face_box[5]]
                    else:
                        if not any(prev_face_box[1]):
                            frame, textAgeGender, gender_age_data = ageGender(frame, face, faceBox, prev_face_box[4])
                            current_face_positions[str(prev_id)] = [faceBox, prev_face_box[1], textAgeGender, prev_face_box[3], gender_age_data, prev_face_box[5]]
                        elif all(prev_face_box[1]):
                            cv2.putText(frame, prev_face_box[2], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                            current_face_positions[str(prev_id)] = [faceBox, prev_face_box[1], prev_face_box[2], prev_face_box[3], prev_face_box[4], prev_face_box[5]]
                        else:
                            frame, isFaceDetected, faceText, gender_age_data = faceRecognition(frame, face, faceBox, prev_face_box[4])
                            if isFaceDetected and not prev_face_box[1][0]:
                                current_face_positions[faceText] = [faceBox, [prev_face_box[1][1], isFaceDetected], faceText, prev_face_box[3], gender_age_data, prev_face_box[5]]
                            else:
                                current_face_positions[str(prev_id)] = [faceBox, [prev_face_box[1][1], isFaceDetected], faceText, prev_face_box[3], gender_age_data, prev_face_box[5]]
                    found_matching_face = True
                    break

            if not found_matching_face:
                frame, isFaceDetected, faceText, ageGenderData = faceRecognition(frame, face, faceBox, [0, 0, 0, 0])
                shown_number_of_times = 0
                if isFaceDetected:
                    if faceText in prev_face_positions:
                        ageGenderData = prev_face_positions[faceText][4]
                        shown_number_of_times = prev_face_positions[faceText][5]
                    current_face_positions[faceText] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData, shown_number_of_times]
                else:
                    face_id_counter += 1
                    current_face_positions[str(face_id_counter)] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData, 0]

        else:
            frame, isFaceDetected, faceText, ageGenderData = faceRecognition(frame, face, faceBox, [0, 0, 0, 0])
            shown_number_of_times = 0
            if isFaceDetected:
                if faceText in prev_face_positions:
                    ageGenderData = prev_face_positions[faceText][4]
                    shown_number_of_times = prev_face_positions[faceText][5]
                current_face_positions[faceText] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData, shown_number_of_times]
            else:
                face_id_counter += 1
                current_face_positions[str(face_id_counter)] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData, 0]

    time_at_end = time.time()
    current_face_positions = saveFaceImage(frame_org, current_face_positions, time_at_end, prev_face_positions)

    # ADD lost faces back to dict
    for prev_face, prev_face_data in prev_face_positions.items():
        found_match = False
        for current_face, current_face_data in current_face_positions.items():
            if current_face == prev_face:
                found_match = True
        if not found_match:
            current_face_positions[prev_face] = [None, prev_face_data[1], prev_face_data[2], prev_face_data[3], prev_face_data[4], prev_face_data[5]]


    # print("current:", current_face_positions)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), auditoryPrompt(current_face_positions, frame.shape, time_at_end), face_id_counter


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def detectBlur(frame, size=60, thresh=10):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (height, width) = gray.shape
    (centerX, centerY) = (int(width / 2.0), int(height / 2.0))

    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)

    fftShift[centerY - size:centerY + size, centerX - size:centerX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return (mean, mean <= thresh)


def combine_lists_with_overlap(previous_frame, current_frame):
    global isTextDetectionOK
    prev_index_3overlap = []
    current_index_3overlap = []
    for i, prev_word in enumerate(previous_frame):
        for j, current_word in enumerate(current_frame):
            if previous_frame[i][0] == current_frame[j][0]:
                # Check 2 more words
                if i+3 < len(previous_frame) and j+3 < len(current_frame):
                    if previous_frame[i+1][0] == current_frame[j+1][0] and previous_frame[i+2][0] == current_frame[j+2][0] and previous_frame[i+3][0] == current_frame[j+3][0]:
                        prev_index_3overlap.append(i)
                        current_index_3overlap.append(j)
                        break
                else:
                    continue

    if len(prev_index_3overlap) < 1:
        if not isTextDetectionOK:
            return []
        isTextDetectionOK = False
        return previous_frame
    else:
        isTextDetectionOK = True

    # print(prev_index_3overlap)
    # print(current_index_3overlap)
    # For every found overlap add to combined, if gaps fill in

    combined = []


    for i in range (0, prev_index_3overlap[0]):
        # print("DODDAJ NA ZAČETKU IZ PREJŠNJEGA")
        combined.append(previous_frame[i])

    for i in range(0, len(prev_index_3overlap)):
        combined.append(current_frame[current_index_3overlap[i]])
        if previous_frame[prev_index_3overlap[i]][2] == float('inf'): #TIME
            combined[len(combined)-1][2] = float('inf')
        if i+1 < len(prev_index_3overlap):
            if prev_index_3overlap[i+1] - prev_index_3overlap[i] == current_index_3overlap[i+1] - current_index_3overlap[i] and prev_index_3overlap[i+1] - prev_index_3overlap[i] > 1:
                diff = prev_index_3overlap[i+1] - prev_index_3overlap[i]
                for j in range(1, diff):
                    if previous_frame[prev_index_3overlap[i]+j][1] > current_frame[current_index_3overlap[i]+j][1]:
                        combined.append(previous_frame[prev_index_3overlap[i]+j])
                    else:
                        combined.append(current_frame[current_index_3overlap[i]+j])
                    if previous_frame[prev_index_3overlap[i]+j][2] == float('inf'):#TIME
                        combined[len(combined) - 1][2] = float('inf')
            elif prev_index_3overlap[i+1] - prev_index_3overlap[i] > current_index_3overlap[i+1] - current_index_3overlap[i] and prev_index_3overlap[i+1] - prev_index_3overlap[i] > 1:
                diff = prev_index_3overlap[i+1] - prev_index_3overlap[i]
                for j in range(1, diff):
                    combined.append(previous_frame[prev_index_3overlap[i]+j])
                    if previous_frame[prev_index_3overlap[i+1]][2] == float('inf'):#TIME
                        combined[len(combined) - 1][2] = float('inf')
            elif prev_index_3overlap[i+1] - prev_index_3overlap[i] < current_index_3overlap[i+1] - current_index_3overlap[i] and current_index_3overlap[i+1] - current_index_3overlap[i] > 1:
                diff = current_index_3overlap[i+1] - current_index_3overlap[i]
                for j in range(1, diff):
                    combined.append(current_frame[current_index_3overlap[i]+j])
                    if previous_frame[prev_index_3overlap[i+1]][2] == float('inf'):#TIME
                        combined[len(combined) - 1][2] = float('inf')

    # print(len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1], "PRIMERJAMO Z", len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1])
    if len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] == len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        # print("NA KONCU MEŠANO")
        diff = len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1]
        for j in range(1, diff):
            if previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j][1] > current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j][1]:
                combined.append(previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j])
            else:
                combined.append(current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j])
            if previous_frame[prev_index_3overlap[len(prev_index_3overlap) - 1] + j][2] == float('inf'):  # TIME
                combined[len(combined) - 1][2] = float('inf')
    elif len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] > len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        # print("NA KONCU IZ PREVIOUS")
        diff = len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1]
        for j in range(1, diff):
            combined.append(previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j])
            if previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j][2] == float('inf'):
                combined[len(combined) - 1][2] = float('inf')
    elif len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] < len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        # print("NA KONCU IZ CURRENT")
        diff = len(current_frame) - current_index_3overlap[len(current_index_3overlap) - 1]
        if current_index_3overlap[0] - current_index_3overlap[len(current_index_3overlap)-1] > 0:
            # print("ZAZNANO PONAVLJANJE")
            diff = current_index_3overlap[0] - current_index_3overlap[len(current_index_3overlap)-1]
        for j in range(1, diff):
            combined.append(current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j])

    if current_index_3overlap[0] > 4:
        if current_index_3overlap[0] < current_index_3overlap[len(current_index_3overlap) - 1]:
            # print("DODAJ NA KONEC IZ ZAČETKA ZDAJŠNJEGA")
            for i in range(0, current_index_3overlap[0]):
                combined.append(current_frame[i])

    # print("COMBINED:", combined)
    return combined


def textMain(frame, read_queue):
    global previous_thread
    # print("================================")
    pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame, (5, 5), 1)
    # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    text_res = pytesseract.image_to_data(frame, lang='slv', output_type='data.frame')
    text_res = text_res[text_res.conf >= 85]
    filtered_rows = []
    for index, row in text_res.iterrows():
        word = row['text']
        if not isinstance(word, str):
            continue
        if word.strip() == '':
            continue
        word_without_symbol = ''.join(letter for letter in word if letter.isalnum())
        if word_without_symbol.strip() == '':
            continue
        if word_without_symbol[0].isupper():
            filtered_rows.append(row)
            continue
        if slo_dict.check(word_without_symbol):
            filtered_rows.append(row)

    # Create a new DataFrame containing the filtered rows
    text_res = pd.DataFrame(filtered_rows)
    # print(text_res)
    lines = []
    time_lines = time.time()
    for index, row in text_res.iterrows():
        word = row['text']
        conf = row['conf']
        lines.append([word, conf, time_lines])

    # print("LINES:", lines)
    # print("READ QUEUE", read_queue)
    if lines == []:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), read_queue

    if read_queue == []:
        read_queue = lines
    else:
        read_queue = combine_lists_with_overlap(read_queue, lines)

    # combined_string = ""
    # for item in read_queue:
        # combined_string += item[0]

    # Using list comprehension
    # combined_string = ' '.join([item[0] for item in read_queue])

    # print("before speech:", len(read_queue))
    # asyncio.run(speech(name_prompt))

    new_thread = None
    if not previous_thread or not previous_thread.is_alive():
        num_word = 0
        prompt = ''
        # print(read_queue)
        for i in range(0, len(read_queue)):
            # print(read_queue[i][2])
            if read_queue[i][2] == float('inf'):
                continue
            num_word += 1
            read_queue[i][2] = float('inf')
            prompt = prompt + " " + read_queue[i][0]
            if num_word > 10 and ('.' in read_queue[i][0] or '!' in read_queue[i][0] or '?' in read_queue[i][0]):
                break
            if num_word > 17 and (',' in read_queue[i][0]):
                break
            if num_word > 25:
                break
        # print(read_queue)
        print(prompt)
        new_thread = threading.Thread(target=speech, args=(prompt,))
        new_thread.start()

    if new_thread:
        previous_thread = new_thread
    # print("after speech:")

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), read_queue


layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet('gad/frozen_east_text_detection.pb')

def detectTextArea(frame):
    orig = frame.copy()
    (H, W) = orig.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (480, 480)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(frame, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(orig, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    num_word_in_middle = 0
    text_block_on_left_and_right = False
    for box in boxes:
        x, y, x_end, y_end = box
        # print("x, x_end", x, x_end, (x+x_end)/2)
        # print("newW", newW)
        if (x+x_end) / 2 > newW * 0.35 and (x+x_end) / 2 < newW * 0.65:
            num_word_in_middle += 1
        min_x = min(min_x, x * rW)
        min_y = min(min_y, y * rH)
        max_x = max(max_x, x_end * rW)
        max_y = max(max_y, y_end * rH)
        cv2.rectangle(orig, (int(x * rW), int(y * rH)), (int(x_end * rW), int(y_end * rH)), (0, 255, 0), 1)

    if len(boxes) > 2:
        if num_word_in_middle/len(boxes) < 0.05:
            text_block_on_left_and_right = True
    (frame_H, frame_W) = frame.shape[:2]
    if min_x == float('inf') or min_y == float('inf') or max_x == float('inf') or max_y == float('inf'):
        return orig, (0, 0, frame_W-1, frame_H-1), False
    min_x = int(min_x)
    min_y = int(min_y)
    max_x = int(max_x)
    max_y = int(max_y)

    # Padding
    min_x = max(0, min_x - 10)
    min_y = max(0, min_y - 10)
    max_x = min(max_x + 10, frame_W-1)
    max_y = min(max_y + 10, frame_H-1)

    # Draw the final bounding box on the original image
    cv2.rectangle(orig, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    return orig, (min_x, min_y, max_x, max_y), text_block_on_left_and_right


def main():
    use_face_or_text_main = True
    num_blurry = 0

    def update_frame(prev_frame_time, prev_face_positions, face_id_counter, read_queue):
        nonlocal use_face_or_text_main
        nonlocal num_blurry
        global previous_thread
        global num_no_faces
        rval, frame = vc.read()
        if rval:

            if use_face_or_text_main:
                (mean, blurry) = detectBlur(frame)
                if blurry:
                    num_blurry += 1
                else:
                    num_blurry = 0
                color = (0, 0, 255) if blurry else (0, 255, 0)
                textBlur = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                textBlur = textBlur.format(mean)
                cv2.putText(frame, textBlur, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                if num_blurry == 0:
                    frame, prev_face_positions, face_id_counter = faceMain(frame, prev_face_positions, face_id_counter)
                    if num_no_faces == 20:
                        new_thread = None
                        if not previous_thread or not previous_thread.is_alive():
                            print(' Pred vami ni zaznanih oseb!')
                            new_thread = threading.Thread(target=speech, args=('Pred vami ni zaznanih oseb!',))
                            new_thread.start()
                        if new_thread:
                            previous_thread = new_thread
                elif num_blurry == 10:
                    new_thread = None
                    if not previous_thread or not previous_thread.is_alive():
                        print(' Slika je zamegljena, stojte pri miru!')
                        new_thread = threading.Thread(target=speech, args=('Slika je zamegljena, stojte pri miru!',))
                        new_thread.start()
                    if new_thread:
                        previous_thread = new_thread
            else:
                frame, (min_x, min_y, max_x, max_y), text_block_on_left_and_right = detectTextArea(frame)
                # print(min_x, min_y, max_x, max_y, frame.shape)
                if min_x < 10 and frame.shape[1] - max_x < 10 and text_block_on_left_and_right:
                    new_thread = None
                    if not previous_thread or not previous_thread.is_alive():
                        print(' Celo besedilo ni v objektivu, poglejte levo ali desno!')
                        new_thread = threading.Thread(target=speech, args=('Celo besedilo ni v objektivu, poglejte levo ali desno!',))
                        new_thread.start()
                    if new_thread:
                        previous_thread = new_thread
                elif min_x < 10 and frame.shape[1] - max_x < 10:
                    new_thread = None
                    if not previous_thread or not previous_thread.is_alive():
                        print(' Celo besedilo ni v objektivu, stopite nazaj!')
                        new_thread = threading.Thread(target=speech, args=('Celo besedilo ni v objektivu, stopite nazaj!',))
                        new_thread.start()
                    if new_thread:
                        previous_thread = new_thread
                elif min_x < 10:
                    new_thread = None
                    if not previous_thread or not previous_thread.is_alive():
                        print(' Poglejte malo bolj levo, ker je besedilo na robu!')
                        new_thread = threading.Thread(target=speech, args=('Poglejte malo bolj levo, ker je besedilo na robu!',))
                        new_thread.start()
                    if new_thread:
                        previous_thread = new_thread
                elif frame.shape[1] - max_x < 10:
                    new_thread = None
                    if not previous_thread or not previous_thread.is_alive():
                        print(' Poglejte malo bolj desno, ker je besedilo na robu!')
                        new_thread = threading.Thread(target=speech, args=('Poglejte malo bolj desno, ker je besedilo na robu!',))
                        new_thread.start()
                    if new_thread:
                        previous_thread = new_thread
                else:
                    (mean, blurry) = detectBlur(frame[min_y:max_y, min_x:max_x])
                    if blurry:
                        num_blurry += 1
                    else:
                        num_blurry = 0
                    color = (0, 0, 255) if blurry else (0, 255, 0)
                    textBlur = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                    textBlur = textBlur.format(mean)
                    cv2.putText(frame, textBlur, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                    if num_blurry == 0:
                        _, read_queue = textMain(frame[min_y:max_y, min_x:max_x], read_queue)
                    elif num_blurry == 10:
                        new_thread = None
                        if not previous_thread or not previous_thread.is_alive():
                            print('Slika je zamegljena, stojte pri miru!')
                            new_thread = threading.Thread(target=speech, args=('Slika je zamegljena, stojte pri miru!',))
                            new_thread.start()
                        if new_thread:
                            previous_thread = new_thread

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(round(fps, 2))
            cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Display the frame in Tkinter window
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

        window.after(10, update_frame, prev_frame_time, prev_face_positions, face_id_counter, read_queue)  # Update every 10 milliseconds

    def toggle_functionality():
        nonlocal use_face_or_text_main
        nonlocal toggle_button
        use_face_or_text_main = not use_face_or_text_main
        if use_face_or_text_main:
            toggle_button.config(text='Spremeni na zaznavo besedila')
        else:
            toggle_button.config(text='Spremeni na zaznavo obrazov')

    vc = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _, frame = vc.read()

    window = tk.Tk()  # Makes main window
    window.wm_title("Pomoč slabovidnim")
    window.config(background="#FFFFFF")

    use_face_or_text_main = True
    toggle_button = tk.Button(window, text='Spremeni na zaznavo besedila', command=toggle_functionality)
    toggle_button.grid(row=1, column=0, padx=10, pady=2)

    # Graphics window
    imageFrame = tk.Frame(window, width=1280, height=720)
    imageFrame.grid(row=0, column=0, padx=10, pady=2)

    lmain = tk.Label(imageFrame)
    lmain.grid(row=0, column=0)

    update_frame(0, {}, 0, [])  # Start the periodic frame update

    window.mainloop()

    vc.release()


def speech(text):
    # print("speech start")
    engine.say(text)
    engine.runAndWait()
    # print("speech end")


if __name__ == '__main__':

    try:
        main()
    finally:
        for img in Path('.').glob('face_images/*.jpg'):
            if img.name[0:len(img.name) - 4].isdigit():
                os.remove('face_images/' + img.name)

