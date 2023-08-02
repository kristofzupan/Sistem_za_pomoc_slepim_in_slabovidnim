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
import enchant
import Levenshtein

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

#person_image = face_recognition.load_image_file("face_images/Kristof.jpg")
#person_encoding = face_recognition.face_encodings(person_image)[0]

known_faces = []
known_faces_names = []

isTextDetectionOK = True

for p in Path('.').glob('face_images/*.jpg'):
    person_image = face_recognition.load_image_file("face_images/" + p.name)
    person_encoding = face_recognition.face_encodings(person_image)
    if len(person_encoding) > 0:
        known_faces.append(person_encoding[0])
        known_faces_names.append(p.name[0:len(p.name) - 4])


def ageGender(frame, face, faceBox, genderAgeData):
    #print("AGE GENDER")
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
    #print("FACE RECOGNITION")
    face_smaller = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)

    rgb_face = cv2.cvtColor(face_smaller, cv2.COLOR_BGR2RGB)

    face_locations = [
        (0 + 2, rgb_face.shape[1] - 2, rgb_face.shape[0] - 2, 0 + 2)
    ]
    encoding_test_faces = face_recognition.face_encodings(rgb_face, face_locations)

    if len(encoding_test_faces) > 0:
        encoding_test_face = encoding_test_faces[0]

        for i in range (0, len(known_faces)):
            if known_faces[i].size == 0:
                del known_faces[i]
                del known_faces_names[i]
                i -= 1

        match = face_recognition.compare_faces(known_faces, encoding_test_face)

        for i in range(0, len(match)):
            if match[i]:
                cv2.putText(frame, known_faces_names[i], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                return frame, True, known_faces_names[i], genderAgeData

    frame, textAgeGender, gender_age_data = ageGender(frame, face, faceBox, genderAgeData)
    return frame, False, textAgeGender, gender_age_data


def addNewEncoding(face_id):
    #print("ADD ENCODING")
    known_faces.append(np.array([]))
    index = len(known_faces) - 1
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    known_faces_names.append(str(face_id))
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def changeEncoding(face_id):
    #print("CHANGE ENCODING")
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    index = known_faces_names.index(str(face_id))
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def saveFaceImage(frame, face_positions, time_at_end, prev_face_positions):
    #print("SAVE FACE IMAGE")
    for face_id, face_data in face_positions.items():
        face = frame[max(0, face_data[0][1] - 10): min(face_data[0][3] + 10, frame.shape[0] - 1), max(0, face_data[0][0] - 10): min(face_data[0][2] + 10, frame.shape[1] - 1)]
        if face_data[1][len(face_data[1])-1]:
            continue
        if time_at_end - face_data[3] > 3:
            if os.path.exists('face_images/' + str(face_id) + '.jpg'):
                continue
            cv2.imwrite('face_images/'+str(face_id)+'.jpg', face)
            thread1 = threading.Thread(target=addNewEncoding, args=(face_id,))
            thread1.start()
        elif time_at_end - face_data[3] > 0:
            if prev_face_positions == {} or face_data[4][1] < 0.9 or face_id not in prev_face_positions: #or face_id > len(prev_face_positions)-1:
                continue
            #print("PREV FACE[face_id]:", prev_face_positions[face_id])
            if prev_face_positions[face_id][4][1] != face_data[4][1]:
                cv2.imwrite('face_images/'+str(face_id)+'.jpg', face)
                thread1 = threading.Thread(target=addNewEncoding, args=(face_id,))
                thread2 = threading.Thread(target=changeEncoding, args=(face_id,))
                if str(face_id) not in known_faces_names:
                    #print("NEW FACE ADDED")
                    thread1.start()
                else:
                    #print("CHANGED FACE ENCOD")
                    thread2.start()

    return face_positions


def auditoryPrompt(face_positions, frame_shape, time_at_end):
    #print("AUDITORY PROMPT")
    for face_id, face_data in face_positions.items():
        if face_data[0] is None:
            continue
        if time_at_end - face_data[3] > 3:
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
                print(name_prompt + " je na levi.")
            elif (face_data[0][0] / frame_shape[1] + face_data[0][2] / frame_shape[1]) / 2 > 0.65:
                print(name_prompt + " je na desni.")
            else:
                print(name_prompt + " je naravnost pred vami.")

            face_positions[str(face_id)][5] += 1
            face_positions[str(face_id)][3] = face_data[3] + 30
    return face_positions


def faceMain(frame_org, prev_face_positions, face_id_counter):
    #print("FACE MAIN")
    frame = cv2.cvtColor(frame_org.copy(), cv2.COLOR_BGR2RGB)
    current_face_positions = {} #DICT: faceBox, []array(length: 2) was face recognized in prev frames, text for display, time, array for max conf ageGender (gender index, gender max conf, age index, age max conf), readout #times

    result, faces = detectFace(faceNeuralNet, frame.copy())

    # if not faces:
    # print("No face detected")
    #print("Prev: ", prev_face_positions)

    for faceBox in faces:
        face = frame[max(0, faceBox[1] - 10): min(faceBox[3] + 10, frame.shape[0] - 1), max(0, faceBox[0] - 10): min(faceBox[2] + 10, frame.shape[1] - 1)]
        if face.shape[0] < 1 or face.shape[1] < 1:
            continue

        #print(face.shape)
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

    #ADD lost faces back to dict
    for prev_face, prev_face_data in prev_face_positions.items():
        found_match = False
        for current_face, current_face_data in current_face_positions.items():
            if current_face == prev_face:
                found_match = True
        if not found_match:
            current_face_positions[prev_face] = [None, prev_face_data[1], prev_face_data[2], prev_face_data[3], prev_face_data[4], prev_face_data[5]]


    #print("current:", current_face_positions)
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
                #Check 2 more words
                if i+2 < len(previous_frame) and j+2 < len(current_frame):
                    if previous_frame[i+1][0] == current_frame[j+1][0] and previous_frame[i+2][0] == current_frame[j+2][0]:
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

    #print(prev_index_3overlap)
    #print(current_index_3overlap)
    #For every found overlap add to combined, if gaps fill in

    combined = []


    for i in range (0, prev_index_3overlap[0]):
        #print("DODDAJ NA ZAČETKU IZ PREJŠNJEGA")
        combined.append(previous_frame[i])

    for i in range(0, len(prev_index_3overlap)):
        combined.append(current_frame[current_index_3overlap[i]])
        if i+1 < len(prev_index_3overlap):
            if prev_index_3overlap[i+1] - prev_index_3overlap[i] == current_index_3overlap[i+1] - current_index_3overlap[i] and prev_index_3overlap[i+1] - prev_index_3overlap[i] > 1:
                diff = prev_index_3overlap[i+1] - prev_index_3overlap[i]
                for j in range(1, diff):
                    if previous_frame[prev_index_3overlap[i]+j][1] > current_frame[current_index_3overlap[i]+j][1]:
                        combined.append(previous_frame[prev_index_3overlap[i]+j])
                    else:
                        combined.append(current_frame[current_index_3overlap[i]+j])
            elif prev_index_3overlap[i+1] - prev_index_3overlap[i] > current_index_3overlap[i+1] - current_index_3overlap[i] and prev_index_3overlap[i+1] - prev_index_3overlap[i] > 1:
                diff = prev_index_3overlap[i+1] - prev_index_3overlap[i]
                for j in range(1, diff):
                    combined.append(previous_frame[prev_index_3overlap[i]+j])
            elif prev_index_3overlap[i+1] - prev_index_3overlap[i] < current_index_3overlap[i+1] - current_index_3overlap[i] and current_index_3overlap[i+1] - current_index_3overlap[i] > 1:
                diff = current_index_3overlap[i+1] - current_index_3overlap[i]
                for j in range(1, diff):
                    combined.append(current_frame[current_index_3overlap[i]+j])

    #print(len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1], "PRIMERJAMO Z", len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1])
    if len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] == len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        #print("NA KONCU MEŠANO")
        diff = len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1]
        for j in range(1, diff):
            if previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j][1] > current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j][1]:
                combined.append(previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j])
            else:
                combined.append(current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j])
    elif len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] > len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        #print("NA KONCU IZ PREVIOUS")
        diff = len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1]
        for j in range(1, diff):
            combined.append(previous_frame[prev_index_3overlap[len(prev_index_3overlap)-1] + j])
    elif len(previous_frame) - prev_index_3overlap[len(prev_index_3overlap)-1] < len(current_frame) - current_index_3overlap[len(current_index_3overlap)-1]:
        #print("NA KONCU IZ CURRENT")
        diff = len(current_frame) - current_index_3overlap[len(current_index_3overlap) - 1]
        if current_index_3overlap[0] - current_index_3overlap[len(current_index_3overlap)-1] > 0:
            #print("ZAZNANO PONAVLJANJE")
            diff = current_index_3overlap[0] - current_index_3overlap[len(current_index_3overlap)-1]
        for j in range(1, diff):
            combined.append(current_frame[current_index_3overlap[len(current_index_3overlap)-1] + j])

    if current_index_3overlap[0] > 4:
        if current_index_3overlap[0] < current_index_3overlap[len(current_index_3overlap) - 1]:
            #print("DODAJ NA KONEC IZ ZAČETKA ZDAJŠNJEGA")
            for i in range(0, current_index_3overlap[0]):
                combined.append(current_frame[i])

    #print("COMBINED:", combined)
    return combined


def textMain(frame, read_queue):
    print("================================")
    pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.GaussianBlur(frame, (5, 5), 1)
    #frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    text_res = pytesseract.image_to_data(frame, lang='slv', output_type='data.frame')
    text_res = text_res[text_res.conf >= 80]
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
        if slo_dict.check(word_without_symbol):
            filtered_rows.append(row)

    # Create a new DataFrame containing the filtered rows
    text_res = pd.DataFrame(filtered_rows)
    #print(text_res)
    lines = []
    time_lines = time.time()
    for index, row in text_res.iterrows():
        word = row['text']
        conf = row['conf']
        lines.append((word, conf, time_lines))
    #lines = text_res.groupby('block_num')['text'].apply(list)
    #conf_lines = text_res.groupby('block_num')['conf'].mean()
    #for i in range(0, len(lines)):
        #print(lines[i], conf_lines[i])
    #print(lines)
    #for i in range(0, len(text_res["text"])):
    #    x = text_res["left"][i]
    #    y = text_res["top"][i]
    #    w = text_res["width"][i]
    #    h = text_res["height"][i]

    #    text = text_res["text"][i]
    #    conf = int(text_res["conf"][i])

    #    if conf > 0.90:
    #        print("Confidence: {}".format(conf))
    #        print("Text: {}".format(text))

            # We then strip out non-ASCII text so we can
            # draw the text on the image We will be using
            # OpenCV, then draw a bounding box around the
            # text along with the text itself
    #        text = "".join(text).strip()
    #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
    #print("LINES:", lines)
    #print("READ QUEUE", read_queue)
    if lines == []:
        return frame, read_queue

    if read_queue == []:
        read_queue = lines
    else:
        read_queue = combine_lists_with_overlap(read_queue, lines)

    combined_string = ""
    for item in read_queue:
        combined_string += item[0]

    # Using list comprehension
    combined_string = ' '.join([item[0] for item in read_queue])
    print(combined_string)

    return frame, read_queue


def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_frame_time = 0
    new_frame_time = 0

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    face_id_counter = 0
    prev_face_positions = {}

    read_queue = []

    while rval:
        #"NEW FRAME"
        (mean, blurry) = detectBlur(frame)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        textBlur = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
        textBlur = textBlur.format(mean)

        if not blurry:
            frame, prev_face_positions, face_id_counter = faceMain(frame, prev_face_positions, face_id_counter)
            #frame, read_queue = textMain(frame, read_queue)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(round(fps, 2))
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, textBlur, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        cv2.imshow("preview", frame)

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == '__main__':
    try:
        main()
    finally:
        for img in Path('.').glob('face_images/*.jpg'):
            if img.name[0:len(img.name) - 4].isdigit():
                os.remove('face_images/' + img.name)

