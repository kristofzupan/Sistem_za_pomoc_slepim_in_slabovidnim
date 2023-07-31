import threading

import cv2
import face_recognition
import time
from pathlib import Path
from pytesseract import *
from difflib import SequenceMatcher
import os.path
from itertools import islice


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

for p in Path('.').glob('face_images/*.jpg'):
    person_image = face_recognition.load_image_file("face_images/" + p.name)
    person_encoding = face_recognition.face_encodings(person_image)
    if len(person_encoding) > 0:
        known_faces.append(person_encoding[0])
        known_faces_names.append(p.name[0:len(p.name) - 4])


def ageGender(frame, face, faceBox, genderAgeData):
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
    face_smaller = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)

    rgb_face = cv2.cvtColor(face_smaller, cv2.COLOR_BGR2RGB)

    face_locations = [
        (0 + 2, rgb_face.shape[1] - 2, rgb_face.shape[0] - 2, 0 + 2)
    ]
    encoding_test_faces = face_recognition.face_encodings(rgb_face, face_locations)

    if len(encoding_test_faces) > 0:
        encoding_test_face = encoding_test_faces[0]
        new_list = []
        for element in known_faces:
            if element is not []:
                new_list.append(element)

        match = face_recognition.compare_faces(new_list, encoding_test_face)

        for i in range(0, len(match)):
            if match[i]:
                cv2.putText(frame, known_faces_names[i], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                return frame, True, known_faces_names[i], genderAgeData

    frame, textAgeGender, gender_age_data = ageGender(frame, face, faceBox, genderAgeData)
    return frame, False, textAgeGender, gender_age_data


def addNewEncoding(face_id):
    known_faces.append([])
    index = len(known_faces) - 1
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    known_faces_names.append(str(face_id))
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def changeEncoding(face_id):
    new_person_image = face_recognition.load_image_file("face_images/" + str(face_id) + ".jpg")
    new_person_encoding = face_recognition.face_encodings(new_person_image)
    index = known_faces_names.index(str(face_id))
    if len(new_person_encoding) > 0:
        known_faces[index] = new_person_encoding[0]


def saveFaceImage(frame, face_positions, time_at_end, prev_face_positions):
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
            if prev_face_positions == {} or face_data[4][1] < 0.9 or face_id > len(prev_face_positions)-1:
                continue
            print("PREV FACE[face_id]:", prev_face_positions[face_id])
            if prev_face_positions[face_id][4][1] != face_data[4][1]:
                cv2.imwrite('face_images/'+str(face_id)+'.jpg', face)
                thread1 = threading.Thread(target=addNewEncoding, args=(face_id,))
                thread2 = threading.Thread(target=changeEncoding, args=(face_id,))
                if str(face_id) not in known_faces_names:
                    print("NEW FACE ADDED")
                    thread1.start()
                else:
                    print("CHANGED FACE ENCOD")
                    thread2.start()

    return face_positions


def auditoryPrompt(face_positions, frame_shape, time_at_end):
    for face_id, face_box in face_positions.items():
        if time_at_end - face_box[3] > 3:
            if (face_box[0][0] / frame_shape[1] + face_box[0][2] / frame_shape[1]) / 2 < 0.35:
                print(face_box[2] + " je na levi.")
            elif (face_box[0][0] / frame_shape[1] + face_box[0][2] / frame_shape[1]) / 2 > 0.65:
                print(face_box[2] + " je na desni.")
            else:
                print(face_box[2] + " je naravnost pred vami.")

            face_positions[face_id][3] = face_box[3] + 30
    return face_positions


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def faceMain(frame_org, prev_face_positions, face_id_counter):
    frame = cv2.cvtColor(frame_org.copy(), cv2.COLOR_BGR2RGB)
    current_face_positions = {} #DICT: faceBox, []array(length: 2) was face recognized in prev frames, text for display, time, array for max conf ageGender (gender index, gender max conf, age index, age max conf)

    result, faces = detectFace(faceNeuralNet, frame.copy())

    # if not faces:
    # print("No face detected")
    print("Prev: ", prev_face_positions)

    for faceBox in faces:
        face = frame[max(0, faceBox[1] - 10): min(faceBox[3] + 10, frame.shape[0] - 1), max(0, faceBox[0] - 10): min(faceBox[2] + 10, frame.shape[1] - 1)]
        if face.shape[0] < 1 or face.shape[1] < 1:
            continue

        #print(face.shape)
        if len(prev_face_positions) > 0:
            found_matching_face = False
            for prev_id, prev_face_box in prev_face_positions.items():
                overlap_area = calculate_overlap_area(faceBox, prev_face_box[0])


                # cv2.imshow("face", face)
                if overlap_area > 0.5:
                    if len(prev_face_box[1]) < 2:
                        frame, isFaceDetected, faceText, gender_age_data = faceRecognition(frame, face, faceBox, prev_face_box[4])
                        if faceText.isdigit() and not prev_face_box[1][0]:
                            current_face_positions[int(faceText)] = [faceBox, [prev_face_box[1][0], isFaceDetected], faceText, prev_face_box[3], gender_age_data]
                        else:
                            current_face_positions[prev_id] = [faceBox, [prev_face_box[1][0], isFaceDetected], faceText, prev_face_box[3], gender_age_data]
                    else:
                        if not any(prev_face_box[1]):
                            frame, textAgeGender, gender_age_data = ageGender(frame, face, faceBox, prev_face_box[4])
                            current_face_positions[prev_id] = [faceBox, prev_face_box[1], textAgeGender, prev_face_box[3], gender_age_data]
                        elif all(prev_face_box[1]):
                            cv2.putText(frame, prev_face_box[2], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                            current_face_positions[prev_id] = [faceBox, prev_face_box[1], prev_face_box[2], prev_face_box[3], prev_face_box[4]]
                        else:
                            frame, isFaceDetected, faceText, gender_age_data = faceRecognition(frame, face, faceBox, prev_face_box[4])
                            if faceText.isdigit() and not prev_face_box[1][0]:
                                current_face_positions[int(faceText)] = [faceBox, [prev_face_box[1][1], isFaceDetected], faceText, prev_face_box[3], gender_age_data]
                            else:
                                current_face_positions[prev_id] = [faceBox, [prev_face_box[1][1], isFaceDetected], faceText, prev_face_box[3], gender_age_data]
                    found_matching_face = True
                    break

            if not found_matching_face:
                frame, isFaceDetected, faceText, ageGenderData = faceRecognition(frame, face, faceBox, [0, 0, 0, 0])

                if faceText.isdigit():
                    current_face_positions[int(faceText)] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData]
                else:
                    face_id_counter += 1
                    current_face_positions[face_id_counter] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData]

        else:
            frame, isFaceDetected, faceText, ageGenderData = faceRecognition(frame, face, faceBox, [0, 0, 0, 0])
            if faceText.isdigit():
                current_face_positions[int(faceText)] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData]
            else:
                face_id_counter += 1
                current_face_positions[face_id_counter] = [faceBox, [isFaceDetected], faceText, time.time(), ageGenderData]

    time_at_end = time.time()
    current_face_positions = saveFaceImage(frame_org, current_face_positions, time_at_end, prev_face_positions)

    print("current:", current_face_positions)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), auditoryPrompt(current_face_positions, frame.shape, time_at_end), face_id_counter


def textMain(frame, read_queue):
    print("================================")
    pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.GaussianBlur(frame, (5, 5), 1)
    #frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    text_res = pytesseract.image_to_data(frame, lang='slv', output_type=Output.DICT)

    for i in range(0, len(text_res["text"])):
        x = text_res["left"][i]
        y = text_res["top"][i]
        w = text_res["width"][i]
        h = text_res["height"][i]

        text = text_res["text"][i]
        conf = int(text_res["conf"][i])

        if conf > 0.75:
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))

            # We then strip out non-ASCII text so we can
            # draw the text on the image We will be using
            # OpenCV, then draw a bounding box around the
            # text along with the text itself
            text = "".join(text).strip()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 1)

    cv2.imshow("preview", frame)
    return read_queue


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

        frame, prev_face_positions, face_id_counter = faceMain(frame, prev_face_positions, face_id_counter)
        #read_queue = textMain(frame, read_queue)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(round(fps, 2))
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("preview", frame)

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == '__main__':
    main()
