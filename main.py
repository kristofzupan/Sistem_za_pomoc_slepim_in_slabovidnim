import math

import cv2
import face_recognition
import time
from pathlib import Path
from mtcnn import MTCNN


def faceRecognition():
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

configGender = "gad/gender_deploy.prototxt"
modelGender = "gad/gender_net.caffemodel"
genderNeuralNet = cv2.dnn.readNet(modelGender, configGender)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-22)', '(23-34)', '(35-45)', '(46-59)', '(60-100)']
genderList = ['Male', 'Female']

person_image = face_recognition.load_image_file("face_images/Kristof.jpg")
person_encoding = face_recognition.face_encodings(person_image)[0]

known_faces = []
known_faces_names = []

for p in Path('.').glob('face_images/*.jpg'):
    person_image = face_recognition.load_image_file("face_images/" + p.name)
    person_encoding = face_recognition.face_encodings(person_image)[0]
    known_faces.append(person_encoding)
    known_faces_names.append(p.name[0:len(p.name) - 4])


def ageGender(face, faceBox):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNeuralNet.setInput(blob)
    genderPreds = genderNeuralNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNeuralNet.setInput(blob)
    agePreds = ageNeuralNet.forward()
    age = ageList[agePreds[0].argmax()]

    cv2.putText(result, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    age = age[1:len(age)-1]
    age = age.split("-")
    if gender == "Male":
        return f"Moški med {age[0]} in {age[1]} let"
    else:
        return f"Ženska med {age[0]} in {age[1]} let"


def faceDetect(face, faceBox):
    face = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)

    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_locations = [
        (0 + 2, rgb_face.shape[1] - 2, rgb_face.shape[0] - 2, 0 + 2)
    ]
    encoding_test_faces = face_recognition.face_encodings(rgb_face, face_locations)

    if (len(encoding_test_faces) > 0):
        encoding_test_face = encoding_test_faces[0]

        match = face_recognition.compare_faces(known_faces, encoding_test_face)

        for i in range(0, len(match)):
            if match[i]:
                cv2.putText(result, known_faces_names[i], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, cv2.LINE_AA)
                return True, known_faces_names[i]
        else:
            return False, ageGender(face, faceBox)
    else:
        return False, ageGender(face, faceBox)


def auditoryPrompt(face_positions, frame_shape):
    for face_id, face_box in face_positions.items():
        if time.time() - face_box[3] > 2:
            if (face_box[0][0] / frame_shape[1] + face_box[0][2] / frame_shape[1]) / 2 < 0.35:
                print(face_box[2] + " je na levi.")
            elif (face_box[0][0] / frame_shape[1] + face_box[0][2] / frame_shape[1]) / 2 > 0.65:
                print(face_box[2] + " je na desni.")
            else:
                print(face_box[2] + " je naravnost pred vami.")

            face_positions[face_id][3] = float('inf')
    return face_positions


if __name__ == '__main__':
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

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

    while rval:

        current_face_positions = {}

        result, faces = detectFace(faceNeuralNet, frame)

        #if not faces:
            #print("No face detected")

        for faceBox in faces:
            face = frame[max(0, faceBox[1] - 10): min(faceBox[3] + 10, frame.shape[0] - 1), max(0, faceBox[0] - 10): min(faceBox[2] + 10, frame.shape[1] - 1)]

            #print("Prev: ", prev_face_positions)
            if len(prev_face_positions) > 0:
                found_matching_face = False
                for prev_id, prev_face_box in prev_face_positions.items():
                    overlap_area = calculate_overlap_area(faceBox, prev_face_box[0])

                    if overlap_area > 0.5:
                        if len(prev_face_box[1]) < 2:
                            isFaceDetected, faceText = faceDetect(face, faceBox)
                            current_face_positions[prev_id] = [faceBox, [prev_face_box[1][0], isFaceDetected], faceText, prev_face_box[3]]
                        else:
                            if not any(prev_face_box[1]):
                                current_face_positions[prev_id] = [faceBox, prev_face_box[1], ageGender(face, faceBox), prev_face_box[3]]
                            elif any(prev_face_box[1]):
                                cv2.putText(result, prev_face_box[2], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                                current_face_positions[prev_id] = [faceBox, prev_face_box[1], prev_face_box[2], prev_face_box[3]]
                            else:
                                isFaceDetected, faceText = faceDetect(face, faceBox)
                                current_face_positions[prev_id] = [faceBox, [prev_face_box[1][1], isFaceDetected], faceText, prev_face_box[3]]
                                cv2.putText(result, prev_face_box[2], (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                        found_matching_face = True
                        break

                if not found_matching_face:
                    face_id_counter += 1
                    isFaceDetected, faceText = faceDetect(face, faceBox)
                    current_face_positions[face_id_counter] = [faceBox, [isFaceDetected], faceText, time.time()]

            else:
                for i in range(len(faces)):
                    face_id_counter += 1
                    isFaceDetected, faceText = faceDetect(face, faceBox)
                    current_face_positions[face_id_counter] = [faces[i], [isFaceDetected], faceText, time.time()]

        prev_face_positions = auditoryPrompt(current_face_positions, frame.shape)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(round(fps, 2))
        cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("preview", result)

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()
