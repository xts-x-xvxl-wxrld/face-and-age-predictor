import cv2
import time
import argparse


def face_box(net, frame, conf_thresh=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (224, 224), [104, 117, 123], True, True)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            x1 = int(detections[0, 0, i, 3] * frame_height)
            y1 = int(detections[0, 0, i, 4] * frame_width)
            x2 = int(detections[0, 0, i, 5] * frame_height)
            y2 = int(detections[0, 0, i, 6] * frame_width)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 255), int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, bboxes


parser = argparse.ArgumentParser(description=
                                 'Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="cpu", help="Device to inference on")

args = parser.parse_args()


args = parser.parse_args()

faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'

genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

if args.device == "cpu":
    ageNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif args.device == "gpu":
    ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

cap = cv2.VideoCapture(args.input if args.input else 0)
padding = 20

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    frameFace, bboxes = face_box(faceNet, frame)
    if not bboxes:
        print('no face detected')
        continue

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding): min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        print("Gender Output : {}".format(genderPred))
        print('Gender: {}, conf={:3f}'.format(gender, genderPred[0].max()))

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        print('Age output {}'.format(agePred))
        print('Age : {}, conf={:.3f}'.format(age, agePred[0].max()))

         