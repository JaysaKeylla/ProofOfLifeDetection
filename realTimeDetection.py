import torch
import cv2
from collections import defaultdict
from PIL import Image
from torch.autograd import Variable
import os
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import imutils
import dlib
from imutils import face_utils
rom scipy.spatial import distance as dist
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# Test data preprocessing
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
test_data = datasets.ImageFolder('dataset/test/', transform=test_transforms)
print(test_data.class_to_idx)
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 5


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# predict the image


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def isBlinking(history, maxFrames):

    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False


# Load face detector
proto_path = os.path.sep.join(["face_detector/", "deploy.prototxt"])
model_path = os.path.sep.join(
    ["face_detector/", "res10_300x300_ssd_iter_140000.caffemodel"])
face_detector = cv.dnn.readNetFromCaffe(proto_path, model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('liveliness_detection_2.pt')
model.eval()
COUNTER = 0
TOTAL = 0

cap = cv.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# cap = cv.VideoCapture(0)
# eyes_detected = defaultdict(str)
# loop over the frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    blob = cv.dnn.blobFromImage(
        cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    face_detector.setInput(blob)
    detections = face_detector.forward()
    # print(len(detections))

    # ensure at least one face was found
    for i in range(0, detections.shape[2]):
        # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.95:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY-10:endY+10, startX-10:endX+10]

            face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            output = predict_image(face_pil)

            output = predict_image(face_pil)
            if output == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv.convexHull(leftEye)
                    rightEyeHull = cv.convexHull(rightEye)
                    #cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    #cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
#                     eye_status = '1'

#                     if ear < EYE_AR_THRESH:
#                         eye_status = '0'
#                     eyes_detected[i] += eye_status

#                     if isBlinking(eyes_detected[i],3):
#                         cv.putText(frame, 'Blink', (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                         cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                    if ear < EYE_AR_THRESH:
                        COUNTER += 1

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL = True

                        else:
                            TOTAL = False

                        # reset the eye frame counter
                            COUNTER = 0

                    if TOTAL:
                        cv.putText(frame, 'Real', (startX, startY - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv.rectangle(frame, (startX, startY),
                                     (endX, endY), (0, 255, 0), 2)

                    else:
                        cv.putText(frame, 'Fake', (startX, startY - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv.rectangle(frame, (startX, startY),
                                     (endX, endY), (0, 0, 255), 2)

            else:
                cv.putText(frame, 'Fake', (startX, startY - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.rectangle(frame, (startX, startY),
                             (endX, endY), (0, 0, 255), 2)

 # show the output frame and wait for a key press
    cv.imshow("Frame", frame)
    #frame = cv.flip(frame,0)
    out.write(frame)

    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        COUNTER = []
        TOTAL = []
        break

cap.release()
out.release()

cv.destroyAllWindows()
type(frame)
detections[(0)]

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while(True):
    ret, frame = cap.read()

    if ret == True:

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:

        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
type(frame)
