
# Importing library
import numpy as np
import cv2 as cv
import os

# Load face detector
proto_path = os.path.sep.join(["face_detector/", "deploy.prototxt"])
model_path = os.path.sep.join(["face_detector/","res10_300x300_ssd_iter_140000.caffemodel"])
face_detector = cv.dnn.readNetFromCaffe(proto_path, model_path)

# list of different types of videos
type_of_videos = ["01.G/","02.Mc/","03.Mf/","04.Mu/","05.Pq/","06.Ps/","07.Vl/","08.Vm/"]
output_destination = ["01.G-pics/","02.Mc-pics/","03.Mf-pics/","04.Mu-pics/","05.Pq-pics/","06.Ps-pics/","07.Vl-pics/","08.Vm-pics/"]

for item in range(len(type_of_videos)):

    # Process the videos into a list called inputs
    inputs = os.listdir(type_of_videos[item])

    saved = 0

    for video in range(len(inputs)):
        cap = cv.VideoCapture(type_of_videos[item]+inputs[video])
        read =0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # counting the numbers of frame
            read += 1
            if type_of_videos[item] == "01.G/":
                if read % 25 != 0:
                    continue
            elif type_of_videos[item] == "03.Mf/":
                if read % 6 != 0:
                    continue
            else:
                if read % 12 != 0:
                    continue

            # grab the frame dimensions and construct a blob from the frame
            (h,w) = frame.shape[:2]
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            #cv.imshow("test Window", frame)
            #cv.waitKey(50)

            detections = face_detector.forward()


            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > 0.6:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = frame[startY-10:endY+10, startX-10:endX+10]

                    # write the frame to disk
                    p = os.path.sep.join([output_destination[item],"{}.png".format(saved)])
                    cv.imwrite(p, face)
                    saved += 1
        cap.release()

        cv.destroyAllWindows()