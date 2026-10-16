import ntpath
import random

from PIL import Image
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Initialize globally (do this once at startup)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for first GPU

detInpWidth = 416;
detInpHeight = 416;
detModelConfiguration = "./rot_det_model/rd.cfg"
detModelWeight = "./rot_det_model/rd.bin"
detNet = cv2.dnn.readNet(detModelWeight, detModelConfiguration)
det_model = cv2.dnn_DetectionModel(detNet)
det_model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True, crop=False)

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def getRotatedAngle(img):
    angle = [0] * 4

    det_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det_img = cv2.cvtColor(det_img, cv2.COLOR_GRAY2BGR)
    classes, scores, boxes = det_model.detect(det_img, 0.5, 0.5)
    if len(classes) != 0:
        if classes[0] == 0:
            angle[0] += 1
        elif classes[0] == 1:
            temp_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
            classes, scores, boxes = det_model.detect(temp_img, 0.5, 0.5)
            if len(classes) != 0 and classes[0] == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                angle[3] += 1
            elif len(classes) != 0 and classes[0] == 0:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                angle[1] += 1
        elif classes[0] == 2:
            temp_img = cv2.rotate(img, cv2.ROTATE_180)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
            classes, scores, boxes = det_model.detect(temp_img, 0.5, 0.5)
            if len(classes) != 0 and classes[0] == 0:
                img = cv2.rotate(img, cv2.ROTATE_180)
                angle[2] += 1
        elif classes[0] == 3:
            temp_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
            classes, scores, boxes = det_model.detect(temp_img, 0.5, 0.5)
            if len(classes) != 0 and classes[0] == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                angle[1] += 1
            elif len(classes) != 0 and classes[0] == 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                angle[3] += 1

    rot_angle = np.argmax(angle)

    return rot_angle

input_path = r"/datasets/newout/nizar_tests/fake/pad"
full_file_paths = get_filepaths(input_path)
for src_path in full_file_paths:
    head, fname = ntpath.split(src_path)
    if os.path.isdir(src_path) == False and (fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png')):
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        angle = getRotatedAngle(img)
        # angle = random.choice([0, 1, 2, 3])
        if angle == 1:
            temp_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 2:
            temp_img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 3:
            temp_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            temp_img = img

        fn = fname.split(".")[0]
        # path = os.path.join(head, fn+"_fix.jpg")
        path = os.path.join(head, fname)
        cv2.imwrite(path, temp_img)
        print(path)
