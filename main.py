import cv2
import keras 
import numpy as np
import sys
from copy import deepcopy
from matplotlib import pyplot as plt

# if len(sys.argv) != 2:
#     print("Invalid arguments")
#     exit(0)

# tp_idx = sys.argv[1]
# img = cv2.imread('tests/{}.png'.format(tp_idx))
img = cv2.imread('tests/1.png')


#################################################################################
# Image processing

# Load the model
model = keras.models.load_model('model.h5') 

solution = img.copy()
solution_gray = cv2.cvtColor(solution, cv2.COLOR_BGR2GRAY)

# Removing noise
dst = cv2.fastNlMeansDenoising(solution,None,50,7,21)

# src_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(dst, (3, 3))

# Thresholding for easier edge detection
ret, src_gray = cv2.threshold(src_gray, 240, 255, cv2.THRESH_BINARY)


def find_if_close(cnt1, cnt2):
    """
    Test whether or not two contours are close to each other
    :param cnt1: First contour
    :param cnt2: Second contour
    :return: True / False
    """
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 5:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def thresh_callback(val):
    """
    Does edge detection, after which a contour is created for each object. Contours that are close to each other are
    merged into one contour by calculating the convex hull. Contours that are completely inside another contour are
    ignored. Contours are used for calculating all of the bounding boxes which are then drawn on the picture.
    :param val: Parameter for Canny algorithm, edges with intensity under val are ignored, edges with intensity over
    2 * val are sure to be edges, edges with intensity between are decided based on connectivity
    :return: All bounding boxes, indices of bounding boxes to be ignored as they are completely inside other bounding
    boxes
    """
    threshold = val

    # Edge detection and finding contours
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    # Nearby contours are put in groups
    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist is True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    # Each group of nearby contours is merged into one by calculating convex hull
    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    contours = unified

    # Bounding box for each contour
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    ignore = []

    # Adding bounding boxes that do not represent real objects to ignored list
    for i in range(len(boundRect)):
        for j in range(len(boundRect)):
            if i == j:
                continue
            if boundRect[j][0] >= boundRect[i][0] and boundRect[j][1] >= boundRect[i][1]:
                if boundRect[j][0] + boundRect[j][2] <= boundRect[i][0] + boundRect[i][2] and \
                        boundRect[j][1] + boundRect[j][3] <= boundRect[i][1] + boundRect[i][3]:
                    ignore.append(j)

    # Drawing a rectangle on picture for each bounding box
    for i in range(len(contours)):
        color = (255, 0, 0)
        # latitude = boundRect[i][2] * boundRect[i][3]
        # if latitude < 100:
        #     continue
        if i in ignore:
            continue
        cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    return boundRect, ignore


thresh = 10
boundingBoxes, ignore = thresh_callback(thresh)

labelNames = ["tshirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# print(boundingBoxes)

for i in range(len(boundingBoxes)):
    imageHeight = boundingBoxes[i][3]
    imageWidth = boundingBoxes[i][2]
    # if imageHeight <= 10 or imageWidth <= 10:
    #     continue

    # Skip bounding boxes inside of other bounding boxes
    if i in ignore:
        continue

    # Make the bounding box a perfect square so that scaling down to 28x28 results in pictures with higher quality
    if imageWidth > imageHeight:
        newImage = np.full((imageWidth, imageWidth), 255, np.uint8)
        razlika = (imageWidth - imageHeight) // 2
        newImage[razlika : razlika + boundingBoxes[i][3], :] = solution_gray[boundingBoxes[i][1]: boundingBoxes[i][1] + boundingBoxes[i][3],
                     boundingBoxes[i][0]: boundingBoxes[i][0] + boundingBoxes[i][2]]
        res = cv2.resize(newImage, None, fx=28 / imageWidth, fy=28 / imageWidth, interpolation=cv2.INTER_AREA)
    else:
        newImage = np.full((imageHeight, imageHeight), 255, np.uint8)
        razlika = (imageHeight - imageWidth) // 2
        newImage[:, razlika: razlika + boundingBoxes[i][2]] = solution_gray[boundingBoxes[i][1]: boundingBoxes[i][1] + boundingBoxes[i][3],
                     boundingBoxes[i][0]: boundingBoxes[i][0] + boundingBoxes[i][2]]
        res = cv2.resize(newImage, None, fx=28 / imageHeight, fy=28 / imageHeight, interpolation=cv2.INTER_AREA)

    # Inverting bits needed because model i trained on black background
    res = cv2.bitwise_not(res)

    # Normalization
    res = res.astype('float32')
    res = res / 255
    res = cv2.resize(res, (28, 28))

    # Reshaping to suit model
    res = res.reshape(1, 28, 28, 1)

    # Prediction and drawing predicted class
    probabilities = model.predict(res) 
    # probabilities = [0]
    prediction = probabilities.argmax()
    label = labelNames[prediction]
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw = cv2.putText(img, label, (boundingBoxes[i][0], boundingBoxes[i][1]), font, 0.5, (0, 0, 255), 1,
                       cv2.LINE_AA)

solution = draw.copy()

cv2.imshow('Solution: ', solution)

cv2.waitKey(0)

#################################################################################

# Save solution to output file
# cv2.imwrite("tests/out_{}.png".format(tp_idx), solution)
