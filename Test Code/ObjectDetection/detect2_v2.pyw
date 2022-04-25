from cmath import sqrt
from pickletools import uint8
import cv2
import numpy as np
import math
import os
import sys

# PATH to video
path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture(path + "\\15FPS_720P.mp4")

kernel = np.ones((6,10), np.uint8)

frame_seq = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Amount of frames: ' + str(frame_seq))

# Fixed frame (used as a workspace)
frame_fix = frame_seq - 2511
cap.set(1, frame_fix)

while(True):
    ret, img = cap.read()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgE = cv2.Sobel(imgGrey,cv2.CV_8U,1,0, ksize=3)
    imgD = cv2.Sobel(imgGrey,cv2.CV_8U,0,1,ksize=3)
    imgp = abs(imgE) + abs(imgD)
    _, thrash = cv2.threshold(imgp, 40, 80, cv2.THRESH_BINARY)

    # Closing
    img_dilation = cv2.dilate(thrash, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=3)

    # Opening
    img_erosion2 = cv2.erode(img_erosion, kernel, iterations=3)
    img_dilation2 = cv2.dilate(img_erosion2, kernel, iterations=3)

    # To get the boundaries only
    img_erosion_final = cv2.erode(img_dilation2, kernel, iterations=1)
    img_final = abs(img_dilation2) - abs(img_erosion_final)

    contours, _ = cv2.findContours(img_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Hough Line Transform 
    cdst = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(img_final, 1, np.pi / 180, 170, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    #dot = cv2.circle(cdst, (585,300), radius=20, color=(255, 0, 0), thickness=-1)

    # for contour in contours:
    #     approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    #     cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1] - 5
    #     if len(approx) == 4:
    #         x1 ,y1, w, h = cv2.boundingRect(approx)
    #         aspectRatio = float(w)/h
    #         if aspectRatio >= 0.95 and aspectRatio <=1.05:
    #             cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    #         else:
    #             cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Result', cdst)
    print('LINES: ' + str(l))

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print('Done.')
cap.release()
cv2.destroyAllWindows()