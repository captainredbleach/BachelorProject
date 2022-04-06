from cmath import sqrt
import numpy as np
import cv2

cap = cv2.VideoCapture("D:\\P6\\Code\\ObjectDetection\\15FPS_720P.mp4")
kernel = np.ones((1,10), np.uint8)
kerneld = np.ones((50,2), np.uint8)

while(True):
    ret, img = cap.read()
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgE = cv2.Sobel(imgGrey,cv2.CV_8U,1,0, ksize= 1)
    imgd = cv2.Sobel(imgGrey,cv2.CV_8U,0,1,ksize= 1)
    imgp = abs(imgE) + abs(imgd)
    _, thrash = cv2.threshold(imgp, 80, 255, cv2.THRESH_BINARY)
    img_erosion = cv2.erode(thrash, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=10)
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 4:
            x1 ,y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            if aspectRatio >= 0.95 and aspectRatio <=1.05:
                cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Result', thrash)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()