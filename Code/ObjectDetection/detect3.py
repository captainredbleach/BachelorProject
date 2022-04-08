import numpy as np
import cv2
import os
import math
import threading

def process(rgb, hsv):
    
     
    # Threshold of brown in HSV space
    lower_brown = np.array([5, 80, 100])
    upper_brown = np.array([15, 130, 140])
 
    # preparing the mask to overlay
    mask_b = cv2.inRange(hsv, lower_brown, upper_brown)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-brown regions
    result_b = cv2.bitwise_and(frame, frame, mask = mask_b)
    
    lower_g = np.array([30, 40, 80])
    upper_g = np.array([35, 50, 150])
 
    # preparing the mask to overlay
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-brown regions
    result_g = cv2.bitwise_and(frame, frame, mask = mask_g)
 
    
    imgGrey = cv2.cvtColor(result_g, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgGrey, 30, 255, cv2.THRESH_BINARY)
    img_dilation = cv2.dilate(thresh, kernel, iterations=10)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=12)
    
    img_erosion2 = cv2.erode(img_erosion, kernel, iterations=6)
    img_dilation2 = cv2.dilate(img_erosion, kernel, iterations=30)
    
    finalE = cv2.erode(img_dilation2, kernel, iterations=2)
    
    img_final = abs(img_dilation2) - abs(finalE)
    
    
    
    #img_dilation = cv2.dilate(img_erosion, kernel, iterations=15)
    #cv2.imshow("ok", imgGrey)
    #Hough Line Transform 

    linesP = cv2.HoughLinesP(img_final, 1, np.pi / 180, 50, None, 50, 10)

    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(rgb, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            
    return rgb
    
    
     
    
 


path = os.path.dirname(os.path.realpath(__file__))

cap = cv2.VideoCapture(path + "\\15FPS_720P.mp4")
kernel = np.ones((5,5), np.uint8)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

#result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)




while(1):
    _, frame = cap.read()
    
    if _ == False:
        break
    
    rgb = frame
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    t1 = threading.Thread(target=process, args=(rgb, hsv))
    t1.start()
    t1.join()
    
    #result.write(rgb)
    cv2.imshow('result', rgb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
#result.release()