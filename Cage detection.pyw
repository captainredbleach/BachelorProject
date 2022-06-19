import math
import multiprocessing
from threading import Thread
import numpy as np
import cv2
import os
import time
from multiprocessing.pool import ThreadPool
from collections import deque


def draw_flow(img, prevgray, gray, step=36):
    
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 10, 25, 2, 15, 1.7, 0)
    
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)    
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    
    directionX = [fx[x] for x in range(len(lines)) if np.abs(fx[x]) > 15]
    directionY = [fy[y] for y in range(len(lines)) if np.abs(fy[y]) > 15]
    
    if len(directionX)>=1:
        dx = np.sum(directionX)
        dy = np.sum(directionY)
        tan = np.abs(dy)/np.abs(dx) 
        if (np.abs(dx) > 300) or (np.abs(dy) > 300):
            return tan

    return None
    

def filtering(frame):
    frame = cv2.bilateralFilter(frame, 5, 75, 75)
    
    frame = cv2.medianBlur(frame, 3)
    
    frame = cv2.GaussianBlur(frame,(3,3), cv2.BORDER_DEFAULT)
    
    return frame

def findbox(hsv_Box, kernel, bgr, x,y,w,h):
    Lower_box = np.array([10, 70, 60])
    Upper_box = np.array([25, 85, 255])
    mask_box = cv2.inRange(hsv_Box, Lower_box, Upper_box)
    mask_box = cv2.morphologyEx(mask_box, cv2.MORPH_DILATE, kernel, iterations=4)
    
    result_box = cv2.bitwise_and(bgr[y:y + h, x:x + w], bgr[y:y + h, x:x + w], mask = mask_box)
    result_box = cv2.cvtColor(result_box, cv2.COLOR_BGR2GRAY)
    
    thresh_box = cv2.threshold(result_box, 1, 255, cv2.THRESH_BINARY)[1]
    closening = cv2.morphologyEx(thresh_box, cv2.MORPH_CLOSE, kernel, iterations=6)
    opening = cv2.morphologyEx(closening, cv2.MORPH_OPEN, kernel, iterations=8)
    closening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=8)
    
    Box_Contours = cv2.findContours(closening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cX = 0
    if Box_Contours[1] is not None:
        #print("Cage with box")
        Box_Contours = Box_Contours[0] if len(Box_Contours) == 2 else Box_Contours[1]
        bc = max(Box_Contours, key = cv2.contourArea)
        bx,by,bw,bh = cv2.boundingRect(bc)
        if (bw >= 75) and (bh >= 75):
            cv2.rectangle(bgr[y:y + h, x:x + w], (bx, by), (bx + bw, by + bh), (0,0,255), 2)
            M = cv2.moments(bc)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(bgr[y:y + h, x:x + w], (cX, cY), 7, (255, 255, 255), -1)
        
    if cX == 0: return None, None, False
    
    return (x + cX), None, True


def process(backSub, cframe, pframe):
    kernel = np.ones((5,5), np.uint8)
    bgr = cframe.copy()
    Foreground_Mask = backSub.apply(cframe, 0, learningRate = 0.9)
    Foreground_Mask = cv2.morphologyEx(Foreground_Mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    Foreground_Mask = cv2.morphologyEx(Foreground_Mask, cv2.MORPH_OPEN, kernel, iterations=15)
    Foreground = cv2.bitwise_and(cframe, cframe, mask = Foreground_Mask)
    cv2.normalize(Foreground, Foreground, 0, 255, cv2.NORM_MINMAX)
    ForegroundGrey = cv2.cvtColor(Foreground, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(9,9))
    diff = clahe.apply(ForegroundGrey)
    
    
    edge = cv2.Canny(diff, 150, 160, apertureSize=3, L2gradient = True)
    edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=4)
    
    Foreground = filtering(Foreground)
    hsv = cv2.cvtColor(Foreground, cv2.COLOR_BGR2HSV_FULL)
    hsv = filtering(hsv)
    
    Lower_Grey = np.array([0, 10, 30])
    Upper_Grey = np.array([360, 40, 200])
    # preparing the mask to overlay
    GreyMask = cv2.inRange(hsv, Lower_Grey, Upper_Grey)
    GreyMask = cv2.morphologyEx(GreyMask, cv2.MORPH_CLOSE, kernel, iterations=4)
    GreyMask = cv2.morphologyEx(GreyMask, cv2.MORPH_OPEN, kernel, iterations=4)
    
    
    Combined_Mask = cv2.bitwise_and(GreyMask, edge)
    Combined_Mask = cv2.morphologyEx(Combined_Mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    Combined_Mask = cv2.morphologyEx(Combined_Mask, cv2.MORPH_ERODE, kernel, iterations=4)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    frame = filtering(diff)
    Cage_Isolated = cv2.bitwise_and(frame, frame, mask = Combined_Mask)
    
    
    thresha = cv2.threshold(Cage_Isolated, 140, 255, cv2.THRESH_TOZERO_INV)[1]
    thresh = cv2.threshold(thresha, 20, 255, cv2.THRESH_BINARY)[1]
    
    
    closening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=60)
    opening = cv2.morphologyEx(closening, cv2.MORPH_OPEN, kernel, iterations=67)

    
    Contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res = None
    debug = None
    Tan = None
    CframeGrey = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
    PframeGrey = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
    if Contours[1] is not None:
        
        Contours = Contours[0] if len(Contours) == 2 else Contours[1]
        c = max(Contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        if (w < 1100) and (h < 600) and (w >= 300) and (h > 200):
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0,0,255), 2)
            hsv_Box = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV)
            hsv_Box = filtering(hsv_Box)
            Box_ROI = hsv_Box[y:y + h, x:x + w]
            
            res, debug, findflow = findbox(Box_ROI, kernel, bgr, x,y,w,h)
            if findflow:
                Tan = draw_flow(bgr[y:y + h, x:x + w], PframeGrey[y:y + h, x:x + w], CframeGrey[y:y + h, x:x + w])

    return bgr, res, Tan, debug

def frameIO():
    thread_num = multiprocessing.cpu_count()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    video_name = "15FPS_720PL.mp4"
    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name)) #cv2.VideoCapture(0)
    fps = np.rint(cap.get(cv2.CAP_PROP_FPS))
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    
    pts = []
    angleTan =[]
    dirX = 0
    right = 0
    left = 0
    watchdog = 0
    dX = 0
    prev_frame = None
    
    while cap.isOpened():
        
        while len(pending_task) > 0 and pending_task[0].ready():
            #print(len(pending_task))
            res, diX, Tan, debug = pending_task.popleft().get()
            
                
            if diX is not None and (99 < diX and diX < 1179):
                pts.append(diX)
                watchdog = 0
                #print(diX)
                
            if Tan is not None:
                angleTan.append(Tan)
                
            cv2.imshow('result', res)
            
            if debug is not None:
                cv2.imshow('debug', debug)
                
            if len(pts) > 0:
                if diX is not None and (99 > diX or diX > 1179):
                    watchdog += 2
                else: watchdog += 1

                if watchdog > 10:

                    for i in range((len(pts)//2)):
                        dX = pts[i+len(pts)//2] - pts[i]
                        if dX >= 30:
                            right += 1
                        elif dX <= -30:
                            left += 1
                    
                    print('\033[92m', end ="") #color green
                    
                    if (0.5 <= right / (left + 1)) and (0.5 <= left / (right + 1)):
                        print('\033[93m', end ="") #Color orange
                        print("WARNING: Result unreliable")
                        
                    if not((left >= 10) ^ (right >= 10)):
                        print('\033[93m', end ="") #Color orange
                        print("WARNING: Not enough votes")
                        
                    if len(pts) < 10:
                        print('\033[91m', end ="") #Color red
                        print("ERROR: Sample size is too small")
                    elif len(pts) < 20:
                        print('\033[93m', end ="") #Color orange
                        print("WARNING: Sample size is small")

                    if len(angleTan) < 1:
                        print('\033[91m', end ="") #Color red
                        print("ERROR: Angle can not be determined")
                    elif np.average(angleTan)>=1:
                        print('\033[91m', end ="") #Color red
                        print("ERROR: Movement is vertical")

                    print(left,"|",right)
                    dirX = "Moving left" if left > right else "Moving right" if left < right else "Movement not determined"
                    print(dirX+'\033[0m') #Color reset
                    print()
                    dX = 0
                    right = 0
                    left = 0
                    pts = []
                    angleTan =[]

        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            if not ret:
                break
            
            cur_frame = cv2.resize(cur_frame, (1280, 720))
            if prev_frame is not None:
                task = pool.apply_async(process, (backSub, cur_frame, prev_frame))
                pending_task.append(task)
        
        prev_frame = cur_frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                #continue
            
        time.sleep(1 / fps)
        
    cv2.destroyAllWindows()
    cap.release()

def main():
    T1 = Thread(target=frameIO) 
    T1.start()
    T1.join()
    #RFID her

if __name__ == '__main__':
    main()