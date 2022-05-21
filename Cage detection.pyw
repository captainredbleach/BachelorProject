import multiprocessing
from threading import Thread
import numpy as np
import cv2
import os
import time
from multiprocessing.pool import ThreadPool
from collections import deque

def filtering(frame):
    frame = cv2.bilateralFilter(frame, 5, 75, 75)
    
    frame = cv2.medianBlur(frame, 3)
    
    frame = cv2.GaussianBlur(frame,(3,3), cv2.BORDER_DEFAULT)
    
    return frame

def findbox(TempC, kernel, rgb, x,y,w,h):
    Lower_box = np.array([10, 50, 80])
    Upper_box = np.array([25, 110, 255])
    mask_box = cv2.inRange(TempC, Lower_box, Upper_box)
    
    result_box = cv2.bitwise_and(rgb[y:y + h, x:x + w], rgb[y:y + h, x:x + w], mask = mask_box)
    result_box = cv2.cvtColor(result_box, cv2.COLOR_BGR2GRAY)
    
    thresh_box = cv2.threshold(result_box, 1, 255, cv2.THRESH_BINARY)[1]
    closening = cv2.morphologyEx(thresh_box, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closening, cv2.MORPH_OPEN, kernel, iterations=8)
    closening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    Box_Contours = cv2.findContours(closening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cX = 0
    if Box_Contours[1] is not None:
        #print("Cage with box")
        Box_Contours = Box_Contours[0] if len(Box_Contours) == 2 else Box_Contours[1]
        bc = max(Box_Contours, key = cv2.contourArea)
        bx,by,bw,bh = cv2.boundingRect(bc)
        if (bw >= 50) and (bh > 50):
            cv2.rectangle(rgb[y:y + h, x:x + w], (bx, by), (bx + bw, by + bh), (0,0,255), 2)
            M = cv2.moments(bc)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(rgb[y:y + h, x:x + w], (cX, cY), 7, (255, 255, 255), -1)
            
    if cX == 0: return None, None
    
    if 280 < (x + cX) < 1000:
        return (x + cX), None
    
    return None, None


def process(backSub, cframe):
    kernel = np.ones((5,5), np.uint8)
    bgr = cframe.copy()
    Foreground_Mask = backSub.apply(cframe)
    Foreground_Mask = cv2.morphologyEx(Foreground_Mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    Foreground_Mask = cv2.morphologyEx(Foreground_Mask, cv2.MORPH_OPEN, kernel, iterations=15)
    Foreground = cv2.bitwise_and(cframe, cframe, mask = Foreground_Mask)
    cv2.normalize(Foreground, Foreground, 0, 255, cv2.NORM_MINMAX)
    ForegroundGrey = cv2.cvtColor(Foreground, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(9,9))
    diff = clahe.apply(ForegroundGrey)
    
    
    edge = cv2.Canny(diff, 150, 160, apertureSize=3, L2gradient = True)
    edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=1)
    
    Foreground = filtering(Foreground)
    hsv = cv2.cvtColor(Foreground, cv2.COLOR_BGR2HSV)
    hsv = filtering(hsv)
    
    Lower_Grey = np.array([0, 10, 30])
    Upper_Grey = np.array([180, 50, 200])
    # preparing the mask to overlay
    GreyMask = cv2.inRange(hsv, Lower_Grey, Upper_Grey)
    GreyMask = cv2.morphologyEx(GreyMask, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    
    Combined_Mask = cv2.bitwise_and(GreyMask, edge)
    Combined_Mask = cv2.morphologyEx(Combined_Mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    Combined_Mask = cv2.morphologyEx(Combined_Mask, cv2.MORPH_ERODE, kernel, iterations=2)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    frame = filtering(diff)
    Cage_Isolated = cv2.bitwise_and(frame, frame, mask = Combined_Mask)
    
    
    thresha = cv2.threshold(Cage_Isolated, 100, 255, cv2.THRESH_TOZERO_INV)[1]
    thresh = cv2.threshold(thresha, 5, 255, cv2.THRESH_BINARY)[1]
    
    
    closening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=60)
    opening = cv2.morphologyEx(closening, cv2.MORPH_OPEN, kernel, iterations=67)

    res = None
    debug = None
    Contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if Contours[1] is not None:
        
        Contours = Contours[0] if len(Contours) == 2 else Contours[1]
        c = max(Contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        if (w < 1100) and (h < 600) and (w >= 300) and (h > 200):
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0,0,255), 2)
            hsv_Box = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV)
            hsv_Box = filtering(hsv_Box)
            Box_ROI = hsv_Box[y:y + h, x:x + w]
            res, debug = findbox(Box_ROI, kernel, bgr, x,y,w,h)

    return bgr, res, debug

def frameIO():
    thread_num = multiprocessing.cpu_count()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    video_name = "VIDEO_CLIPS//cage1-packages_1.mp4"
    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))
    fps = np.rint(cap.get(cv2.CAP_PROP_FPS)) * thread_num
    backSub = cv2.createBackgroundSubtractorKNN(100, detectShadows=False)
    
    pts = []
    dirX = 0
    right = 0
    left = 0
    watchdog = 0
    dX = 0
    
    while cap.isOpened():
        
        while len(pending_task) > 0 and pending_task[0].ready():
            #print(len(pending_task))
            res, diX, debug = pending_task.popleft().get()
            if diX is not None:
                pts.append(diX)
                watchdog = 0
                #print(diX)
                
            cv2.imshow('result', res)
            
            if debug is not None:
                cv2.imshow('debug', debug)
                
            if len(pts) > 0:
                watchdog += 1
                
                if watchdog > 10:
                    Warningtag = 0
                    if len(pts) <= 11:
                        print('\033[91m'+"ERROR: Sample size is too small") #Color red
                        Warningtag = 1
                    elif len(pts) <= 21:
                        print('\033[93m'+"WARNING: Sample size is small") #Color orange
                    else:
                        print('\033[92m') #Color green

                    for i in range((len(pts)//2)):
                        dX = pts[i+len(pts)//2] - pts[i]
                        if dX > 20:
                            right += 1
                        elif dX < -20:
                            left += 1
                    if (0.5 <= right / (left + 1) <= 2) or (0.5 <= left / (right + 1) <= 2):
                        if  Warningtag != 1:
                            print('\033[93m') #Color orange
                        print("WARNING: Result unreliable")
                    print(left,"|",right)
                    dirX = "Moving left" if left > right else "Moving right" if left < right else "Movement not determined"
                    print(dirX+'\033[0m') #Color reset
                    dX = 0
                    right = 0
                    left = 0
                    pts = []

        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            if not ret:
                break
            
            cur_frame = cv2.resize(cur_frame, (1280, 720))
    
            task = pool.apply_async(process, (backSub, cur_frame))
            pending_task.append(task)
            
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