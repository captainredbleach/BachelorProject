import multiprocessing
from threading import Thread
import numpy as np
import cv2
import os
import time
from multiprocessing.pool import ThreadPool
from collections import deque

backSub = cv2.createBackgroundSubtractorMOG2(200, 32, detectShadows=False)

def filtering(frame):
    frame = cv2.bilateralFilter(frame, 5, 75, 75)
    
    frame = cv2.medianBlur(frame, 3)
    
    frame = cv2.GaussianBlur(frame,(3,3), cv2.BORDER_DEFAULT)
    
    return frame

def findbox(TempC, kernel, rgb, x,y,w,h):
    lower_b = np.array([10, 40, 100])
    upper_b = np.array([25, 150, 180])
    mask_b = cv2.inRange(TempC, lower_b, upper_b)
    
    result_b = cv2.bitwise_and(rgb[y:y + h, x:x + w], rgb[y:y + h, x:x + w], mask = mask_b)
    result_b = cv2.cvtColor(result_b, cv2.COLOR_BGR2GRAY)
    
    thresh_b = cv2.threshold(result_b, 1, 255, cv2.THRESH_BINARY)[1]
    dilation_b = cv2.dilate(thresh_b, kernel, iterations=2)
    erosion_b = cv2.dilate(dilation_b, kernel, iterations=3)
    
    bcnts = cv2.findContours(erosion_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cX = 0
    if bcnts[1] is not None:
        #print("Cage with box")
        bcnts = bcnts[0] if len(bcnts) == 2 else bcnts[1]
        bc = max(bcnts, key = cv2.contourArea)
        bx,by,bw,bh = cv2.boundingRect(bc)
        if (bw >= 100) and (bh > 100):
            cv2.rectangle(rgb[y:y + h, x:x + w], (bx, by), (bx + bw, by + bh), (0,0,255), 2)
            M = cv2.moments(bc)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(rgb[y:y + h, x:x + w], (cX, cY), 7, (255, 255, 255), -1)
            
    if cX == 0: return None, None
    return (x + cX), None


def process(self, cframe):
    bgr = cframe.copy()
    fgMask = backSub.apply(cframe)
    fg = cv2.bitwise_and(cframe, cframe, mask = fgMask)
    cv2.normalize(fg, fg, 0, 255, cv2.NORM_MINMAX)
    fgg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(9,9))
    diff = clahe.apply(fgg)
    
    kernel = np.ones((5,5), np.uint8)
    edge = cv2.Canny(diff, 150, 180, apertureSize=3, L2gradient = True)
    edge = cv2.dilate(edge, kernel, iterations=2)
    edge = cv2.erode(edge, kernel, iterations=2)
    
    fg = filtering(fg)
    hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
    hsv = filtering(hsv)
    lower_g = np.array([0, 10, 40])
    upper_g = np.array([180, 20, 80])
    # preparing the mask to overlay
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    mask_g = cv2.dilate(mask_g, None, iterations=2)
    mask_g = cv2.erode(mask_g, None, iterations=2)
    
    mask_E = cv2.bitwise_and(mask_g, edge)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    frame = filtering(diff)
    result_g = cv2.bitwise_and(frame, frame, mask = mask_E)
    
    
    thresh = cv2.threshold(result_g, 20, 255, cv2.THRESH_BINARY)[1]
    
    closening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=35)
    opening = cv2.morphologyEx(closening, cv2.MORPH_OPEN, kernel, iterations=60)
    closening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    res = None
    debug = None
    cnts = cv2.findContours(closening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts[1] is not None:
        
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        if (w < 1100) and (h < 600) and (w >= 300) and (h > 200):
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0,0,255), 2)
            hsvb = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV)
            hsvb = filtering(hsvb)
            TempC = hsvb[y:y + h, x:x + w]
            res, debug = findbox(TempC, kernel, bgr, x,y,w,h)

    return bgr, res, debug

def frameIO():
    thread_num = multiprocessing.cpu_count()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    video_name = "15FPS_720PL.mp4"
    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))
    fps = np.rint(cap.get(cv2.CAP_PROP_FPS))
    prev_frame = None
    
    pts = []
    dirX = ''
    
    while cap.isOpened():
        
        while len(pending_task) > 0 and pending_task[0].ready():
            #print(len(pending_task))
            res, diX, debug = pending_task.popleft().get()
            
            if diX is not None:
                pts.append(diX)
                
            cv2.imshow('result', res)
            
            if debug is not None:
                cv2.imshow('debug', debug)
                
            if len(pts) >= 1:
                dX = pts[-1] - pts[0]
                
                if np.abs(dX) > 100:
                    dirX = 'Right' if np.sign(dX) == 1 else 'Left'
                    print("dirx ", dirX, dX)
                    dX = 0
                    pts = []
            
                
        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            if not ret:
                break
            
            cur_frame = cv2.resize(cur_frame, (1280, 720))
    
            task = pool.apply_async(process, (0, cur_frame))
            pending_task.append(task)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
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