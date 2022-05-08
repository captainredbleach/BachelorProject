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

def draw_flow(img, prevgray, gray, step=24):
    
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.3, 10, 25, 2, 15, 1.7, 0)
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)    
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    
    direction = [fx[x] for x in np.arange(len(lines)) if np.abs(fx[x]) > 20]
    
    if len(direction)>=1:
        d = np.sum(direction) // len(direction)

def findbox(TempC, kernel, rgb, x,y,w,h):
    lower_b = np.array([20, 50, 100])
    upper_b = np.array([30, 100, 130])
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
    if cX != 0:
        return (x + cX), mask_b
    else:
        return None, None


def process(cframe, pframe):
    bgr = cframe.copy()
    
    hsv = cv2.cvtColor(cframe, cv2.COLOR_BGR2HSV_FULL)
    diff = cv2.absdiff(pframe, cframe, 0.95)
    cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6,11))
    diff = clahe.apply(diff)
    
    kernel = np.ones((5,5), np.uint8)
    edge = cv2.Canny(diff, 150, 180, apertureSize=3, L2gradient = True)
    edge = cv2.dilate(edge, kernel, iterations=3)
    edge = cv2.erode(edge, kernel, iterations=8)
    frame = filtering(diff)
    #lower_g = np.array([5, 30, 10])
    #upper_g = np.array([20, 255, 30])
    lower_g = np.array([0, 25, 75])
    upper_g = np.array([360, 30, 120])
    hsv = filtering(hsv)
    # preparing the mask to overlay
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    mask_g = cv2.dilate(mask_g, None, iterations=2)
    mask_g = cv2.erode(mask_g, None, iterations=3)
    
    mask_E = cv2.bitwise_and(mask_g, edge)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    result_g = cv2.bitwise_and(frame, frame, mask = mask_E)
    
    
    thresha = cv2.threshold(result_g, 35, 255, cv2.THRESH_TOZERO_INV)[1]
    thresh = cv2.threshold(thresha, 5, 255, cv2.THRESH_BINARY)[1]
    img_dilation = cv2.dilate(thresh, kernel, iterations=69)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=59)
    
    img_erosion2 = cv2.erode(img_erosion, kernel, iterations=1)
    img_dilation2 = cv2.dilate(img_erosion2, kernel, iterations=75)
    
    finalE = cv2.erode(img_dilation2, kernel, iterations=65)

    res = None
    br = None
    cnts = cv2.findContours(finalE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts[1] is not None:
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if (w < 1100) and (h < 600) and (w >= 300) and (h > 200):
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0,0,255), 2)
            TempC = hsv[y:y + h, x:x + w]
            gray = cv2.cvtColor(cframe[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            prevgray = cv2.cvtColor(pframe[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            t1 = Thread(target=draw_flow, args=(bgr[y:y + h, x:x + w], prevgray, gray))
            t1.start()
            res, br = findbox(TempC, kernel, bgr, x,y,w,h)
            t1.join()
            
    
    return bgr, res, br

def frameIO():
    thread_num = multiprocessing.cpu_count()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    video_name = "15FPS_720PL.mp4"
    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))
    fps = np.rint(cap.get(cv2.CAP_PROP_FPS))
    prev_frame = None
    pts = deque(maxlen = 20)
    dirX = ''
    counter = 0
    while cap.isOpened():
        while len(pending_task) > 0 and pending_task[0].ready():
            #print(len(pending_task))
            res, diX, debug = pending_task.popleft().get()
            pts.appendleft(diX)
            counter += 1
            cv2.imshow('result', res)
            
            if debug is not None:
                cv2.imshow('debug', debug)
            if pts[-1] and pts[0] is not None and len(pts) >= 2 and counter > 6:
                dX = pts.pop() - pts.popleft()
                if 500 > np.abs(dX) > 200:
                    dirX = 'Left' if np.sign(dX) == 1 else 'Right'
                    print("dirx ", dirX, dX)
                counter = 0
            
                
        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            
            if ret:
                cur_frame = cv2.resize(cur_frame, (1280, 720))
                
        
                if prev_frame is not None:
                    task = pool.apply_async(process, (cur_frame, prev_frame))
                    pending_task.append(task)
                    
            else:
                break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(1 / fps)
        prev_frame = cur_frame.copy()
    cv2.destroyAllWindows()
    cap.release()

def main():
    T1 = Thread(target=frameIO) 
    T1.start()
    T1.join()
    #RFID her

if __name__ == '__main__':
    main()