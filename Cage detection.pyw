import multiprocessing
from threading import Thread
import numpy as np
import cv2
import os
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import current_process
from collections import deque
import queue

my_queue = queue.Queue()

def storeInQueue(f):
    def wrapper(*args):
        my_queue.put(f(*args))
    return wrapper

def filtering(frame):
    frame = cv2.bilateralFilter(frame, 5, 75, 75)
    
    frame = cv2.medianBlur(frame, 3)
    
    frame = cv2.GaussianBlur(frame,(3,3), cv2.BORDER_DEFAULT)
    
    return frame

@storeInQueue
def findbox(TempC, kernel, rgb, x,y,w,h):
    lower_b = np.array([25, 50, 100])
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
        for bc in bcnts:
            bc = max(bcnts, key = cv2.contourArea)
            bx,by,bw,bh = cv2.boundingRect(bc)
            if (bw >= 100) and (bh > 100):
                cv2.rectangle(rgb[y:y + h, x:x + w], (bx, by), (bx + bw, by + bh), (0,0,255), 2)
                M = cv2.moments(bc)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(rgb[y:y + h, x:x + w], (cX, cY), 7, (255, 255, 255), -1)
    if cX != 0:
        return (x + cX), None
    else:
        return None, None


def process(rgb, hsv, frame):
    
    edge = cv2.Canny(frame, 150, 200, apertureSize=3, L2gradient = True)
    
    frame = filtering(frame)
    #lower_g = np.array([5, 30, 10])
    #upper_g = np.array([20, 255, 30])
    lower_g = np.array([0, 0, 30])
    upper_g = np.array([360, 15, 120])
 
    # preparing the mask to overlay
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    
    mask_E = cv2.bitwise_and(mask_g, edge)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    result_g = cv2.bitwise_and(frame, frame, mask = mask_E)
    
    kernel = np.ones((5,5), np.uint8)
    thresha = cv2.threshold(result_g, 58, 255, cv2.THRESH_TOZERO_INV)[1]
    thresh = cv2.threshold(thresha, 1, 255, cv2.THRESH_BINARY)[1]
    img_dilation = cv2.dilate(thresh, kernel, iterations=32)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=33)
    
    img_erosion2 = cv2.erode(img_erosion, kernel, iterations=1)
    img_dilation2 = cv2.dilate(img_erosion2, kernel, iterations=50)
    
    finalE = cv2.erode(img_dilation2, kernel, iterations=70)

    res = None
    br = None
    cnts = cv2.findContours(finalE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        c = max(cnts, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if (w < 1100) and (h < 600) and (w >= 300) and (h > 200):
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0,0,255), 2)
            TempC = hsv[y:y + h, x:x + w]
            t1 = Thread(target=findbox, args=(TempC, kernel, rgb, x,y,w,h)) 
            t1.start()
            res, br = my_queue.get()
            
    
    return rgb, res, br

def frameIO():
    thread_num = multiprocessing.cpu_count()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    video_name = "15FPS_720PL.mp4"
    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))
    #fps = np.rint(cap.get(cv2.CAP_PROP_FPS)) * thread_num
    prev_frame = None
    pts = deque(maxlen = 20)
    dirX = ''
    while cap.isOpened():
        while len(pending_task) > 0 and pending_task[0].ready():
            #print(len(pending_task))
            res, diX, debug = pending_task.popleft().get()
            pts.appendleft(diX)
            cv2.imshow('result', res)
            
            if debug is not None:
                cv2.imshow('debug', debug)   
            if pts[-1] and pts[0] is not None:
                dX = pts.pop() - pts.popleft()
                if np.abs(dX) > 100:
                    dirX = 'Venstre' if np.sign(dX) == 1 else 'Højre'
                    print("dirx ", dirX)
            
                
        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            
            if ret:
                cur_frame = cv2.resize(cur_frame, (1280, 720))
                
        
                if prev_frame is not None:

                    rgb = cur_frame.copy()
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV_FULL)
                    diff = cv2.absdiff(prev_frame, rgb, 0.95)
                    cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
                    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(6,11))
                    diff = clahe.apply(diff)

                    task = pool.apply_async(process, (rgb, hsv, diff))
                    pending_task.append(task)
                    
            else:
                break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #time.sleep(1 / fps)
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