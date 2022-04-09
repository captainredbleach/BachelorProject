import numpy as np
import cv2
import os
import time
from multiprocessing.pool import ThreadPool
from collections import deque

def process(rgb, hsv, frame):
    
    
    frame = cv2.medianBlur(frame, 11)
    
    frame = cv2.GaussianBlur(frame,(5,5),0)
    
    lower_g = np.array([30, 35, 80])
    upper_g = np.array([40, 40, 255])
 
    # preparing the mask to overlay
    mask_g = cv2.inRange(hsv, lower_g, upper_g)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-grey regions
    result_g = cv2.bitwise_and(frame, frame, mask = mask_g)
    
    result_g = abs(result_g) * 5
 
    
    imgGrey = cv2.cvtColor(result_g, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgGrey, 30, 255, cv2.THRESH_BINARY)
    img_dilation = cv2.dilate(thresh, kernel, iterations=50)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=80)
    
    img_erosion2 = cv2.erode(img_erosion, kernel, iterations=6)
    img_dilation2 = cv2.dilate(img_erosion2, kernel, iterations=60)
    
    finalE = cv2.erode(img_dilation2, kernel, iterations=1)
    
    img_final = abs(img_dilation2) - abs(finalE)
    
    
    
    cnts = cv2.findContours(img_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0,0,255), 2)

    
    #Hough Line Transform 
    #linesP = cv2.HoughLinesP(img_final, 1, np.pi / 180, 20, None, 0, 0)

    # Draw the lines
    #if linesP is not None:
    #    for i in range(0, len(linesP)):
    #        l = linesP[i][0]
    #        cv2.line(rgb, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            
    return rgb
    
    

if __name__ == '__main__':     
    video_name = "15FPS_720P.mp4"

    # Define the fps for the video
    fps = 30

    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))

    # Dequeue for storing the previous K frames
    prev_frame = None 
    kernel = np.ones((5,5), np.uint8)
    
    thread_num = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    #size = (frame_width, frame_height)
    #result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)




    while(cap.isOpened()):
        while len(pending_task) > 0 and pending_task[0].ready():
            res = pending_task.popleft().get()
            cv2.imshow('result', res)
            
        
        #cur_frame_cpy = cur_frame.copy()
        if len(pending_task) < thread_num:
            ret, cur_frame = cap.read()
            if ret:
                
        
                if prev_frame is not None:
                    rgb = cur_frame
                    # It converts the BGR color space of image to HSV color space
                    hsv = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2HSV)
                    
                    diff = cv2.absdiff(prev_frame, cur_frame)
                    diff = diff * 3
                    
                    
                    
                    task = pool.apply_async(process, (rgb, hsv, diff))
                    pending_task.append(task)
                    
                    #result.write(rgb)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(1 / fps)
        prev_frame = cur_frame
            
    cv2.destroyAllWindows()
    cap.release()
    #result.release()