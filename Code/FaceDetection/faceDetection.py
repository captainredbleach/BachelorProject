import numpy as np
import cv2
import threading

def math(faces, img, gray):
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def capture():
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            
            scaleFactor=1.2,
            minNeighbors=5
            ,     
            minSize=(20, 20)
        )

        t1 = threading.Thread(target=math, args=(faces, img, gray))
        t1.start()
        t1.join()

        cv2.imshow('video',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    capture()
