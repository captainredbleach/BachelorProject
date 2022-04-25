import os
import cv2
import time

if __name__ == '__main__':

    # Define the name of the video file
    video_name = "15FPS_720P-C.mp4"

    # Define the fps for the video
    fps = 30

    path = os.path.dirname(os.path.realpath(__file__))
    cap = cv2.VideoCapture(os.path.join(path, video_name))

    # Dequeue for storing the previous K frames
    prev_frame = None

    # Iterate over frames in the in the video
    while (cap.isOpened()):
        ret, cur_frame = cap.read()
        cur_frame_cpy = cur_frame.copy()

        if ret == True:

            # Background subtraction based on the previous frames
            if prev_frame is not None:

                # Subtract the current frame from the previous frames
                diff = cv2.absdiff(prev_frame, cur_frame)

                # Remove noise
                diff = cv2.medianBlur(diff, 5)

                # Display the resulting frame
                cv2.imshow('diff', diff)

                # Turn into grayscale
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                # Apply gauusian blur to the gray image
                blur = cv2.GaussianBlur(gray, (5, 5), 0)

                # Apply thresholding to the blur image
                ret, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

                # Erode the thresholded image
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded = cv2.erode(thresh, kernel, iterations=3)

                # Dialate the thresholded image
                dilated = cv2.dilate(eroded, kernel, iterations=10)

                cv2.imshow('dilated', dilated)

            # Display the resulting frame
            cv2.imshow('frame', cur_frame_cpy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1 / fps)
            prev_frame = cur_frame
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
