import cv2
import os
import numpy as np

IMAGES_DIR = 'images'
BASE_FILE_NAME = 'base.jpg'
FILE_NAME = os.path.join(IMAGES_DIR, BASE_FILE_NAME)
LOWER_GREEN = np.array([65, 60, 60])
UPPER_GREEN = np.array([80, 255, 255])
kernel = np.ones((5,5), int)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

base_file = None

if os.path.exists(FILE_NAME):
    base_file = cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)

while(True):
    ret, frame = vid.read()
    
    # cv2.imshow('frame', frame)
    if base_file is not None:
        # Let's verify the cape here
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, LOWER_GREEN, UPPER_GREEN)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        
        res = cv2.bitwise_and(frame, frame, mask=dilated)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(res_gray, 3, 255, cv2.THRESH_BINARY)
        countours, _ = cv2.findContours(thresh, -1, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_cnt = []
        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        if max_area > 100000:
            mask = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
            # cv2.imshow('Mask', mask)
            # cv2.waitKey(0)
            cv2.drawContours(frame, [cnt], -1, (0, 0, 0), -1)
            # cv2.imshow('Frame', frame)
            # cv2.waitKey(0)

            roi = cv2.bitwise_and(base_file, mask)

            # cv2.imshow('roi', roi)
            # cv2.waitKey(0)

            frame_final = cv2.bitwise_xor(roi, frame, mask=None)

            cv2.imshow('frame', frame_final)
            # cv2.waitKey(0)
            
            print('Max area {0}'.format(max_area))
            
            # cv2.imshow('frame', res)
        else:
            cv2.imshow('frame', frame)

    else:
        # If pressed "C" key, then save the image to disk
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print('Capturing image')
            base_file = frame.copy()
            cv2.imwrite(FILE_NAME, base_file)
        
        cv2.imshow('frame', frame)

    # If pressed "Q" key, then quit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()