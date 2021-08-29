import cv2
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
import time
import autopy
from autopy.mouse import Button
from mediapipe.python.solutions.hands import HandLandmark

w, h = 640, 480
W, H = autopy.screen.size()

cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)
pTime = 0

x1, x2, y1, y2 = 0, 0, 0, 0
ox, oy = 0, 0

dis = 30
smt = 3

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                # h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == HandLandmark.WRIST:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

                    x1 = np.interp(cx, (0, w), (0, W))
                    y1 = np.interp(cy, (0, h), (0, H))

                    if 0 <= x1 < W and 0 <= y1 < H and abs(x1 - ox) > smt and abs(y1 - oy) > smt:
                        autopy.mouse.move(x1, y1)
                        ox = x1
                        oy = y1

                        # autopy.mouse.click()

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
