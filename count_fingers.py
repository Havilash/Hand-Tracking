import time
import hand_tracking_module as htm
import cv2
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()

video_size = (640, 480)

cap = cv2.VideoCapture(0)
cap.set(3, video_size[0])
cap.set(4, video_size[1])
title = "Hand Detector"

c_time = 0
p_time = 0

detector = htm.HandDetector(max_num_hands=1)
while True:
    success, img = cap.read()

    img = detector.find_hands(img)
    lms = detector.get_positions(img, draw=False)

    if lms:
        fingers_up = detector.fingers_up(img, lms)
        cv2.putText(img, str(fingers_up.count(True)), (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)

    # FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(title, img)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # ESC or X button
        break
