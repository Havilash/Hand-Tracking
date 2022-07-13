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

    dist_range = [30, 175]
    dist = 0
    dist_vol = 0
    dist_percent = 0
    if lms:
        p1 = np.array(lms[8][:2])
        p2 = np.array(lms[4][:2])
        x1, y1 = p1
        x2, y2 = p2

        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        dist = np.sqrt(squared_dist)

        dist_vol = np.interp(dist, dist_range, volume_range[:2])
        dist_percent = int(np.interp(dist, dist_range, [0, 100]))

        volume.SetMasterVolumeLevel(dist_vol, None)

        cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (255, 0, 0), cv2.FILLED)

        if dist_percent == 0:
            cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 10, (0, 255, 0), cv2.FILLED)

    # volume-bar
    dist_rect = np.interp(dist, dist_range, [200, 70])
    cv2.rectangle(img, (30, int(dist_rect)), (60, 200), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(img, (30, 70), (60, 200), (255, 0, 0), 2)
    cv2.putText(img, f"{dist_percent} %", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    # FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(title, img)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # ESC or X button
        break
