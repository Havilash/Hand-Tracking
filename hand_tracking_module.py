import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector:
    finger_tips = [4, 8, 12, 16, 20]

    def __init__(self, **kwargs):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(**kwargs)

        self.cx = None
        self.cy = None
        self.results = None

    def find_hands(self, img, draw=True):
        bbox = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:  # if hands exists
            for hand in self.results.multi_hand_landmarks:  # loop through hands
                if draw:
                    mp.solutions.drawing_utils.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_positions(self, img, hand_num=0, draw=True):
        lms = []
        if self.results.multi_hand_landmarks:  # if hands exists
            hand = self.results.multi_hand_landmarks[hand_num]
            for lm in hand.landmark:
                h, w, c = img.shape
                x, y, z = int(np.interp(lm.x, [0, 1], [0, w])), \
                          int(np.interp(lm.y, [0, 1], [0, h])), \
                          lm.z

                lms.append([x, y, z])

                if draw:
                    cv2.circle(img, (x, y), 5, (255, 255, 255), 2)
        return lms

    def get_bbox(self, img, lms, draw=True):
        bbox = None
        if lms:
            lms_x = [item[0] for item in lms]
            lms_y = [item[1] for item in lms]
            bbox = (min(lms_x), min(lms_y)), (max(lms_x), max(lms_y))

            if draw:
                cv2.rectangle(img, (bbox[0][0] - 10, bbox[0][1] - 10), (bbox[1][0] + 10, bbox[1][1] + 10),
                              (255, 0, 0), 2)
        return bbox

    def get_center(self, lms):
        cx = (sum([lms[i - 3][0] for i in self.finger_tips])) / 5
        cy = (sum([lms[i - 3][1] for i in self.finger_tips])) / 5

        return cx, cy

    def fingers_up(self, img, lms, draw=True):
        fingers_up = []
        closed_radius = 80

        cx, cy = self.get_center(lms)

        p2 = np.array([cx, cy])
        for i in self.finger_tips:
            p1 = np.array(lms[i][:2])

            squared_dist = np.sum((p1 - p2) ** 2, axis=0)
            dist = np.sqrt(squared_dist)

            if dist > closed_radius:
                fingers_up.append(True)
                if draw:
                    cv2.circle(img, (int(p1[0]), int(p1[1])), 4, (255, 0, 0), cv2.FILLED)
            else:
                fingers_up.append(False)
        return fingers_up

    def get_point2point_distance(self, pos1, pos2):
        p1 = np.array(pos1)
        p2 = np.array(pos2)

        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        dist = np.sqrt(squared_dist)

        return dist

    def get_cam2hand_distance(self, lms):
        p1 = self.get_center(lms)
        p2 = lms[0][:2]
        dist = self.get_point2point_distance(p1, p2)
        dist_percent = np.interp(dist, [0, 400], [0, 1])

        return dist_percent


def main():
    video_size = (640, 480)

    cap = cv2.VideoCapture(0)
    cap.set(3, video_size[0])
    cap.set(4, video_size[1])
    title = "Hand Detector"

    c_time = 0
    p_time = 0

    detector = HandDetector()
    while True:
        success, img = cap.read()

        img = detector.find_hands(img)
        lms = detector.get_positions(img)
        if lms:
            print(lms)
            detector.get_bbox(img, lms)

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow(title, img)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # ESC or X button
            break


if __name__ == "__main__":
    main()
