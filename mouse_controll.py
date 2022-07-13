import cv2
import mediapipe as mp
import hand_tracking_module as htm
import time
import numpy as np
import pyautogui

MORSE_CODE_DICT = {'a': '.-', 'b': '-...',
                   'c': '-.-.', 'd': '-..', 'e': '.',
                   'f': '..-.', 'g': '--.', 'h': '....',
                   'i': '..', 'j': '.---', 'k': '-.-',
                   'l': '.-..', 'm': '--', 'n': '-.',
                   'o': '---', 'p': '.--.', 'q': '--.-',
                   'r': '.-.', 's': '...', 't': '-',
                   'u': '..-', 'v': '...-', 'w': '.--',
                   'x': '-..-', 'y': '-.--', 'z': '--..',
                   '1': '.----', '2': '..---', '3': '...--',
                   '4': '....-', '5': '.....', '6': '-....',
                   '7': '--...', '8': '---..', '9': '----.',
                   '0': '-----', ', ': '--..--', '.': '.-.-.-',
                   '?': '..--..', '/': '-..-.', '-': '-....-',
                   '(': '-.--.', ')': '-.--.-'}


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


pyautogui.FAILSAFE = False

clicked = False
smoothening = 2.3

screen_size = pyautogui.size()
video_size = (1280, 720)
video_screen_size = (640, 360)
vid_scr_spacing = int((video_size[0] - video_screen_size[0]) / 2), int((video_size[1] - video_screen_size[1]) / 2)

cap = cv2.VideoCapture(0)
cap.set(3, video_size[0])
cap.set(4, video_size[1])
title = "Hand Detector"

img_canvas = np.zeros((*video_screen_size, 3), np.uint8)

c_time = 0  # previous time
p_time = 0  # current time

p_loc = [0, 0]  # previous location
c_loc = [0, 0]  # current location

start_morse = None
end_morse = None
morse_code = ''
morse_alphabet = ''

detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    lms = detector.get_positions(img, draw=False)
    if lms:
        # click radius
        cam2hand = detector.get_cam2hand_distance(lms)
        click_radius = 160 * cam2hand
        cv2.circle(img, lms[4][:2], int(click_radius), (255, 0, 0), 1)

        fingers_up = detector.fingers_up(img, lms)
        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:  # Right Click
            # scroll
            x, y = lms[8][:2]

            c_loc = [x, y]
            diff = c_loc[1] - p_loc[1]
            p_loc = c_loc

            if diff > 50:
                diff = 50
            elif diff < -50:
                diff = -50

            pyautogui.scroll(int(diff) * 10, _pause=False)

            # click
            dist = detector.get_point2point_distance(lms[4][:2], lms[5][:2])

            if dist < click_radius:
                if not clicked:
                    clicked = True
                    pyautogui.click(button='right', _pause=False)
                cv2.circle(img, (x, y), 12, (255, 255, 255), 2)
            else:
                clicked = False
        elif fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:  # Left Click
            # move mouse
            x, y = lms[8][:2]
            screen_x = np.interp(x, (vid_scr_spacing[0], video_screen_size[0] + vid_scr_spacing[0]),
                                 (0, screen_size[0]))
            screen_y = np.interp(y, (vid_scr_spacing[1], video_screen_size[1] + vid_scr_spacing[1]),
                                 (0, screen_size[1]))

            # smooth move
            c_loc = [screen_x, screen_y]
            c_loc[0] = p_loc[0] + (c_loc[0] - p_loc[0]) / smoothening
            c_loc[1] = p_loc[1] + (c_loc[1] - p_loc[1]) / smoothening
            p_loc = c_loc

            pyautogui.moveTo(*c_loc, _pause=False)

            # click
            dist = detector.get_point2point_distance(lms[4][:2], lms[5][:2])

            # print(dist)

            if dist < click_radius:
                if not clicked:
                    clicked = True
                    pyautogui.mouseDown(_pause=False)
                cv2.circle(img, (x, y), 12, (255, 255, 255), 2)
            else:
                clicked = False
                pyautogui.mouseUp(_pause=False)
        elif not fingers_up[1] and not fingers_up[2] and not fingers_up[3] and fingers_up[4]:  # Morse
            # input
            dist = detector.get_point2point_distance(lms[4][:2], lms[5][:2])

            if dist < click_radius:
                if not clicked:
                    clicked = True
                    start_morse = time.time()
                cv2.circle(img, lms[20][:2], 12, (255, 255, 255), 2)
            else:
                if clicked:
                    clicked = False
                    end_morse = time.time()

            if start_morse and end_morse:
                diff = end_morse - start_morse
                if diff < 0.3:
                    morse_code += '.'
                elif 0.3 < diff < 1.5:
                    morse_code += '-'
                else:
                    if morse_alphabet:
                        pyautogui.press(morse_alphabet, _pause=False)
                    morse_code = ''
                    morse_alphabet = ''

                morse_alphabet = get_key_from_value(MORSE_CODE_DICT, morse_code)

                start_morse = None
                end_morse = None


        else:
            p_loc = lms[8][:2]
            pyautogui.mouseUp(_pause=False)

    # copy img cutout to canvas
    img_canvas = img[vid_scr_spacing[1]:video_screen_size[1] + vid_scr_spacing[1],
                 vid_scr_spacing[0]:video_screen_size[0] + vid_scr_spacing[0]]

    # morse
    if morse_code:
        cv2.putText(img_canvas, f"{morse_code} = {str(morse_alphabet)}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    # FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img_canvas, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # cv2.imshow(title, img)
    cv2.imshow(title, img_canvas)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:  # ESC or X button
        break
