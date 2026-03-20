import pickle
from itertools import product
import numpy as np
import cv2
from PIL import ImageGrab
import mss
import time
import math


def capture_once():
    '''
    Function Used for Rune Capture. 
    TODO : Modify this into mss format.
    '''
    rune_left_x, rune_left_y, rune_right_x, rune_right_y = 453, 185, 913, 320
    ps = np.array(ImageGrab.grab(bbox=(rune_left_x,rune_left_y,rune_right_x,rune_right_y)).convert('RGB'))
    ps = cv2.cvtColor(ps, cv2.COLOR_BGR2RGB)
    return ps


def angle_averaging(angles):
    if(len(angles) == 0):
        return np.nan
    xx, yy = 0, 0
    for angle in angles:
        xx += np.cos(np.deg2rad(angle))
        yy += np.sin(np.deg2rad(angle))
    res = math.degrees(np.arctan2(yy,xx))
    if(res < 0):
        res += 360
    return res


def get_initial_answer(yolo) :
    flag = False
    for _ in range(3) :
        time.sleep(0.05)
        cv2.imwrite('src/question.jpg', capture_once())
        prev = yolo.detect_v5("src/question.jpg")
        if len(prev["res"]) == 4:
            flag = True
            break
    assert flag, "abnormal detection"

    flag = False
    for _ in range(3) :
        time.sleep(0.05)
        cv2.imwrite('src/question.jpg', capture_once())
        cur = yolo.detect_v5("src/question.jpg")

        if prev["res"] == cur["res"] :
            flag = True
            break
    assert flag, "abnormal detection"

    return cur["res"], cur["rotate_index"], cur["centers"]

def get_angle(crop, x_center, y_center) :


    xx, yy = 0, 0
    radius = 25
    default = [i for i in range(-radius, radius+1)]
    dxdy = list(product(default, default))

    ## 빨간 영역의 무게중심
    cnt = 0
    for (dx, dy) in dxdy:
        try :
            if( crop[int(y_center - dy)][int(x_center + dx)][2] != 0 ):
                xx += dx
                yy += dy
                cnt += 1
        except :
            pass
    xx, yy = xx / cnt, yy / cnt
    ang = math.degrees(np.arctan2(yy,xx))

    if(ang < 0):
        ang += 360
    return ang

def masking(img) :
    from matplotlib import pyplot as plt

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,175,191), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (179,175,191), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2)
    crop = cv2.bitwise_and(img, img, mask=mask)
    return crop

def chulkuk_parser(after_chulkuk_angle) :
    return [find_nearest(angle_averaging(angle_list)) for angle_list in after_chulkuk_angle]



def find_nearest(value):
    try :
        return round(value / 90) * 90
    except :
        return -1


def rune_video(num = 60):
    rune_left_x, rune_left_y, rune_right_x, rune_right_y = 453, 185, 913, 320
    with mss.mss() as sct:
        monitor = {"left": rune_left_x, "top": rune_left_y, "width": rune_right_x - rune_left_x, "height": rune_right_y - rune_left_y}
        temp = []
        for i in range(num):
            img = np.array(sct.grab(monitor))[:, :, :3]
            temp.append(img)
            time.sleep(5/1000)
        cv2.destroyAllWindows()
    with open('rune_video.pkl', 'wb') as f:
        pickle.dump(temp, f)

