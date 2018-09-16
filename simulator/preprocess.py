import cv2
import numpy as np
from PIL import ImageGrab
import win32gui
import skimage.measure
import win32api
import win32con
import time
from simulator.tools import *


def print_screen(name):
    hwnd = win32gui.FindWindow(None, 'BlueStacks App Player')
    rect = win32gui.GetWindowRect(hwnd)
    # rect = [r * 1.5 for r in rect]
    img = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))
    frame = np.array(img)
    frame = frame[:, :, [2, 1, 0]]
    # cv2.imshow('frame', frame)
    cv2.imwrite(name, frame)


if __name__ == '__main__':
    # templates = load_templates()
    # mini_templates = dict()
    # for t in templates:
    #     if t == 'Joker':
    #         mini_templates[t] = cv2.imread('./templates/Joker_mini.png', cv2.IMREAD_GRAYSCALE)
    #     else:
    #         mini_templates[t] = cv2.resize(templates[t], (0, 0), fx=0.7, fy=0.7)
    # cards, _ = get_cards_bboxes(cv2.imread('debug3.png'), mini_templates, 1)
    # print(cards)
    # print_screen('test.png')
    # img = cv2.imread('train.png', cv2.IMREAD_GRAYSCALE)
    # img = 255 - img
    # # labels = np.zeros_like(img, dtype=np.int32)
    # labels = skimage.measure.label(img, 4, 0)
    # for l in np.unique(labels):
    #     print(l)
    #     if l == 0:
    #         continue
    #     labelMask = np.zeros(labels.shape, dtype="uint8")
    #     labelMask[labels == l] = 255
    #     # mask = labels.copy()
    #     # mask[labels != l] = 0
    #     # mask[labels > 0] = 255
    #     pts = cv2.findNonZero(labelMask)
    #     rect = cv2.boundingRect(pts)
    #     labelMask = labelMask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    #     cv2.imwrite('1.png', labelMask)
    #     cv2.imshow('test', labelMask)
    #     cv2.waitKey(0)
    i = 0
    while os.path.exists('./photo/%d.png' % i):
        i += 1
    print(who_is_lord(cv2.imread('./photo/%d.png' % 9)))
    # print_screen('./photo/%d.png' % i)
    # i = 0
    # while True:
    #     time.sleep(0.2)
    #     print_screen('./video/f%d.png' % i)
    #     i += 1
    # # print(rect)
    # img = cv2.imread('fzcvt.png')
    # img = img[45:61, 9:107]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('mask', mask)
    # cv2.imwrite('train.png', mask)
    # cv2.waitKey(0)
