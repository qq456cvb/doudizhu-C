#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Yang You & Liangwei Li

"""
import cv2
import numpy as np

from skimage import data, segmentation, color, measure
from skimage.future import graph
from matplotlib import pyplot as plt
import os
from simulator.config import ConfigurationOffline
import time
from PIL import ImageGrab
import win32gui
import skimage.measure
import win32api, win32con

cf_offline = ConfigurationOffline()

DEBUG = False


def locate_cards_position(img, x_left, x_max, y_up, y_bottom, cards_up, cards_bottom, mini=False):
    bboxes = []
    while True:
        while not np.all(img[y_up:y_bottom, x_left] > 250):
            if x_left >= x_max:
                break
            x_left += 1
        if x_left >= x_max:
            break
        x_right = x_left
        while not (np.all(img[y_up:y_bottom, x_right] < 180) and x_right - x_left > (50 if mini else 80)):
            x_right += 1

        bboxes.append([x_left + 2, cards_up, x_left + (50 if mini else 85), cards_bottom])
        # bbox = bboxes[-1]
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
        # cv2.imshow('test', img)
        # cv2.waitKey()
        x_left = x_right
    # if DEBUG:
    #     for bbox in bboxes:
    #         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0))
    #     cv2.imshow('test', img)
    #     cv2.waitKey()
    return bboxes

    # print((x_right - x_left - 130) / 82.5)


def load_templates():
    card_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'Joker']
    res = dict()
    for c in card_names:
        res[c] = cv2.imread('./templates/%s.png' % c, cv2.IMREAD_GRAYSCALE)
        # pts = cv2.findNonZero(255 - res[c])
        # rect = cv2.boundingRect(pts)
        # res[c] = res[c][rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        # cv2.imwrite('%s.png' % c, res[c])
        # cv2.imshow('test', res[c][rect[1]:rect[3], rect[0]:rect[2]])
        # cv2.waitKey(0)
    return res


def load_mini_templates(templates):
    mini_templates = dict()
    for t in templates:
        mini_templates[t] = cv2.resize(templates[t], (0, 0), fx=0.59, fy=0.59)
    return mini_templates


def load_tiny_templates():
    card_names = [str(i) for i in range(10)]
    res = dict()
    for c in card_names:
        res[c] = cv2.imread('./templates/tiny%s.png' % c, cv2.IMREAD_GRAYSCALE)
        res[c] = cv2.resize(res[c], (0, 0), fx=21 / 13, fy=21 / 13)
    return res


def parse_card_type(templates, img, bbox, binarize=True, recursive=True):
    subimg = img[max(bbox[1] - 3, 0):bbox[3] + 3, max(bbox[0] - 3, 0):bbox[2] + 3]
    if binarize:
        gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = subimg
    max_response = 0.
    max_t = None
    if DEBUG:
        cv2.imshow('subimg', mask)
        cv2.waitKey(0)
    for t in templates:
        if templates[t] is not None:
            if mask.shape[0] < templates[t].shape[0] or mask.shape[1] < templates[t].shape[1]:
                continue
            res = cv2.matchTemplate(mask, templates[t], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_response:
                max_response = max_val
                max_t = t
    if max_response < 0.7:
        return None
    # filter joker
    if max_t == 'Joker' and recursive:
        _, test = cv2.threshold(subimg[:, :, 2], 127, 255, cv2.THRESH_BINARY)
        refined = parse_card_type(templates, test, (0, 0, test.shape[1], test.shape[0]), False, False)
        if refined == 'Joker':
            max_t = '*'
        else:
            max_t = '$'
    return max_t


def parse_card_cnt(templates, img, bbox, binarize=True):
    subimg = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if binarize:
        subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        _, subimg = cv2.threshold(subimg, 200, 255, cv2.THRESH_BINARY)
    labels = measure.label(subimg, 4, 0)
    cnt = ''
    for l in np.unique(labels):
        if l == 0:
            continue
        labelMask = np.zeros(labels.shape, dtype="uint8")
        labelMask[labels == l] = 255
        pts = cv2.findNonZero(labelMask)
        rect = cv2.boundingRect(pts)
        # cv2.imshow('l', labelMask)
        # cv2.waitKey(0)
        cnt += str(parse_card_type(templates, subimg, [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]],
                                   binarize=False))
    if cnt == '':
        return 0
    cnt = int(cnt)
    return cnt


# spin lock, pos: x, y
def spin(pos, color, interval=0.1, max_wait=0.):
    wait = 0.
    while not np.array_equal(get_window_img()[pos[1], pos[0], :], np.array(color)):
        time.sleep(interval)
        wait += interval
        if max_wait != 0. and wait >= max_wait:
            return False
    return True


def spin_multiple(pos, color, interval=0.1):
    while True:
        for i in range(len(pos)):
            if np.array_equal(get_window_img()[pos[i][1], pos[i][0], :], np.array(color[i])):
                return i
        time.sleep(interval)


# get the image pixels(as np.array)
def get_window_img(path):
    return np.array(cv2.imread(path))


def show_img(img_array):
    cv2.imshow('image', img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_color(truth_color, compared_color, difference=0):
    """
    judge whether a pixel belongs to a specific color
    :param truth_color: the standard color ([**, **, **])
    :param compared_color: the color to be judged
    :return:
    """
    cnt = 0
    for idx in range(cf_offline.channels):
        if np.abs(int(truth_color[idx]) - int(compared_color[idx])) <= difference:
            cnt += 1
    if cnt == 3:
        return True
    else:
        return False


def draw_bounding_box(image, bbox):
    """
    draw a bounding box of an image
    :param image:
    :param bbox:
    :return:
    """
    start_x = bbox[0]
    start_y = bbox[1]
    end_x = bbox[2]
    end_y = bbox[3]
    image[start_y, start_x:end_x, :] = 255
    image[end_y, start_x:end_x, :] = 255
    image[start_y:end_y, start_x, :] = 255
    image[start_y:end_y, end_x, :] = 255
    return image


def get_current_button_action(current_image):
    actions = {}
    # find all buttons
    for button in cf_offline.button_information:
        find_flag = True
        top = cf_offline.button_information[button][0]
        left_pos = cf_offline.button_information[button][1]
        button_array = cf_offline.button_information[button][2]
        effective_band = current_image[top, left_pos:left_pos + cf_offline.two_words_button_width, :]
        length = effective_band.shape[0]
        assert (length == button_array.shape[0])
        for idx in range(length):
            if not compare_color(button_array[idx], effective_band[idx], difference=0):
                find_flag = False
                break
        if find_flag:
            actions[button] = [left_pos, cf_offline.button_up_margin, left_pos + cf_offline.two_words_button_width,
                               cf_offline.button_down_margin]
        if button == "continuous defeat" and find_flag:
            actions[button] = [1654, 201, 1704, 241]
    # judge whether end
    result = is_win(current_image)
    if result == 0:
        actions['defeat'] = [1650, 114, 1706, 156]
    elif result == 1:
        actions['victory'] = [1650, 114, 1706, 156]
    return actions


def is_win(current_image):
    if compare_color(cf_offline.winning_color,
                     current_image[cf_offline.winning_losing_top, cf_offline.winning_losing_left, :], 0):
        return 1
    elif compare_color(cf_offline.losing_color,
                       current_image[cf_offline.winning_losing_top, cf_offline.winning_losing_left, :], 0):
        return 0
    else:
        return -1


def who_is_lord(image):
    """
    judge which player is lord
    :param image: current image
    :return: 0: self is lord, 1: left side player is lord, 2:right side player is lord, -1:no one is lord
    """
    if compare_color(image[cf_offline.self_lord_y, cf_offline.self_lord_x, :], cf_offline.self_lord_color, 0):
        return 0
    elif compare_color(image[cf_offline.left_lord_y, cf_offline.left_lord_x, :], cf_offline.left_lord_color, 0):
        return 1
    elif compare_color(image[cf_offline.right_lord_y, cf_offline.right_lord_x, :], cf_offline.right_lord_color, 0):
        return 2
    else:
        return -1


def get_window_rect(hwnd):
    rect = win32gui.GetWindowRect(hwnd)
    return rect


def grab_screen():
    # base_time = time.time()
    # while True:
    #     i = (time.time() - base_time) // 0.5 + 11
    #     print('./video/f%d.png' % i)
    #     yield cv2.imread('./video/f%d.png' % i)
    hwnd = win32gui.FindWindow(None, 'BlueStacks App Player')
    rect = win32gui.GetWindowRect(hwnd)
    # rect = [r * 1.5 for r in rect]
    img = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))

    frame = np.array(img)
    frame = frame[46:1126, 6:1926, :]
    frame = frame[:, :, [2, 1, 0]]
    cv2.imwrite('test.png', frame)
    # cv2.imwrite(name, frame)
    return frame


def click(x, y, offset=(0, 0)):
    win32api.SetCursorPos((offset[0] + x, offset[1] + y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, offset[0] + x, offset[1] + y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, offset[0] + x, offset[1] + y, 0, 0)


# get cards and their bboxes, role = 0 for self, 1 for left, 2 for right
def get_cards_bboxes(img, templates, role=0, bboxes=None):
    if bboxes is None:
        if role == 0:
            bboxes = locate_cards_position(img, 0, img.shape[1] - 1, 870, 880, 860, 930)
        elif role == 1:
            bboxes = locate_cards_position(img, 0, img.shape[1] // 2, 467, 477, 460, 500, True)
            bboxes += locate_cards_position(img, 0, img.shape[1] // 2, 642, 652, 528, 568, True)
        elif role == 2:
            bboxes = locate_cards_position(img, img.shape[1] // 2, img.shape[1] - 1, 467, 477, 460, 500, True)
            bboxes += locate_cards_position(img, img.shape[1] // 2, img.shape[1] - 1, 642, 652, 528, 568, True)
        else:
            raise Exception('unexpected role')
    cards = []
    for bbox in bboxes:
        cards.append(parse_card_type(templates, img, bbox))
    return cards, bboxes


def get_opponent_cnts(img, templates):
    return (parse_card_cnt(templates, img, [301, 371, 336, 398], True),
            parse_card_cnt(templates, img, [954, 371, 988, 398], True))


import multiprocessing


class A(multiprocessing.Process):
    def __init__(self):
        super(A, self).__init__()

    def set_test(self):
        self.test = 'TEST'
        self.start()

    def run(self):
        print(self.test)


if __name__ == '__main__':
    # img = cv2.imread('./photo/load_right.png')
    # tiny_templates = load_tiny_templates()
    # print(parse_card_cnt(tiny_templates, img, [301, 371, 336, 398], True))
    # print(parse_card_cnt(tiny_templates, img, [954, 371, 988, 398], True))
    # exit()
    # cv2.imshow('img', subimg)
    # cv2.waitKey(0)
    # 4,30 - 1284, 750
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # s = hsv[:, :, 1]
    # v = hsv[:, :, 2]
    # s[s < 20] = 0
    # s[s >= 20] = 255
    # v = cv2.Canny(v, 70, 140)
    # cv2.imshow('v', v)
    # cv2.imshow('s', s)
    # cv2.waitKey(0)
    # exit()

    # bboxes_self = locate_cards_position(img, 44, 1257, 518, 502, 554, False, 170)
    #
    # bboxes_left = locate_cards_position(img, 280, 645, 240, 180, 215, True, 170)
    # bboxes_left += locate_cards_position(img, 280, 645, 310, 245, 281, True, 170)
    #
    # bboxes_right = locate_cards_position(img, 645, 1000, 240, 180, 215, True, 200)
    # bboxes_right += locate_cards_position(img, 645, 1000, 310, 245, 281, True, 200)
    # templates = load_templates()
    # mini_templates = dict()
    # for t in templates:
    #     if t == 'Joker':
    #         mini_templates[t] = cv2.imread('./templates/Joker_mini.png', cv2.IMREAD_GRAYSCALE)
    #     else:
    #         mini_templates[t] = cv2.resize(templates[t], (0, 0), fx=0.7, fy=0.7)
    # for bbox in bboxes_self:
    #     pass
    #     print(parse_card_type(templates, img, bbox), end=', ')
    # print('')
    # for bbox in bboxes_left:
    #     print(parse_card_type(mini_templates, img, bbox), end=', ')
    # print('')
    # for bbox in bboxes_right:
    #     print(parse_card_type(mini_templates, img, bbox), end=', ')

    # img[251, :, :] = 255
    # img[:, 1210, :] = 255
    # show_img(img)
    img = cv2.imread('./debug.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    from timeit import default_timer as timer
    # print(get_current_button_action(img))
    templates = load_templates()
    mini_templates = load_mini_templates(templates)
    print(get_cards_bboxes(img, templates)[0])
    print(get_cards_bboxes(img, mini_templates, 1)[0])
    print(get_cards_bboxes(img, mini_templates, 2)[0])
    # bboxes = locate_cards_position(img, 0, img.shape[1] - 1, 870, 880)[0]

    # print(','.join([parse_card_type(templates, img, bbox) for bbox in bboxes]))

    # img = img[np.mean(img, axis=2) > 230]
    # cv2.imshow('H', img[:, :, 0])
    # cv2.imshow('S', img[:, :, 1])
    # cv2.imshow('V', img[:, :, 2])
    # cv2.waitKey()
