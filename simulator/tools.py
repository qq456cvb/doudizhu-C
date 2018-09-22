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
from simulator.config import Configuration
import time
from PIL import ImageGrab
import win32gui
import skimage.measure
import win32api, win32con
cf = Configuration()

DEBUG = False


def whether_addic(img):
    useful_band = img[cf.addict_top, cf.addict_left:cf.addict_left + cf.addict_width, :]
    for i in range(cf.addict_width):
        compared = useful_band[i]
        tru_color = cf.addict_window[i]
        if not compare_color(tru_color, compared):
            return False
    return True


def locate_cards_position(img, x_left, x_right, y, y_up, y_bottom, mini=False, thresh=210):
    while not (np.all(img[y, x_left] > thresh) and np.all(img[y - 5, x_left] > thresh) and np.max(img[y, x_left]) - np.min(img[y, x_left]) < 15):
        if x_left == img.shape[1] - 1:
            break
        x_left += 1
    while not (np.all(img[y, x_right] > thresh) and np.all(img[y - 5, x_right] > thresh) and np.max(img[y, x_right]) - np.min(img[y, x_right]) < 15):
        if x_right == 0:
            break
        x_right -= 1
    n_cards = int(round((x_right - x_left - (74 if mini else 140)) / (33 if mini else 55.1)) + 1)
    if n_cards == 1:
        spacing = 33 if mini else 55
    else:
        spacing = (x_right - x_left - (74 if mini else 140)) / (n_cards - 1)
    # print(n_cards)
    # # print(np.var(img[730, x_left]), np.var(img[730, x_right]))
    if DEBUG:
        cv2.line(img, (x_left, y_up), (x_left, y_bottom), (0, 255, 0), 3)
        cv2.line(img, (x_right, y_up), (x_right, y_bottom), (0, 0, 255), 3)
        cv2.imshow('test', img)
        cv2.waitKey(0)
    if x_left > x_right or x_right - x_left < (64 if mini else 130):
        return []
    bboxes = []
    for i in range(n_cards):
        left = x_left + int(i * spacing)
        bboxes.append([left, y_up, left + int(spacing), y_bottom])
    # x = x_left + int(13 * scale)
    # while x < x_right - int(13 * scale):
    #     x_cmp = x - int(7 * scale)
    #     if np.all(img[y, x] > 200) and np.all(img[y, x_cmp] > 200) and np.all(img[y, x_cmp - int(3*scale)] > 200):
    #         if np.all(img[y, x].astype(np.int32) - img[y, x_cmp].astype(np.int32) > 20) and np.mean(img[y, x_cmp:x]) > 200:
    #             # cv2.line(img, (x, 700), (x, 800), (0, 255, 0), 1)
    #             # cv2.line(img, (x_cmp, 700), (x_cmp, 800), (0, 0, 255), 1)
    #             pos.append(x)
    #             # print('find one')
    #             x += int(13 * scale)
    #     x += 1
    # pos.insert(0, x_left)
    # bboxes = []
    # # XYXY format
    # for p in pos:
    #     bboxes.append([p, y - int(16 * scale), p + spacing, y + int(36 * scale)])
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
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
    if max_response < 0.65:
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
        cnt += str(parse_card_type(templates, subimg, [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]], binarize=False))
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


def compare_color(truth_color, compared_color, difference=cf.max_pixel_difference):
    """
    judge whether a pixel belongs to a specific color
    :param truth_color: the standard color ([**, **, **])
    :param compared_color: the color to be judged
    :return:
    """
    cnt = 0
    for idx in range(cf.channels):
        if np.abs(int(truth_color[idx]) - int(compared_color[idx])) <= difference:
            cnt += 1
    if cnt == 3:
        return True
    else:
        return False


def find_all_buttons(effective_band, truth_colors):
    """
    find all the buttons
    :param truth_colors: yellow and blue ([**, **, **])
    :param effective_band: the band to be scanned
    :return: a list formed by the start position of all the buttons, if no button is found, return []
    """
    yellow_bbox = []
    blue_bbox = []
    end_bbox = []
    end_continue_bbox = []
    idx = 0
    while idx < cf.img_size[1]:
        if compare_color(truth_colors[0], effective_band[idx]):
            yellow_bbox.append([idx, cf.button_up_margin, idx + cf.two_words_button_width, cf.button_down_margin])
            idx += cf.two_words_button_width
        elif compare_color(truth_colors[1], effective_band[idx]):
            blue_bbox.append([idx, cf.button_up_margin, idx + cf.two_words_button_width, cf.button_down_margin])
            idx += cf.two_words_button_width
        elif compare_color(truth_colors[2], effective_band[idx]):
            end_bbox.append([idx, cf.end_line_y - cf.button_height // 2, idx + cf.end_button_width,
                             cf.end_line_y + cf.button_height // 2])
            idx += cf.end_button_width
        # elif compare_color(truth_colors[3], effective_band[idx]):
        #     end_continue_bbox.append([idx, cf.end_line_y - cf.button_height // 2, idx + cf.end_button_width,
        #                      cf.end_line_y + cf.button_height // 2])
        #     idx += cf.end_button_width
        else:
            idx += 1
    return [yellow_bbox, blue_bbox, end_bbox, end_continue_bbox]


def identify_current_button(effective_band, start_x, standard_array):
    cnt = 0
    temp = effective_band[start_x]
    assert standard_array.shape[0] == cf.two_words_button_width or standard_array.shape[0] == cf.end_button_width
    while cnt < cf.two_words_button_width:
        if not compare_color(standard_array[cnt], effective_band[start_x + cnt]):
            return False
        cnt += 1
    return True


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

def get_current_button_action(current_img):
    """
    get the action represented by the buttons shown in the current image
    :param current_img: the 3 channels pictures
    :return: a dictionary whose keys are all the actions represented by the buttons shown in the picture and whose
    values are the corresponding bounding box of each button
    """
    res_actions = {}
    effective_band = current_img[cf.mid_line, :, :]
    buttons = find_all_buttons(effective_band=effective_band, truth_colors=cf.colors)
    if buttons:
        for specific_color in buttons:    # buttons[0] represents yellow buttons and buttons[1] represents blues buttons
            if specific_color:
                for bbox in specific_color:
                    start_x = bbox[0]     # the start x coordinate value
                    for action in cf.actions:
                        if identify_current_button(
                                effective_band=effective_band,
                                start_x=start_x,
                                standard_array=cf.actions[action]
                        ):
                            res_actions[action] = bbox
                            break
    # non-continuous end
    effective_band = current_img[cf.end_line_y, :, :]
    buttons = find_all_buttons(effective_band=effective_band, truth_colors=cf.colors)
    if buttons:
        for specific_color in buttons:
            if specific_color:
                for bbox in specific_color:
                    start_x = bbox[0]  # the start x coordinate value
                    for action in cf.actions:
                        if identify_current_button(
                                effective_band=effective_band,
                                start_x=start_x,
                                standard_array=cf.actions[action]
                        ):
                            res_actions[action] = bbox
                            break
    # continuous end
    effective_band = current_img[cf.end_continue_line_y, :, :]
    buttons = find_all_buttons(effective_band=effective_band, truth_colors=cf.colors)
    if buttons:
        for specific_color in buttons:
            if specific_color:
                for bbox in specific_color:
                    start_x = bbox[0]  # the start x coordinate value
                    for action in cf.actions:
                        if identify_current_button(
                                effective_band=effective_band,
                                start_x=start_x,
                                standard_array=cf.actions[action]
                        ):
                            res_actions[action] = bbox
                            break
    # whether push a window (chuntian)
    if whether_push_a_window(current_img):
        res_actions['chuntian_window'] = [cf.push_window_left, cf.push_window_top, cf.push_window_left + cf.push_window_width,
                                      cf.push_window_top + cf.push_window_height]
    # whether avoid addiction window
    if whether_addic(current_img):
        res_actions['addict_window'] = [989, 149, 1018, 170]
    return res_actions


def who_is_lord(image):
    """
    judge which player is lord
    :param image: current image
    :return: 0: self is lord, 1: left side player is lord, 2:right side player is lord, -1:no one is lord
    """
    if compare_color(image[cf.self_lord_y, cf.self_lord_x, :], cf.self_lord_color, 0):
        return 0
    elif compare_color(image[cf.left_lord_y, cf.left_lord_x, :], cf.left_lord_color, 0):
        return 1
    elif compare_color(image[cf.right_lord_y, cf.right_lord_x, :], cf.right_lord_color, 0):
        return 2
    else:
        return -1


def whether_push_a_window(img):
    color = img[cf.push_window_top, cf.push_window_left, :]
    if compare_color(cf.push_window_color, color, difference=0):
        return True
    else:
        return False


def get_window_rect(hwnd):
    rect = win32gui.GetWindowRect(hwnd)
    return rect


def is_win(image):
    zone_chosen = image[cf.winning_start_y:cf.winning_start_y + cf.winning_square,
                  cf.winning_start_x:cf.winning_start_x + cf.winning_square, :]
    red_chnnel_sum = np.sum(zone_chosen[:, :, 2])
    blue_channel_sum = np.sum(zone_chosen[:, :, 0])
    if red_chnnel_sum > blue_channel_sum:
        return 1
    else:
        return 0


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
    frame = frame[16:-15, 2:-2, :]
    frame = frame[:, :, [2, 1, 0]]
    cv2.imwrite('test.png', frame)
    # cv2.imwrite(name, frame)
    return frame


def click(x,y, offset=(0, 0)):
    win32api.SetCursorPos((offset[0] + x, offset[1] + y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, offset[0] + x, offset[1] + y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,offset[0] + x, offset[1] + y,0,0)


# get cards and their bboxes, role = 0 for self, 1 for left, 2 for right
def get_cards_bboxes(img, templates, role=0):
    if role == 0:
        bboxes = locate_cards_position(img, 44, 1257, 518, 502, 554, False, 200)
    elif role == 1:
        bboxes = locate_cards_position(img, 280, 645, 240, 180, 215, True, 200)
        bboxes += locate_cards_position(img, 280, 645, 310, 245, 281, True, 200)
    elif role == 2:
        bboxes = locate_cards_position(img, 645, 1000, 240, 180, 215, True, 200)
        bboxes += locate_cards_position(img, 645, 1000, 310, 245, 281, True, 200)
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
    a = A()
    a.set_test()
    a.join()
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
    img = cv2.imread('./photo/end_no.png')
    print(is_win(img))
