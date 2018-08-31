from simulator.tools import *
import ctypes
import threading
from simulator.predictor import Predictor
from card import Card

toggle = False


def esc_pressed():
    global toggle
    toggle = not toggle
    print("Escape was pressed.")


class Simulator:
    class State:
        PRE_START = 0
        CALLING = 1
        PLAYING = 2
        END = 3

    def __init__(self):
        self.templates = load_templates()
        self.mini_templates = dict()
        for t in self.templates:
            if t == 'Joker':
                self.mini_templates[t] = cv2.imread('./templates/Joker_mini.png', cv2.IMREAD_GRAYSCALE)
            else:
                self.mini_templates[t] = cv2.resize(self.templates[t], (0, 0), fx=0.7, fy=0.7)
        self.tiny_templates = load_tiny_templates()
        # self.grab_screen = grab_screen()

        self.window_rect = get_window_rect()
        self.current_screen = None
        self.state = Simulator.State.CALLING
        self.predictor = Predictor()
        # for compatibility use the order, self, right, left
        self.history = [[], [], []]

    def spin_lock_on_button(self):
        act = dict()
        while not act:
            time.sleep(0.2)
            act = get_current_button_action(self.current_screen)
        return act

    def click(self, bbox):
        click((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2, (self.window_rect[0], self.window_rect[1]))

    def main_loop(self):
        global toggle
        while True:
            if not toggle:
                time.sleep(0.2)
                continue
            self.current_screen = grab_screen()
            act = self.spin_lock_on_button()
            if self.state == Simulator.State.CALLING:
                # state has changed
                if 'chupai' in act or 'alone_chupai' in act or 'yaobuqi' in act:
                    self.state = Simulator.State.PLAYING
                    continue
                if 'qiangdizhu' in act:
                    self.click(act['buqiang'])
                else:
                    assert 'jiaodizhu' in act
                    self.click(act['bujiao'])
            elif self.state == Simulator.State.PLAYING:
                left_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 1)
                right_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 2)
                self.history[1].extend(right_cards)
                self.history[2].extend(left_cards)
                if 'yaobuqi' in act:
                    self.click(act['yaobuqi'])
                else:
                    last_cards = left_cards
                    if not left_cards:
                        assert len(right_cards) > 0
                        last_cards = right_cards
                    total_cards = np.ones([60])
                    remain_cards = total_cards - Card.char2onehot60(self.history[0] + self.history[1] + self.history[2])
                    handcards, bboxes = get_cards_bboxes(self.current_screen, self.templates, 0)
                    print('current handcards: ', handcards)
                    left_cnt, right_cnt = get_opponent_cnts(self.current_screen, self.tiny_templates)
                    print('left cnt: ', left_cnt, 'right cnt: ', right_cnt)
                    # to be the same as C++ side, right comes before left

                    right_prob_state = remain_cards * (right_cnt / (left_cnt + right_cnt))
                    left_prob_state = remain_cards * (left_cnt / (left_cnt + right_cnt))
                    prob_state = np.concatenate([right_prob_state, left_prob_state])
                    assert prob_state.size == 60
                    intention = self.predictor.predict(handcards, last_cards, prob_state)
                    self.history[0].extend(intention)
                    print('intention is: ', intention)
                    intention.sort(key=lambda k: Card.cards_to_value[k], reverse=True)
                    if len(intention) == 0:
                        self.click(act['buchu'])
                    else:
                        i = 0
                        j = 0
                        to_click = []
                        while j < len(intention):
                            if handcards[i] == intention[j]:
                                to_click.append(bboxes[i])
                                i += 1
                                j += 1
                            else:
                                i += 1
                        for bbox in to_click:
                            self.click(bbox)
                            time.sleep(0.1)
                        self.click(act['chupai'])
            time.sleep(1.)


def hook():
    ctypes.windll.user32.RegisterHotKey(None, 1, 0, win32con.VK_ESCAPE)

    try:
        msg = ctypes.wintypes.MSG()
        while ctypes.windll.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == win32con.WM_HOTKEY:
                esc_pressed()
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageA(ctypes.byref(msg))
    finally:
        ctypes.windll.user32.UnregisterHotKey(None, 1)


if __name__ == '__main__':
    t = threading.Thread(target=hook, args=())
    t.start()
    sim = Simulator()
    sim.main_loop()
    t.join()


