import os, sys, time, win32api
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
sys.path.insert(0, '..')
import pyautogui
from simulator.config import ConfigurationOffline
cf_offline = ConfigurationOffline()
from simulator.tools import click, restart_game


def monitor(steps):
    import re
    with open('train_log/DQN-REALDATA/mean_score.log', 'r') as file:
        c = file.read().splitlines()
        i = - 1
        while True:
            if "Start Epoch" in c[i]:
                break
            i -= 1
    assert "Start Epoch" in c[i]
    current_epoch = int(re.findall("Epoch (.*) \.\.\.", c[i])[0])
    # current_epoch = 485
    assert current_epoch > 0
    print("Start monitoring at epoch {}".format(current_epoch))
    next_epoch = current_epoch + steps
    tic = time.time()
    print_flag = True
    while True:
        toc = time.time()
        interval = toc - tic
        if int(interval) % 60 == 0 and print_flag:
            print("Epoch {}/{}/{} runs {} minutes...".format(current_epoch, current_epoch + 1, current_epoch + 2, interval // 60))
            print_flag = False
        if int(interval) % 60 == 1 and not print_flag:
            print_flag = True
        with open('train_log/DQN-REALDATA/mean_score.log', 'r') as file:
            content = file.read()
            if "Start Epoch {}".format(next_epoch) in content:
                tic = time.time()
                restart_game()
                click(cf_offline.start_botton_pos[0], cf_offline.start_botton_pos[1])
                current_epoch = next_epoch
                print("Start monitoring at epoch {}".format(current_epoch))
                next_epoch = current_epoch + steps
            if interval > 60 * 60 * 3:
                # pyautogui.press('esc')
                tic = time.time()
                restart_game()
                click(cf_offline.start_botton_pos[0], cf_offline.start_botton_pos[1])
                print("Restart game at epoch {}".format(current_epoch))


if __name__ == '__main__':
    # start_epoch = int(input("Please type in start epoch: "))
    monitor(3)
    # import re
    # with open('train_log/DQN-REALDATA/mean_score.log', 'r') as file:
    #     c = file.read().splitlines()
    #     i = - 1
    #     while True:
    #         if "Start Epoch" in c[i]:
    #             break
    #         i -= 1
    # t = re.findall("Epoch (.*) \.\.\.", c[i])[0]
    # print(t)
