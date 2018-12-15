# rid 1 = agent 3 (farmer_up)
# rid 2 = agent 1 (lord)
# rid 3 = agent 2 (farmer_down)

import matplotlib.pyplot as plt
import re


def get_log_info(log_info):
    # log_info = {
    #     "epoch": [],
    #     'lord': {
    #         'baseline_wr': [],
    #         'training_wr': [],
    #     },
    #     'farmer_up': {
    #         'baseline_wr': [],
    #         'training_wr': [],
    #     },
    #     'farmer_down': {
    #         'baseline_wr': [],
    #         'training_wr': [],
    #     }
    # }
    with open('./train_log/DQN-60-MA-SELF_PLAY/log.log', 'r') as file:
        content = file.read()
        lines = content.splitlines()
        front_idx = 0
        end_idx = 1
        while True:
            if "Start Epoch" in lines[front_idx]:
                break
            front_idx += 1
        while True:
            if "Start Epoch" in lines[end_idx]:
                break
            end_idx -= 1
    start_epoch = int(re.findall("Epoch (.*) \.\.\.", lines[front_idx])[0])
    end_epoch = int(re.findall("Epoch (.*) \.\.\.", lines[end_idx])[0])
    assert start_epoch <= end_epoch
    for epoch in range(start_epoch, end_epoch + 1):
        try:
            current_paragraph = re.findall(
                "Epoch {}(.*?)param-summary/agent1/dqn_comb/block0/fc/W-rms:".format(epoch), content, re.S
            )[0]
            log_info["lord"]["baseline_wr"].append(
                float(re.findall("\[2\]_lord_win_rate: (.*?)\n", current_paragraph)[0])
            )
            log_info["farmer_up"]["baseline_wr"].append(
                float(re.findall("\[1\]_farmer_win_rate: (.*?)\n", current_paragraph)[0])
            )
            log_info["farmer_down"]["baseline_wr"].append(
                float(re.findall("\[3\]_farmer_win_rate: (.*?)\n", current_paragraph)[0])
            )
            log_info["lord"]["training_wr"].append(
                float(re.findall("lord_win_rate: (.*?)\n", current_paragraph)[3])
            )
            log_info["farmer_up"]["training_wr"].append(
                float(re.findall("farmer_win_rate: (.*?)\n", current_paragraph)[3])
            )
            log_info["farmer_down"]["training_wr"].append(
                float(re.findall("farmer_win_rate: (.*?)\n", current_paragraph)[3])
            )
            log_info["epoch"].append(epoch)
        except:
            pass
    return log_info


def info_verbose(log_info):
    epochs = log_info["epoch"]
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    # baseline
    ax1.plot(epochs, log_info["lord"]["baseline_wr"], label="lord")
    ax1.plot(epochs, log_info["farmer_up"]["baseline_wr"], label="farmer_up")
    ax1.plot(epochs, log_info["farmer_down"]["baseline_wr"], label="farmer_down")
    ax1.legend()
    ax1.set_title("Baseline")
    # training
    ax2.plot(epochs, log_info["lord"]["training_wr"], label="lord")
    ax2.plot(epochs, log_info["farmer_up"]["training_wr"], label="farmer")
    ax2.legend()
    ax2.set_title("Training")
    plt.show()


def dict_save(log_info, filename):
    import json
    js = json.dumps(log_info)
    with open(filename, 'w') as file:
        file.write(js)
    file.close()


def json_load(filename):
    import json
    with open(filename, 'r') as file:
        dic = file.read()
    return dict(json.loads(dic))


if __name__ == '__main__':
    filename = "log_info/log_info.json"
    l = json_load(filename)
    log_info = get_log_info(l)
    # dict_save(log_info, filename)
    info_verbose(log_info)
    # print(log_info)
    # info_verbose(log_info)
