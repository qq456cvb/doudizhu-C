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


def info_verbose(log_info, e_epoch=None, path=None):
    from scipy.ndimage.filters import gaussian_filter1d
    epochs = log_info["epoch"]
    end_epoch = epochs[-1] if not e_epoch else e_epoch
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    trans = 0.3
    # baseline
    ax1.plot(epochs[:end_epoch], log_info["lord"]["baseline_wr"][:end_epoch], alpha=trans, color='r')
    sm = gaussian_filter1d(log_info["lord"]["baseline_wr"][:end_epoch], sigma=10)
    ax1.plot(epochs[:end_epoch], sm, label="lord", color='r')

    ax1.plot(epochs[:end_epoch], log_info["farmer_up"]["baseline_wr"][:end_epoch], alpha=trans, color='g')
    sm = gaussian_filter1d(log_info["farmer_up"]["baseline_wr"][:end_epoch], sigma=10)
    ax1.plot(epochs[:end_epoch], sm, label="farmer_up", color='g')

    ax1.plot(epochs[:end_epoch], log_info["farmer_down"]["baseline_wr"][:end_epoch], alpha=trans, color='b')
    sm = gaussian_filter1d(log_info["farmer_down"]["baseline_wr"][:end_epoch], sigma=10)
    ax1.plot(epochs[:end_epoch], sm, label="farmer_down", color='b')

    ax1.legend(loc=4)
    ax1.set_ylim([0, 0.8])
    ax1.set_title("Baseline")
    ax1.set_ylabel("Winning Rate", rotation='horizontal')
    ax1.yaxis.set_label_coords(-0.025, 1.05)

    # training
    ax2.plot(epochs[:end_epoch], log_info["lord"]["training_wr"][:end_epoch], alpha=trans, color='c')
    sm = gaussian_filter1d(log_info["lord"]["training_wr"][:end_epoch], sigma=10)
    ax2.plot(epochs[:end_epoch], sm, color='c', label='lord')

    ax2.plot(epochs[:end_epoch], log_info["farmer_up"]["training_wr"][:end_epoch], alpha=trans, color='m')
    sm = gaussian_filter1d(log_info["farmer_up"]["training_wr"][:end_epoch], sigma=10)
    ax2.plot(epochs[:end_epoch], sm, color='m', label="farmer")
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.set_title("Training")
    ax2.set_xlabel("Epoch")
    ax2.xaxis.set_label_coords(1.05, -0.025)

    plt.show()
    if path:
        f.savefig(path)


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
    info_verbose(log_info, 200, 'multi-agent.png')
    # print(log_info)
    # info_verbose(log_info)
