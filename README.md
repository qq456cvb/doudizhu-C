# Dou Di Zhu with Combinationatorial Q-Learning
# Accepted to AIIDE 2020
## Step by step training tutorial
1. Clone the repo
``` 
git clone https://github.com/qq456cvb/doudizhu-C.git
```
2. Change work directory to root
``` 
cd doudizhu-C
```
3. Create env from environment.yml
``` 
conda env create -f environment.yml
```
4. Activate env
```
conda activate doudizhu
```
5. Build C++ files
```
mkdir build
cd build
cmake ..
make
```
6. Have fun training!
```
cd TensorPack/MA_Hierarchical_Q
python main.py
```

## Evaluation against other baselines
1. Download pretrained model from https://jbox.sjtu.edu.cn/l/L04d4A or [GoogleDrive](https://drive.google.com/drive/folders/1YTNR5JYNgNfQpQ9DwhQ3ClcyeXKn_17b?usp=sharing), then put it into `pretrained_model`
2. Build Monte-Carlo baseline and move the lib into root
```
git clone https://github.com/qq456cvb/doudizhu-baseline.git
cd doudizhu-baseline/doudizhu
mkdir build
cd build
cmake ..
make
mv mct.cpython-36m-x86_64-linux-gnu.so [doudizhu-C ROOT]
```
3. Run evaluation scripts in `scripts`
```
cd scripts
python experiments.py
```
## Directory Structure
* `TensorPack` contain different RL algorithms to train agents
* `experiments` contain scripts to evaluate agents' performance against other baselines
* `simulator` contain scripts to evaluate agents' performance against online gaming platform called "QQ Dou Di Zhu" (we provide it for academic use only, use it at your own risk!)
## Miscellaneous
* We provide a Monte-Carlo-Tree-Search algorithm in https://github.com/qq456cvb/doudizhu-baseline
* We provide a configured Dou Di Zhu mini-server in https://github.com/qq456cvb/doudizhu-tornado for you to play interactively. NOTE you should build the server and load pretrained model by yourself! Tutorial coming soon!
* If you meet any problems, open an issue.

## DouZero
Recently, another algorithm called DouZero (https://github.com/kwai/DouZero) has been proposed, to whom may be interested in a strong DouDizhu AI. It is also an actively maintained open-source project.

## References
See our paper https://arxiv.org/pdf/1901.08925.pdf. If you find this algorithm useful or use part of its code in your projects, please consider cite
```
@inproceedings{you2020combinatorial,
  title={Combinatorial Q-Learning for Dou Di Zhu},
  author={You, Yang and Li, Liangwei and Guo, Baisong and Wang, Weiming and Lu, Cewu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  volume={16},
  number={1},
  pages={301--307},
  year={2020}
}
```
