# Dou Di Zhu with Combinational Q-Learning
## Dependency
* TensorPack
* TensorFlow
* Conditional
* Pybind11
## Get Started
Create a folder called `build.linux` (`build` if you're using Windows).

Type `cd build; cmake ..; make`. 

Run `TensorPack/MA_Hierarchical_Q/main.py`.
## Directory Structure
* `TensorPack` contain different RL algorithms to train agents
* `experiments` contain scripts to evaluate agents' performance against other baselines
* `simulator` contain scripts to evaluate agents' performance against online gaming platform called "QQ Dou Di Zhu" (we provide it only for academic use, use it at your own risk!)
## Miscellaneous
* We provide a Monte-Carlo-Tree-Search algorithm in https://github.com/qq456cvb/doudizhu-baseline
* We provide a configured Dou Di Zhu mini-server in https://github.com/qq456cvb/doudizhu-tornado
## References
See our paper https://arxiv.org/pdf/1901.08925.pdf
