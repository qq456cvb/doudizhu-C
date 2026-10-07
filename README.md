# Combinatorial Q-Learning for Dou Di Zhu (AIIDE 2020)

A deep reinforcement learning agent for **Dou Di Zhu** (斗地主, "Fight the Landlord"), the popular Chinese three-player card game. The game's challenge for RL is its **combinatorial action space**: at every step a player must choose among a huge number of card combinations. We propose **Combinatorial Q-Learning (CQL)**, which handles this with a two-stage network — a card-group decomposition stage followed by a move proposal stage — together with order-invariant max-pooling to capture relationships between primitive actions. All agents are trained adversarially from scratch with only knowledge of the game rules, and play at a level comparable to human players.

[Paper (AIIDE 2020 proceedings)](https://ojs.aaai.org/index.php/AIIDE/article/view/7445) | [Preprint](https://arxiv.org/pdf/1901.08925.pdf)

The repository ships a fast **C++ game engine** (move generation, hand decomposition via dancing links, full game logic) exposed to Python through **pybind11**, plus [Tensorpack](https://github.com/tensorpack/tensorpack)-based training pipelines for CQL and several baselines.

## Installation

1. Clone the repo and create the conda environment (Python 3.6, TensorFlow 1.13, Tensorpack 0.8.5):

```bash
git clone https://github.com/qq456cvb/doudizhu-C.git
cd doudizhu-C
conda env create -f environment.yml
conda activate doudizhu
```

2. Build the C++ game environment (requires CMake and pybind11):

```bash
mkdir build
cd build
cmake ..
make
```

This produces the `env` Python module that the training and evaluation scripts import.

## Training

Train the multi-agent combinatorial Q-learning agents (the main algorithm from the paper):

```bash
cd algorithms/cql
python train.py
```

Three agents (landlord and two peasants) are trained adversarially through self-play, each with its own experience replay; training progress is evaluated periodically against rule-based and random baselines.

## Pretrained Models

Download the pretrained checkpoints from [Hugging Face](https://huggingface.co/qq456cvb/doudizhu-C), [SJTU jBox](https://jbox.sjtu.edu.cn/l/L04d4A) or [Google Drive](https://drive.google.com/drive/folders/1YTNR5JYNgNfQpQ9DwhQ3ClcyeXKn_17b?usp=sharing), and put them into `pretrained_model`:

```bash
hf download qq456cvb/doudizhu-C --local-dir pretrained_model
```

## Evaluation

1. Build the Monte-Carlo baseline from [doudizhu-baseline](https://github.com/qq456cvb/doudizhu-baseline) and move the resulting library into the repo root:

```bash
git clone https://github.com/qq456cvb/doudizhu-baseline.git
cd doudizhu-baseline/doudizhu
mkdir build && cd build
cmake ..
make
mv mct.cpython-36m-x86_64-linux-gnu.so [doudizhu-C ROOT]
```

2. Run the evaluation scripts, which play the trained CDQN agent against random and rule-based (RHCP/MCTS) baselines in every seat assignment:

```bash
cd evaluation
python experiments.py
```

## Directory Structure

- `engine/` — C++ game engine (`card.*`, `game.*`, `dancing_links.*`) and the pybind11 `env` module (`env.cpp`).
- `doudizhu/` — shared Python game logic: card encodings and the action space (`card.py`), legal-move masks and helpers (`utils.py`), and a pure-Python game environment (`env.py`).
- `algorithms/` — Tensorpack training code, one package per method, each with a `train.py` entry point:
  - `cql/` — multi-agent combinatorial Q-learning (the paper's method).
  - `hierarchical_q/`, `vanilla_dqn/` — single-agent hierarchical and naive DQN baselines.
  - `a3c/`, `a3c_fc/` — A3C baselines.
  - `policy_sl/`, `value_sl/` — supervised policy/value pretraining.
  - `autoencoder/` — card-group embedding autoencoder; `encoding.npy` holds the embeddings used by `cql`.
- `evaluation/` — evaluation of agents against the baselines.
- `simulator/` — scripts to play against the online platform "QQ Dou Di Zhu" (provided for academic use only; use at your own risk!).

## Related Projects

- [doudizhu-baseline](https://github.com/qq456cvb/doudizhu-baseline) — Monte-Carlo-Tree-Search baseline for Dou Di Zhu.
- [doudizhu-tornado](https://github.com/qq456cvb/doudizhu-tornado) — a web mini-server to play against the agents interactively (build the server and load the pretrained model yourself).
- [DouZero](https://github.com/kwai/DouZero) — a more recent, actively maintained strong Dou Di Zhu AI, for those interested.

## Citation

If you find this algorithm useful or use part of its code in your projects, please consider citing:

```bibtex
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
