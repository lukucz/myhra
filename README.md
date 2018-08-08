# My Implementation of Hybrid Reward Architecture
これは、NIPS2017で発表された論文 "Hybrid Reward Architecture for Reinforcement Learning" で提案されたモデルである Hybrid Reward Architecture (HRA) を、授業の課題として実装したものです。

* https://arxiv.org/abs/1706.04208

この論文中の "Experiment 4.1: Fruit Collection Task" と同様の実験を行えるように実装しました。HRAとDQNは自分で実装しましたが、ゲームの環境については再現性を損なわないようにオリジナルの公開コードをベースに、少し変更を加えて作成しました。

* https://github.com/Maluuba/hra

# Dependencies
本コードを実行させるのに必要な環境とライブラリの一覧です。

* Python 3.5 or higher
* numpy (pip install numpy)
* click (pip install click)
* pyyaml (pip install pyyaml)
* [TensorFlow](https://www.tensorflow.org/) 1.8+

# Usage
DQNは、HRAとの比較用に用意しました。

HRAは、もっとも単純な実装である`hra`と、ドメイン知識を活用した`hra+1`と`hra+2`を用意しました。

* DQN:
```
python ./dqn/train.py
```

* HRA:
```
python ./hra/train.py --mode hra
```
* `--mode` can be either of `hra`, `hra+1`, or `hra+2`.
