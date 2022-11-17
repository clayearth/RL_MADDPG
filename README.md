# RL_MADDPG
A Project by MADDPG

Changed base on the respository:[shariqiqbal2810/maddpg-pytorch](https://github.com/clayearth/maddpg-pytorch)

PyTorch Implementation of MADDPG from [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (Lowe et. al. 2017)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```


## Not Implemented

There are a few items from the paper that have not been implemented in this repo

* Ensemble Training
* Inferring other agents' policies
* Mixed continuous/discrete action spaces

## Acknowledgements

The OpenAI baselines [Tensorflow implementation](https://github.com/openai/baselines/tree/master/baselines/ddpg) and Ilya Kostrikov's [Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf) of DDPG were used as references. After the majority of this codebase was complete, OpenAI released their [code](https://github.com/openai/maddpg) for MADDPG, and I made some tweaks to this repo to reflect some of the details in their implementation (e.g. gradient norm clipping and policy regularization).