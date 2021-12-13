import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    # count all tests
    good_hit = 0
    bad_hit = 0
    out_of_episode_length = 0

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')

            # judge hit(break) condition
            if -1 in dones:
                bad_hit += 1
                print("protection was hit by adversary!")
                break
            elif 1 in dones:
                good_hit += 1
                print("good agents hit adversary!")
                break
            elif t_i == config.episode_length-1:
                out_of_episode_length += 1
                print("out of episode_length!")
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
    print("-------------------------------------------------------------------")
    print("Episodes: %d, good hit: %d, bad hit: %d, good hit percent: %.2f"
          %(config.n_episodes, good_hit, bad_hit, good_hit*1.0/config.n_episodes))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment") # Name of environment
    parser.add_argument("model_name",
                        help="Name of model") # Name of model
    parser.add_argument("run_num", default=1, type=int) # run_x
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory") # Saves gif of each episode into model directory
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy") # 如果采用某一次中间策略而不是最终策略
    parser.add_argument("--n_episodes", default=1, type=int) # 测试轮数
    parser.add_argument("--episode_length", default=100, type=int) #每轮测试步长
    parser.add_argument("--fps", default=30, type=int) # 帧数

    config = parser.parse_args()

    run(config)