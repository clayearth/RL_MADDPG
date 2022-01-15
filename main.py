import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False # torch.cuda.is_available()
# print(USE_CUDA)

#生成并行训练环境
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action = discrete_action) #返回一个scenario_name为env_id的环境，discrete_action离散动作
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000) #每次seed一次保证环境相同
            return env
        return init_env
    if n_rollout_threads == 1: #只有一个环境则返回一个环境
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)]) #返回多个环境

def run(config):
    global torch_agent_actions
    model_dir = Path('./models') / config.env_id / config.model_name #模型地址
    if not model_dir.exists(): #模型不存在则自动生成‘run1’
        curr_run = 'run1'
    else:# 如果模型‘run(X)’存在，则自动生成‘run(X+1)’
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs' #日志
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed) # Sets the seed for generating random numbers. Returns a `torch.Generator` object.
    np.random.seed(config.seed)

    # NOT CUDA
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # 生成并行训练环境
    print(config.discrete_action)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    # MADDPG模型
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)

    # Replay Buffer for multi-agent RL with parallel rollouts
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor

        maddpg.prep_rollouts(device='cpu')
        # maddpg.prep_rollouts(device = 'gpu')

        # 探索度
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps

        # 标度噪声 Scale noise for each agent
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        # 如果discrete_action=False，为连续动作空间，则每次重置噪声self.exploration = OUNoise.reset()
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            # 对每个智能体a，返回Tensor[a_x∈[-1,1],a_y∈[-1,1]]
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # print(torch_agent_actions)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            next_obs, rewards, dones, infos = env.step(actions)

            # ifi = 1 / 30  # inter-frame interval
            # calc_end = time.time()
            # elapsed = calc_end - calc_start
            # if elapsed < ifi:
            #     time.sleep(ifi - elapsed)
            # env.envs[0].render('human')

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones) # 缓冲区
            obs = next_obs # 更新obs

            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')

                for u_i in range(config.n_rollout_threads): # 并行
                    for a_i in range(maddpg.nagents): # 每个智能体
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

            # judge hit(break) condition
            if 1 in dones or -1 in dones:
                break

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)

        # 记录日志
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        # 阶段性存储模型
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()

    # 记录总结日志
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    EPOSODE = 50000
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment") # Name of environment
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents") # Name of directory to store model/training contents
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed") # 随机种子
    parser.add_argument("--n_rollout_threads", default=18, type=int) # 并行训练环境数 1
    parser.add_argument("--n_training_threads", default=20, type=int) # CPU线程数 6
    parser.add_argument("--buffer_length", default=int(1*1e6), type=int) # 缓冲器大小 1e6
    parser.add_argument("--n_episodes", default=EPOSODE, type=int) # 总训练轮数，初始 25000
    parser.add_argument("--episode_length", default=1200, type=int) # 单次训练数据组数 25
    parser.add_argument("--steps_per_update", default=500, type=int) # 网络每组训练步长 100
    parser.add_argument("--batch_size", # Batch size for model training 1024
                        default=4096, type=int,
                        help="Batch size for model training")
    # 探索量，计算探索度=max(0,n_exploration_eps-now_episodes)/n_exploration_eps，随着训练从1线性减小至0，初始25000
    parser.add_argument("--n_exploration_eps", default=EPOSODE, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float) # 初始化噪声量度,默认为0.3
    parser.add_argument("--final_noise_scale", default=0.0, type=float) # 最后噪声量度，二者与探索度一起决定最终噪声量
    parser.add_argument("--save_interval", default=1000, type=int) # 阶段存储参数
    parser.add_argument("--hidden_dim", default=64, type=int) # 隐藏层数目
    parser.add_argument("--lr", default=0.01, type=float) # 学习率0.01
    parser.add_argument("--tau", default=0.02, type=float)
    parser.add_argument("--agent_alg", # 智能体算法
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", # 对手/逃逸者算法
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    # parser.add_argument("--discrete_action", # 离散动作空间，触发时为真(离散)，不触发时为假(连续)
    #                     default=False)
    parser.add_argument("--discrete_action", # 离散动作空间，触发时为真(离散)，不触发时为假(连续)
                        action='store_true')

    config = parser.parse_args()

    run(config)
