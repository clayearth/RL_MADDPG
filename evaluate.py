import argparse
import torch
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run(config):
    global torch_actions
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

    if config.n_episodes == 1:
        e_pos = [[] for i in range(1)]
        p_pos = [[] for i in range(2)]
        e_vel = [[] for i in range(1)]
        p_vel = [[] for i in range(2)]
        e_reward = [[] for i in range(1)]
        p_reward = [[] for i in range(2)]
    ts = [[],[]]
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
            # print(actions) # 三个智能体模型给出的动作决策，整体模型的输出控制量，(action[i][0] + 1.5 )*2为控制量速度v, action[i][1]*pi/2为控制量omega
            obs, rewards, dones, infos = env.step(actions)
            # print(obs) # 三个智能体的观测结果，整体模型的输入控制量，每个智能体均为[自身速度v,theta + 自身角速度omega + 与基地的相对距离x,y + 其他智能体的相对距离和他们的速度x,y,v,theta]

            if config.n_episodes == 1:
                for i in range(infos["n_adv"]):
                    e_pos[i].append(deepcopy(infos["adversary"][0][i]))
                    e_vel[i].append(deepcopy(infos["adversary"][1][i]))
                    e_reward[i].append(deepcopy(infos["adversary"][2][i]))
                for i in range(infos["n_good"]):
                    p_pos[i].append(deepcopy(infos["goodagent"][0][i]))
                    p_vel[i].append(deepcopy(infos["goodagent"][1][i]))
                    p_reward[i].append(deepcopy(infos["goodagent"][2][i]))

            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            if config.n_episodes == 1:
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human')

            # judge hit(break) condition
            if -1 in dones:
                bad_hit += 1
                ts[1].append(t_i)
                print("protection was hit by adversary!")
                break
            elif 1 in dones:
                good_hit += 1
                print("good agents hit adversary!")
                ts[0].append(t_i)
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
    # print("Episodes: %d, good hit: %d, bad hit: %d, good hit percent: %.3f, good ave step:  %.1f, bad ave step:  %.1f"
    #       %(config.n_episodes, good_hit, bad_hit, good_hit*1.0/config.n_episodes, np.mean(np.array(ts[0])),np.mean(np.array(ts[1]))))
    print("Episodes: %d, good hit: %d, bad hit: %d, good hit percent: %.3f"
          %(config.n_episodes, good_hit, bad_hit, good_hit*1.0/config.n_episodes ))

    # from msvcrt import getch
    # getch()
    #
    if config.n_episodes == 1:
        plotpics(e_pos,p_pos,e_vel,p_vel,e_reward,p_reward)

    env.close()

def plotpics(e_pos,p_pos,e_vel,p_vel,e_reward,p_reward):
    e_pos_arr = np.array(e_pos)
    p_pos_arr = np.array(p_pos)
    e_vel_arr = np.array(e_vel)
    p_vel_arr = np.array(p_vel)
    e_reward_arr = np.array(e_reward)
    p_reward_arr = np.array(p_reward)

    # i is e_nums; j is p_nums
    dis_arr = np.array([[[np.sqrt(np.sum(np.square(p_pos_arr[j][k] - e_pos_arr[i][k])))
                          for k in range(len(e_pos_arr[0]))]
                         for j in range(2)]
                        for i in range(1)])

    dis_theta_arr = np.array([[[math.atan2((e_pos_arr[i][k] - p_pos_arr[j][k])[1],
                                           (e_pos_arr[i][k] - p_pos_arr[j][k])[0]) * 180/math.pi
                                for k in range(len(e_pos_arr[0]))]
                               for j in range(2)]
                              for i in range(1)])

    lips = len(e_pos_arr[0])
    step_list = list(range(lips))

    plt.figure(1) # draw move trace
    for i in range(1): # plot pos of e
        plt.plot(e_pos_arr[i][0:lips,0],e_pos_arr[i][0:lips,1],label=('e_'+str(i)))
    for i in range(2): # plot pos of p
        plt.plot(p_pos_arr[i][0:lips,0],p_pos_arr[i][0:lips,1],label=('p_'+str(i)))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend(['p','e'])

    plt.figure(2) # draw distance between e & p
    for i in range(1):
        for j in range(2):
            plt.plot(step_list, dis_arr[i][j],label=("p"+str(j)+"_e"+str(i)))
    # plt.plot(step_list,state_array[0:len(step_list),0])
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('dis')

    plt.figure(3) # draw thetas
    line_shape = ['-','--','-.']
    for i in range(1):
        # plt.plot(step_list, np.array([math.atan2(e_vel_arr[i][m][1],e_vel_arr[i][m][0])*180/math.pi for m in step_list]), ls='-', label=("e"+str(i)+"-vel-theta"))
        for j in range(2):
            if i==0:
                plt.plot(step_list, np.array([p_vel_arr[j][m][1]*180/math.pi for m in step_list]), ls = line_shape[j], label=("p"+str(j)+"-vel-theta"))
            plt.plot(step_list, dis_theta_arr[i][j], ls = line_shape[j], label=("p"+str(j)+"_e"+str(i)+"_dis_theta"))
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('theta')

    plt.figure(4) # darw v of e & p
    for i in range(1): # plot vel of e
        plt.plot(step_list, np.array([e_vel_arr[i][m][0] for m in step_list]), label=('v_e_'+str(i)))
    for i in range(2): # plot vel of p
        plt.plot(step_list, np.array([p_vel_arr[i][m][0] for m in step_list]), label=('v_p_'+str(i)))
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('v')

    plt.figure(5)
    for i in range(1): # plot reward of e
        plt.plot(step_list[0:lips-1], np.array([e_reward_arr[i][m] for m in step_list][0:lips-1]), label=('reward_e_'+str(i)))
    for i in range(2): # plot reward of p
        plt.plot(step_list[0:lips-1], np.array([p_reward_arr[i][m] for m in step_list][0:lips-1]), label=('reward_p_'+str(i)))
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.show()


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
    parser.add_argument("--n_episodes", default=100, type=int) # 测试轮数
    parser.add_argument("--episode_length", default=800, type=int) #每轮测试步长
    parser.add_argument("--fps", default=60, type=int) # 帧数

    config = parser.parse_args()

    run(config)