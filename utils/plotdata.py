import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data

# 平滑处理，类似tensorboard的smoothing函数。
def smooth1(data, x='timestep', y='reward', weight=0.75):

    # data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed
    # save = pd.DataFrame({x: data[x].values, y: smoothed})
    # save.to_csv(save_path + 'smooth_'+ file_name)


def poltdata(env_id, model_name, run_dir):
    df = []
    label = ['escapee1', 'pursuer1', 'pursuer2']
    for i in range(3):
        file = run_dir + "/agent_data/" + os.path.join("agent_" + str(i) + "_reward_" + env_id + model_name + ".pkl")
        with open(file, "rb") as f:
            df.append(pickle.load(f))
        df[i]['agent'] = label[i]

    for i in range(3):
        df[i]["reward"] = smooth1(df[i], "episode", "reward")

    plt.figure(1)
    sns.set(style="darkgrid", font_scale=1)

    df = pd.concat(df)
    df.index = range(len(df))

    sns.lineplot(x="episode", y="reward", hue="agent", style="agent", data=df)

    plt.ylabel("Reward")
    plt.xlabel("Time Steps(1e5)")
    plt.title("MADDPG")

    plt.savefig(run_dir + "/train_reward_plot.svg")
    plt.show()


if __name__ == '__main__':
    env_id = "myenv_experiment"
    model_name = "model_1110"
    run_num = "1"
    run_dir = "D:/Repository/RL_MADDPG/models/" + env_id + "/" + model_name + "/run" + run_num
    poltdata(env_id, model_name, run_dir)
