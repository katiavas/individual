import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        # running_avg[i] = scores[i]
    plt.plot(x, running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)


def plot_learning_curve1(x, scores, scores2,  figure_file):
    running_avg = np.zeros(len(scores))
    running_avg2 = np.zeros(len(scores2))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        running_avg2[i] = np.mean(scores2[max(0, i-100):(i+1)])
        # running_avg[i] = scores[i]
    plt.plot(x, running_avg)
    plt.plot(x, running_avg2)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)

def plot_intrinsic_reward(x, intrinsic_reward, figure_file):
    intrinsic = np.zeros(len(intrinsic_reward))
    for i in range(len(intrinsic)):
        # intrinsic[i] = np.mean(intrinsic_reward[max(0, i-100):(i+1)])
        intrinsic[i] = intrinsic_reward[i]
    plt.plot(x, intrinsic)
    plt.xlabel("Number of episodes")
    plt.ylabel("Intrinsic reward")
    plt.title('Plot of intrinsic reward when ICM is on')
    plt.savefig(figure_file)


def plot_intrinsic_reward_avg(x, intrinsic_reward, figure_file):
    intrinsic = np.zeros(len(intrinsic_reward))
    for i in range(len(intrinsic)):
        intrinsic[i] = np.mean(intrinsic_reward[max(0, i-100):(i+1)])
    plt.plot(x, intrinsic)
    plt.xlabel("Number of episodes")
    plt.ylabel("Intrinsic reward")
    plt.title('Plot average of 100 previous episodes intrinsic reward when ICM is on')
    plt.savefig(figure_file)


def plot_learning_curve_with_shaded_error(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    std = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        std[i] = np.std(np.mean(scores[max(0, i-100):(i+1)]))
        # running_avg[i] = scores[i]

    plt.plot(x, running_avg)
    plt.fill_between(x, running_avg - std, running_avg + std,
                     color='blue', alpha=0.2)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)