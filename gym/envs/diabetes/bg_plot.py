import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


def bg_plot(bg, save=None):
    # =================
    # Fancy plot!
    # =================
    plt.figure()
    plt.plot(bg, lw=3, color=sns.xkcd_rgb['blue'], label='Mean bg')

    # Plotting standard deviation

    # Hyper - hypo
    bg_low = 72 * np.ones(bg.shape[0])
    bg_high = 180 * np.ones(bg.shape[0])
    plt.axhline(72, c='black', linestyle='--', lw=1)
    plt.axhline(180, c='black', linestyle='--', lw=1)

    plt.ylabel('BG [mg/dl]')
    plt.xlabel('Min')
    plt.ylim(0, 400)

    plt.fill_between(range(1440), bg_low, bg_high, alpha=.20,
                                     facecolor=sns.xkcd_rgb["light orange"], label='Within range')

    # plt.title('Blood glucose levels for all patients')
    plt.legend(fontsize='large')

    if save is not None:
        plt.savefig(save)

    plt.show(block=False)
