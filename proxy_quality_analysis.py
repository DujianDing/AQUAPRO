from aquapro_util import load_data, preprocess_dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_proxy_quality(oracle, proxy, index, f=''):
    diff_dist = list()
    archive_oracle = list()
    archive_proxy = list()
    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        archive_oracle += list(oracle_dist)
        archive_proxy += list(proxy_dist)
        diff_dist += list(np.array(oracle_dist) - np.array(proxy_dist))

    font_size_hp = 18
    label_fs = 20
    plt.rcParams['font.size'] = font_size_hp
    plt.hist(diff_dist, bins=20, density=True)
    # plt.title(r'Histogram of $\epsilon=dist^O(x)-dist^P(x)$', fontsize=label_fs)
    # plt.title('%s' % f, fontsize=label_fs)
    plt.xlabel(r'$\epsilon_i=dist^O(x_i)-dist^P(x_i)$', fontsize=label_fs)
    plt.ylabel('Frequency (%)', fontsize=label_fs)
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=font_size_hp)
    plt.yticks(fontsize=font_size_hp)
    plt.tight_layout()
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(5, 4)
    plt.show()


def plot_coreset(oracle, proxy, index, f, t=0.9):
    precis_arrary = np.zeros(len(oracle))
    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        sort_pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
        pd_rank = [i[0] for i in sort_pd]
        true_pos = 0
        for j in range(len(pd_rank)):
            if oracle_dist[pd_rank[j]] <= t:
                true_pos += 1
            precis_arrary[j] += true_pos / (j + 1)

    idx = [i for i in range(len(oracle))]
    plt.plot(idx, precis_arrary / len(index))

    font_size_hp = 22
    label_fs = 28
    plt.rcParams['font.size'] = font_size_hp

    plt.title('%s' % f, fontsize=label_fs)
    # plt.title(r'Precision($D_k$)', fontsize=label_fs)
    plt.xlabel(r'$k$', fontsize=label_fs)
    plt.ylabel('Precision($D_k$)', fontsize=label_fs)
    plt.xticks(fontsize=font_size_hp)
    plt.yticks(fontsize=font_size_hp)
    if f == 'night-street':
        plt.xticks([0, 5000, 10000])
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
    else:
        plt.xticks([0, 2000, 4000])
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(7, 4)
    plt.show()


if __name__ == '__main__':
    num_query = 100
    f_map = {'icd9_mimic': 'Mimic-III', 'jackson10000.csv':'night-street'}
    for Fname in ['voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco'
        Proxy, Oracle = load_data(name=Fname)

        if Fname == 'coco':
            np.random.seed(1)
            Samples = np.random.choice(len(Oracle), size=8000, replace=False)
            Oracle = Oracle[Samples]
            Proxy = Proxy[Samples]
            Fname = Fname + '8000'

        np.random.seed(0)
        Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)

        plot_proxy_quality(oracle=Oracle, proxy=Proxy, index=Index)
        # plot_coreset(oracle=Oracle, proxy=Proxy, index=Index, f=f_map[Fname])






