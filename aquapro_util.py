import numpy as np
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm, norm
from scipy.integrate import quad
from supg.supg.experiments.experiment import run_experiment
from supg.supg.selector import ApproxQuery
from numba import njit
from math import ceil
import seaborn as sns
sns.set_theme(style="ticks")


def nan2mean(dist):
    mean_val = np.nanmean(dist)
    indices = np.where(np.isnan(dist))
    dist[indices] = mean_val


def get_data(filename=None, is_text=False):
    if is_text:
        instance_list = (np.loadtxt(filename) >= 0).astype(int)
    else:
        instance_list = pickle.load(open(filename, 'rb'), encoding='latin1')

    return instance_list


def load_data(name=''):
    if name in ['eICU', 'lyon', 'incor', 'icd9_eICU', 'icd9_mimic']:
        filename_pred = 'data/' + name + '.pred'
        filename_truth = 'data/' + name + '.truth'

        proxy_pred = np.array(get_data(filename=filename_pred))
        oracle_pred = np.array(get_data(filename=filename_truth))

        return proxy_pred, oracle_pred
    elif name in ['voc', 'coco']:
        filename_pred = 'data/' + name + '_output_scores.txt'
        filename_truth = 'data/' + name + '_output_targets.txt'

        proxy_pred = np.array(get_data(filename=filename_pred, is_text=True))
        oracle_pred = np.array(get_data(filename=filename_truth, is_text=True))

        return proxy_pred, oracle_pred
    else:
        filename = './data/' + name
        df = pd.read_csv(filename)

        return np.vstack(np.array(df['proxy_score'])), np.vstack(np.array(df['label']))


def preprocess_dist(oracle, proxy, query):
    if len(oracle[0]) == 1:
        query = np.array([[1]])
        oracle_dist = cdist(query, oracle, metric='cityblock')[0]
        proxy_dist = cdist(query, proxy, metric='cityblock')[0]
    else:
        oracle_dist = cdist(query, oracle, metric='cosine')[0]
        proxy_dist = cdist(query, proxy, metric='cosine')[0]
    nan2mean(proxy_dist)
    nan2mean(oracle_dist)

    return proxy_dist, oracle_dist


@njit
def preprocess_ranks(proxy_dist):
    rank2pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
    ranks = np.array([i[0] for i in rank2pd])

    return ranks


def preprocess_topk_phi(proxy_dist, norm_scale, t):
    norm_cdfs = norm.cdf(x=t, loc=proxy_dist, scale=norm_scale)
    topk2phi = sorted(enumerate(norm_cdfs), key=lambda x: x[1], reverse=True)
    topk = np.array([i[0] for i in topk2phi])
    phi = np.array([i[1] for i in topk2phi])

    return topk, phi


def preprocess_sync(proxy_dist, norm_scale):
    sync_oracle = np.clip([scipy.stats.norm.rvs(loc=_, scale=norm_scale) for _ in proxy_dist], a_min=0,
                          a_max=1)

    return sync_oracle


def plot_statics(oracle_dist, proxy_dist, sync_oracle, t, norm_scale, f):
    sort_pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
    pd_rank = [i[0] for i in sort_pd]
    idx = [i for i in range(len(pd_rank))]
    precis_list = list()
    recall_list = list()
    sync_recalls = list()
    sync_preciss = list()
    true_pos = 0
    all_pos = len(np.where(oracle_dist <= t)[0])
    sync_pos = len(np.where(sync_oracle <= t)[0])
    sync_tp = 0
    for j in range(len(pd_rank)):
        if oracle_dist[pd_rank[j]] <= t:
            true_pos += 1
        if sync_oracle[pd_rank[j]] <= t:
            sync_tp += 1
        precis_list.append(true_pos / (j + 1))
        recall_list.append(true_pos / all_pos)
        sync_recalls.append(sync_tp / sync_pos)
        sync_preciss.append(sync_tp / (j + 1))
    plt.xlabel('objects')
    plt.title('precis/recall and proxy_dist, sigma=%.2f (%s)' % (norm_scale, f))
    plt.axhline(y=t, color='k')
    # plt.plot(idx, pd_val, label='proxy_dist')
    plt.plot(idx, precis_list, label='precis')
    plt.plot(idx, recall_list, label='recall')
    # plt.plot(idx, sync_recalls, label='sync_recall')
    # plt.plot(idx, sync_preciss, label='sync_precis')
    plt.legend()
    plt.show()


def SUPG(oracle_dist, proxy_dist, t, primary_target, p, cost, query_type):
    data = pd.DataFrame({'oracle': oracle_dist, 'proxy': proxy_dist})
    data = data.sort_values('proxy', axis=0, ascending=True).reset_index()
    data['oracle'] = (data['oracle'] <= t).astype(int)
    data['proxy'] = 1 - data['proxy']
    data = data.rename(columns={'index': 'id', 'oracle': 'label', 'proxy': 'proxy_score'})

    if query_type == 'RT':
        exp_spec = {"source": 'outside',
                    "sampler": "ImportanceSampler",
                    'estimator': 'None',
                    "query": ApproxQuery(
                        qtype="rt", min_recall=primary_target,
                        min_precision=-1, delta=1 - p,
                        budget=cost),
                    "selector": "ImportanceRecall",
                    "num_trials": 1}
    elif query_type == 'PT':
        exp_spec = {"source": 'outside',
                    'sampler': 'ImportanceSampler',
                    'estimator': 'None',
                    'query': ApproxQuery(
                        qtype='pt', min_precision=primary_target,
                        min_recall=-1, delta=1 - p,
                        budget=cost),
                    'selector': 'ImportancePrecisionTwoStageSelector',
                    "num_trials": 1}
    else:
        print('unknown query type:', query_type)
        exp_spec = {}

    try:
        precision, recall, prob_s, na_rate = run_experiment(cur_experiment=exp_spec, df=data)
    except ValueError:
        return 0, 0, 0, 0

    return precision, recall, prob_s, na_rate


@njit
def draw_sample_s_m(D, best_s, best_m):
    samples = list()
    if best_m == -1:
        return np.arange(D)
    elif best_s == 1:
        sample = np.random.choice(D, size=best_m, replace=True)
        samples.extend(sample)
    else:
        for _ in range(best_m):
            sample = np.random.choice(D, size=best_s, replace=False)
            samples.extend(sample)
    # find the sorted unique samples
    samples = np.unique(np.array(samples))

    return samples


@njit
def compute_optm(s, p, D, K):
    if K == 0:
        print('K==0')
        return -1

    denom = np.sum(np.array([np.log((D - s - i) / (D - i)) for i in range(K)]))

    return ceil(np.log(1 - p) / denom)


@njit
def array_union(l1, l2):
    return np.unique(np.concatenate((l1, l2)))


@njit
def set_diff(l1, l2):
    l3 = np.array([i for i in l1 if i not in l2])
    return np.unique(l3)


@njit
def find_K_pt(ranks, oracle_dist, t, pt):
    K = 0
    true_pos = 0
    for i in range(len(ranks)):
        if oracle_dist[ranks[i]] <= t:
            true_pos += 1
            if true_pos / (i + 1) >= pt:
                K += 1
            else:
                break
    return K


@njit
def find_best_sm(D, K, mode, prob):
    best_s = 0
    best_m = 0
    best_cost = D

    if mode == 'exact':
        for s in range(1, D - K + 1):
            m = compute_optm(s, prob, D, K)
            cost = D * (1 - (1 - s / D) ** m)
            if cost < best_cost:
                best_s = s
                best_m = m
                best_cost = cost
    elif mode == 'approx_s1':
        best_s = 1
        best_m = compute_optm(best_s, prob, D, K)
    elif mode == 'approx_m1':
        best_m = 1
        denom = np.sum(np.array([1 / (D - i) for i in range(K)]))
        best_s = min(D, ceil(-np.log(1 - prob) / denom))
        # best_s = min(D, ceil(D * (1 - (1 - prob) ** (1 / K))))
    else:
        print('unknown mode', mode)

    return best_s, best_m


def baseline_topk_phi_i(proxy_dist, norm_scale, sk, sp):
    f_sk = 1 - norm.cdf(x=sk, loc=proxy_dist, scale=norm_scale)
    f_sp = 1 - norm.cdf(x=sp, loc=proxy_dist, scale=norm_scale)

    f_sp = np.clip(f_sp, a_min=1e-8, a_max=None)

    return (1 - f_sk) / f_sp


def baseline_topk_pi(proxy_dist, norm_scale, s):
    f = 1 - norm.cdf(x=s, loc=proxy_dist, scale=norm_scale)

    return np.prod(f)


def baseline_topk_topc_tableu(oracle_dist, table_c, k):
    table_all = np.arange(len(oracle_dist))
    k2v_c = sorted([(_, oracle_dist[int(_)]) for _ in table_c], key=lambda x: x[1])
    topk_c = [int(_[0]) for _ in k2v_c[:k]]
    table_u = np.setdiff1d(table_all, table_c)

    return topk_c, table_u


def baseline_topk_xf(sp, sk, p_d, short_dist, norm_scale):
    def intergrand(x):
        xf = np.prod(1 - norm.cdf(x=x, loc=short_dist, scale=norm_scale))
        return norm.pdf(x, loc=p_d, scale=norm_scale) * xf

    intgrl = quad(intergrand, sp, sk)[0]

    term_3 = norm.cdf(x=sp, loc=p_d, scale=norm_scale) * np.prod(1 - norm.cdf(x=sp, loc=short_dist, scale=norm_scale))

    return intgrl + term_3


