from math import floor, ceil, sqrt
from collections import defaultdict

from aquapro_util import load_data, SUPG, preprocess_dist, preprocess_ranks, preprocess_topk_phi, preprocess_sync
from aquapro_util import draw_sample_s_m, compute_optm, array_union, find_K_pt, set_diff, find_best_sm
from aquapro_util import baseline_topk_phi_i, baseline_topk_pi, baseline_topk_topc_tableu, baseline_topk_xf
from numba import njit
from hyper_parameter import norm_scale, eps, pilot_size, pilot_eps, std_offset, repeat
import numpy as np
import pickle
from pathlib import Path

import time


@njit
def test_PQA_PT(oracle_dist, phi, topk, t=0.9, prob=0.9, pt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, 0

    pbs = np.zeros(len(phi) + 1)
    k_star = 0

    for i in range(1, len(phi) + 1):
        if i == 1:
            pbs[0] = 1 - phi[0]
            pbs[1] = phi[0]
        else:
            shift_pbs = np.roll(pbs, 1) * phi[i - 1]
            pbs = pbs * (1 - phi[i - 1]) + shift_pbs

        idx_s = ceil(i * pt)
        precis_prob = np.sum(pbs[idx_s:i + 1])

        if precis_prob >= prob:
            k_star = i

    if k_star == 0:
        return 1, 0, 0

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star


@njit
def test_PQA_RT(oracle_dist, phi, topk, t=0.9, prob=0.9, rt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, len(oracle_dist)

    L = 1
    R = len(phi)

    def pb_distribution(phii, p):
        for j in range(1, len(phii) + 1):
            if j == 1:
                p[0] = 1 - phii[0]
                p[1] = phii[0]
            else:
                shift_p = np.roll(p, 1) * phii[j - 1]
                p = p * (1 - phii[j - 1]) + shift_p

        return p

    while L < R:
        mid = floor((L + R) / 2)

        pbs = pb_distribution(phi[:mid], np.zeros(len(phi) + 1))
        pbc = pb_distribution(phi[mid:], np.zeros(len(phi) + 1))

        recall_prob = 0
        for i in range(mid + 1):
            cdf = np.sum(pbc[:floor((1 - rt) * i / rt) + 1])
            recall_prob += pbs[i] * cdf

        if recall_prob < prob:
            L = mid + 1
        else:
            R = mid

    k_star = L
    max_exp = 0
    pbs = np.zeros(len(phi) + 1)

    for i in range(L, len(phi) + 1):
        if i == L:
            pbs = pb_distribution(phi[:L], np.zeros(len(phi) + 1))
        else:
            shift_pbs = np.roll(pbs, 1) * phi[i - 1]
            pbs = pbs * (1 - phi[i - 1]) + shift_pbs

        exp_precis = np.sum(np.array([pbs[j] * j / i for j in range(i + 1)]))
        if exp_precis >= max_exp:
            k_star = i
            max_exp = exp_precis

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star


def test_PQE_PT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, pt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    precision, recall, _ = test_PQA_PT(oracle_dist, phi, topk, t=t, prob=prob, pt=pt, pilots=samples)

    return precision, recall, _


def test_PQE_RT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, rt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    precision, recall, _ = test_PQA_RT(oracle_dist, phi, topk, t=t, prob=prob, rt=rt, pilots=samples)

    return precision, recall, _


@njit
def test_CSC_PT(oracle_dist, ranks, t=0.9, prob=0.9, pt=0.9, mode='approx_m1', K=None):
    true_ans = np.where(oracle_dist <= t)[0]
    D = len(oracle_dist)

    if K is None:
        K = 0
        true_pos = 0
        for i in range(len(ranks)):
            if oracle_dist[ranks[i]] <= t:
                true_pos += 1
                if true_pos / (i + 1) >= pt:
                    K += 1
                else:
                    break

    if K == 0:
        samples = list()
        counter = 0
        while True:
            sample = draw_sample_s_m(D, min(D, pilot_size * 2 ** counter), 1)
            sample_true = sample[np.where(oracle_dist[ranks[sample]] <= t)[0]]
            samples.extend(sample)
            counter += 1
            if len(sample_true) > 0:
                break
        samples = np.unique(np.array(samples))
        true_pos = 0
        k_star = 0

        for j in range(len(samples)):
            if oracle_dist[ranks[samples[j]]] <= t:
                true_pos += 1
            if true_pos / (j + 1) >= pt:
                k_star = samples[j]

        hoeff_num = ceil(-np.log(1 - prob) / (2 * pilot_eps ** 2))
        hoeff_samples = np.random.choice(k_star + 1, size=hoeff_num, replace=True)
        hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
        hoeff_false = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] > t)[0]]
        precis_lb = len(hoeff_true) / hoeff_num - pilot_eps

        samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
        samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

        union_samples = array_union(ranks[hoeff_samples], ranks[samples])
        union_false = array_union(ranks[hoeff_false], ranks[samples_false])
        union_true = array_union(ranks[hoeff_true], ranks[samples_true])

        if precis_lb >= pt:
            ans = set_diff(array_union(ranks[:k_star + 1], union_samples), union_false)
        else:
            ans = union_true

        true_pos = len(np.intersect1d(ans, true_ans))
        precision = true_pos / len(ans)
        recall = true_pos / len(true_ans)

        cost = len(array_union(samples, hoeff_samples))

        return precision, recall, cost, 1, cost, ans, array_union(samples, hoeff_samples), k_star

    best_s, best_m = find_best_sm(D=D, K=K, mode=mode, prob=prob)

    samples = draw_sample_s_m(D, best_s, best_m)
    samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
    samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

    if len(samples_true) == 0:
        k_star = np.random.choice(len(oracle_dist))
    else:
        k_star = samples[np.min(np.where(oracle_dist[ranks[samples]] <= t)[0])]

    ans = set_diff(array_union(ranks[:k_star + 1], ranks[samples]), ranks[samples_false])

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)
    cost = len(samples)

    return precision, recall, best_s, best_m, cost, ans, samples, k_star


@njit
def test_CSC_RT(oracle_dist, ranks, t=0.9, prob=0.9, rt=0.9, mode='approx_m1', K=None, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    D = len(oracle_dist)

    if K is None:
        K = floor(len(true_ans) * (1 - rt)) + 1

    best_s, best_m = find_best_sm(D=D, K=K, mode=mode, prob=prob)

    samples = draw_sample_s_m(D, best_s, best_m)
    samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
    samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

    if len(samples_true) == 0:
        k_star = np.random.choice(len(oracle_dist))
    else:
        k_star = samples[np.max(np.where(oracle_dist[ranks[samples]] <= t)[0])]

    if pilots is None:
        union_samples = ranks[samples]
        union_false = ranks[samples_false]
        cost = len(samples)
    else:
        union_samples = array_union(ranks[samples], ranks[pilots])
        pilots_false = pilots[np.where(oracle_dist[ranks[pilots]] > t)[0]]
        union_false = array_union(ranks[samples_false], ranks[pilots_false])
        cost = len(array_union(samples, pilots))

    ans = set_diff(array_union(ranks[:k_star + 1], union_samples), union_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, best_s, best_m, cost


def test_CSE_PT(oracle_dist, ranks, t=0.9, prob=0.9, pt=0.9, mode='approx_m1'):
    true_ans = np.where(oracle_dist <= t)[0]
    presample = np.unique(np.random.choice(len(oracle_dist), size=pilot_size, replace=False))
    presample_true = presample[np.where(oracle_dist[ranks[presample]] <= t)[0]]
    presample_false = presample[np.where(oracle_dist[ranks[presample]] > t)[0]]

    K_pt = 0
    true_pos = 0
    for i in range(len(presample)):
        if oracle_dist[ranks[presample[i]]] <= t:
            true_pos += 1
            if true_pos / (i + 1) >= pt:
                K_pt += 1
            else:
                break
    K_pt = ceil(K_pt * len(oracle_dist) / pilot_size)

    k_hat = 0
    true_pos = 0
    for i in range(len(presample)):
        if oracle_dist[ranks[presample[i]]] <= t:
            true_pos += 1
            if true_pos / (i + 1) >= pt:
                k_hat = presample[i]

    _, _, _, _, _, ans, samples, k_star = test_CSC_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode=mode, K=K_pt)

    ans = set_diff(array_union(ans, ranks[presample]), ranks[presample_false])
    cost = len(array_union(ranks[presample], ranks[samples]))

    if K_pt > 0:
        k_star = max(k_star, k_hat)
        hoeff_num = ceil(-np.log(1 - prob) / (2 * pilot_eps ** 2))
        hoeff_samples = np.random.choice(k_star + 1, size=hoeff_num, replace=True)
        hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
        hoeff_false = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] > t)[0]]
        precis_lb = len(hoeff_true) / hoeff_num - pilot_eps

        samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
        samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

        union_samples = array_union(ranks[hoeff_samples], array_union(ranks[presample], ranks[samples]))
        union_true = array_union(ranks[hoeff_true], array_union(ranks[presample_true], ranks[samples_true]))
        union_false = array_union(ranks[hoeff_false], array_union(ranks[presample_false], ranks[samples_false]))

        if precis_lb >= pt:
            ans = set_diff(array_union(ranks[:k_star + 1], union_samples), union_false)
        else:
            ans = union_true

        cost = len(array_union(ranks[hoeff_samples], array_union(ranks[presample], ranks[samples])))

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, cost


def test_CSE_RT(oracle_dist, ranks, delta, t=0.9, prob=0.9, rt=0.9, mode='approx_m1'):
    hoeff_num = ceil(-np.log(delta) / (2 * eps ** 2))
    hoeff_samples = np.random.choice(len(oracle_dist), size=hoeff_num, replace=True)
    hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
    A_lb = max(1, floor((len(hoeff_true) / hoeff_num - eps) * len(oracle_dist)))
    K_rt = floor(A_lb * (1 - rt)) + 1

    precision, recall, _, _, cost = test_CSC_RT(oracle_dist, ranks, t=t, prob=prob, rt=rt, mode=mode,
                                                K=K_rt, pilots=np.unique(hoeff_samples))

    return precision, recall, cost


def test_sample2test_PT(oracle_dist, ranks, bd, t=0.9, pt=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    inc_true_pos = 0
    k_star = 0

    sample = sorted(np.random.choice(len(oracle_dist), size=bd, replace=False))

    for j in range(len(sample)):
        if oracle_dist[ranks[sample[j]]] <= t:
            inc_true_pos += 1
        if inc_true_pos / (j + 1) >= pt:
            k_star = sample[j]

    ans = ranks[:k_star + 1]

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, bd


def test_sample2test_RT(oracle_dist, ranks, bd, t=0.9, rt=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    inc_true_pos = 0
    k_star = 0

    sample = np.random.choice(len(oracle_dist), size=bd, replace=False)
    sample_true_pos = len(sample[np.where(oracle_dist[ranks[sample]] <= t)[0]])
    sample = sorted(sample)

    if sample_true_pos == 0:
        ans = []
    else:
        for j in range(len(sample)):
            if oracle_dist[ranks[sample[j]]] <= t:
                inc_true_pos += 1
            if inc_true_pos / sample_true_pos >= rt:
                k_star = sample[j]
                break
        ans = ranks[:k_star + 1]

    true_pos = len(np.intersect1d(ans, true_ans))
    if len(ans) == 0:
        precision = 1
    else:
        precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, bd


def test_topk(oracle_dist, proxy_dist, scale, t=0.9, prob=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    k = len(true_ans)
    table_c = np.random.choice(len(oracle_dist), size=k, replace=False)
    topk_c, table_u = baseline_topk_topc_tableu(oracle_dist=oracle_dist, table_c=table_c, k=k)

    sk = sorted(oracle_dist[topk_c])[-1]
    sp = sorted(oracle_dist[topk_c])[-2]
    pi = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sk)

    while pi < prob and len(table_c) < len(oracle_dist):
        phi_all = baseline_topk_phi_i(proxy_dist=proxy_dist[table_u], norm_scale=scale, sk=sk, sp=sp)
        k2phi = sorted(np.stack([table_u, phi_all], axis=-1), key=lambda x: x[1], reverse=True)
        gamma = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sp)
        max_delta = 0
        max_indx = None
        for idx, idx_phi in k2phi:
            if max_delta > gamma * idx_phi:
                break
            table_u_short = np.setdiff1d(table_u, idx)
            delta = baseline_topk_xf(sp=sp, sk=sk, p_d=proxy_dist[int(idx)], short_dist=proxy_dist[table_u_short], norm_scale=scale)
            if delta > max_delta:
                max_delta = delta
                max_indx = idx

        if max_indx is not None:
            table_c = np.append(table_c, max_indx)
        else:
            new_sample_size = ceil(len(table_u) / 2)
            new_sample = np.random.choice(table_u, size=new_sample_size, replace=False)
            table_c = np.append(table_c, new_sample)

        topk_c, table_u = baseline_topk_topc_tableu(oracle_dist=oracle_dist, table_c=table_c, k=k)

        sk = sorted(oracle_dist[topk_c])[-1]
        sp = sorted(oracle_dist[topk_c])[-2]
        pi = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sk)

    ans = topk_c

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)
    print(precision, recall, len(table_c))

    return precision, recall, len(table_c)


def exp_PQA_maximal_CR(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=False):
    rt_k_success = defaultdict(list)
    rt_k_precis = defaultdict(list)
    pt_k_success = defaultdict(list)
    pt_k_recall = defaultdict(list)
    scale_list = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])

    for i in range(len(index)):
        proxy_dist, _ = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=norm_scale, t=t)
        sync_oracle = preprocess_sync(proxy_dist, norm_scale)
        true_ans = np.where(sync_oracle <= t)[0]

        # PQA-PT-sync
        _, _, k_star = test_PQA_PT(sync_oracle, phi, topk, t=t, prob=prob, pt=pt)

        for j in scale_list:
            if k_star == 0:
                pt_k_success[j].append(1)
                pt_k_recall[j].append(0)
            else:
                k_hat = ceil(k_star * (1 + j / 100))
                ans = topk[:k_hat]

                true_pos = len(np.intersect1d(ans, true_ans))
                precision = true_pos / len(ans)
                recall = true_pos / len(true_ans)

                pt_k_success[j].append(int(precision >= pt))
                pt_k_recall[j].append(recall)

        # PQA-RT-sync
        _, _, k_star = test_PQA_RT(sync_oracle, phi, topk, t=t, prob=prob, rt=rt)

        for j in scale_list:
            k_hat = int(k_star * (1 + j / 100))
            ans = topk[:k_hat]

            true_pos = len(np.intersect1d(ans, true_ans))
            precision = true_pos / len(ans)
            recall = true_pos / len(true_ans)

            rt_k_success[j].append(int(recall >= rt))
            rt_k_precis[j].append(precision)

    if not is_precompile:
        backup_res = [scale_list, pt_k_recall, pt_k_success, rt_k_precis, rt_k_success, prob]
        Path("./results/PQA/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/PQA/' + fname + '.pkl', 'wb'))


def exp_CSC_minimal_cost(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=False):
    rt_success = defaultdict(list)
    rt_cost = defaultdict(list)
    pt_success = defaultdict(list)
    pt_cost = defaultdict(list)
    mode_list = ['exact', 'approx_s1', 'approx_m1', 'rand_s', 'rand_sm']

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)
        true_ans = np.where(oracle_dist <= t)[0]
        K_rt = floor(len(true_ans) * (1 - rt)) + 1
        K_pt = find_K_pt(ranks, oracle_dist, t=t, pt=pt)

        def random_start(m, p, o_dist, r, K, s_hat, m_hat, query_type):
            if m == 'exact':
                pass
            elif m == 'approx_s1':
                s_hat = 1
                m_hat = compute_optm(s=s_hat, p=p, D=len(o_dist), K=K)
            elif m == 'approx_m1':
                m_hat = 1
                denom = np.sum(np.array([1 / (len(o_dist) - _) for _ in range(K)]))
                s_hat = min(len(o_dist), ceil(-np.log(1 - prob) / denom))
                # s_hat = min(len(o_dist), ceil(len(o_dist) * (1 - (1 - p) ** (1 / K))))
            elif m == 'rand_s':
                s_hat = np.random.choice(len(o_dist) - K, size=1)[0] + 1
                m_hat = compute_optm(s=s_hat, p=p, D=len(o_dist), K=K)
            elif m == 'rand_sm':
                s_hat = np.random.choice(len(o_dist) - K, size=1)[0] + 1
                m_hat = np.random.choice(100, size=1)[0] + 1

            samples = draw_sample_s_m(len(o_dist), s_hat, m_hat)
            samples_true = samples[np.where(o_dist[r[samples]] <= t)[0]]
            samples_false = samples[np.where(o_dist[r[samples]] > t)[0]]

            if len(samples_true) == 0:
                k_star = np.random.choice(len(o_dist))
            elif query_type == 'RT':
                k_star = samples[np.max(np.where(o_dist[r[samples]] <= t)[0])]
            elif query_type == 'PT':
                k_star = samples[np.min(np.where(o_dist[r[samples]] <= t)[0])]
            else:
                k_star = -1
                print('Invalid query type:', query_type)

            ans = set_diff(array_union(r[:k_star + 1], r[samples]), r[samples_false])

            true_pos = len(np.intersect1d(ans, true_ans))
            rec = true_pos / len(true_ans)
            prc = true_pos / len(ans)

            if query_type == 'RT':
                return int(rec >= rt), len(samples)
            elif query_type == 'PT':
                return int(prc >= pt), len(samples)

        # CSC-RT
        _, _, best_s, best_m, cost = test_CSC_RT(oracle_dist, ranks, t=t, prob=prob, rt=rt, mode='exact', K=K_rt)

        for _ in range(repeat):
            for mode in mode_list:
                success, cost = random_start(mode, prob, oracle_dist, ranks, K_rt, best_s, best_m, query_type='RT')
                rt_success[mode].append(success)
                rt_cost[mode].append(cost)

        # CSC-PT
        if K_pt == 0:
            for _ in range(repeat):
                precision, _, _, _, cost, _, _, _ = test_CSC_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='exact',
                                                                K=0)
                pt_success['K0'].append(int(precision >= pt))
                pt_cost['K0'].append(cost)
        else:
            _, _, best_s, best_m, cost, _, _, _ = test_CSC_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='exact',
                                                              K=K_pt)
            for _ in range(repeat):
                for mode in mode_list:
                    success, cost = random_start(mode, prob, oracle_dist, ranks, K_pt, best_s, best_m, query_type='PT')
                    pt_success[mode].append(success)
                    pt_cost[mode].append(cost)

    if not is_precompile:
        backup_res = [mode_list, pt_cost, pt_success, rt_cost, rt_success, prob]
        Path("./results/CSC/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/CSC/' + fname + '.pkl', 'wb'))


def exp_CSC_cpu_overhead(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname=''):
    rt_overhead = defaultdict(list)
    pt_overhead = defaultdict(list)
    mode_list = ['exact', 'approx_s1', 'approx_m1']

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)
        true_ans = np.where(oracle_dist <= t)[0]
        K_rt = floor(len(true_ans) * (1 - rt)) + 1
        K_pt = find_K_pt(ranks, oracle_dist, t=t, pt=pt)

        for _ in range(repeat):
            for mode in mode_list:
                start_time = time.time()
                _, _ = find_best_sm(D=len(oracle_dist), K=K_rt, mode=mode, prob=prob)
                end_time = time.time()
                rt_overhead[mode].append(end_time - start_time)

                if K_pt > 0:
                    start_time = time.time()
                    _, _ = find_best_sm(D=len(oracle_dist), K=K_pt, mode=mode, prob=prob)
                    end_time = time.time()
                    pt_overhead[mode].append(end_time - start_time)
    backup_res = [mode_list, rt_overhead, pt_overhead]
    Path("./results/overhead/").mkdir(parents=True, exist_ok=True)
    pickle.dump(backup_res, open('results/overhead/' + fname + '.pkl', 'wb'))


def exp_comprehensive_comparison(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname=''):
    delta = 1 - sqrt(prob)
    rt_prob = prob / (1 - delta)
    scale_list = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
    CSE_PT_stats = defaultdict(list)
    CSE_RT_stats = defaultdict(list)

    PQE_PT = dict()
    PQE_RT = dict()
    SUPG_PT = dict()
    SUPG_RT = dict()

    for scale in scale_list:
        PQE_PT[scale] = defaultdict(list)
        PQE_RT[scale] = defaultdict(list)
        SUPG_PT[scale] = defaultdict(list)
        SUPG_RT[scale] = defaultdict(list)

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)

        for _ in range(repeat):
            # C-PT
            CSE_PT_start = time.time()
            precision, recall, PT_cost = test_CSE_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='approx_m1')
            CSE_PT_end = time.time()
            CSE_PT_stats['success'].append(int(precision >= pt))
            CSE_PT_stats['CR'].append(recall)
            CSE_PT_stats['cost'].append(PT_cost)
            CSE_PT_stats['overhead'].append(CSE_PT_end - CSE_PT_start)

        avg_PT_cost = np.mean(CSE_PT_stats['cost'][i * repeat:(i + 1) * repeat])

        for scale in scale_list:
            cost = min(ceil(avg_PT_cost * (1 + scale / 100)), len(oracle_dist))
            for _ in range(repeat):
                # SUPG-PT
                SUPG_PT_start = time.time()
                try:
                    precision, recall, _, _ = SUPG(oracle_dist, proxy_dist, t, pt, prob, cost=cost, query_type='PT')
                except ZeroDivisionError:
                    precision = recall = 0
                SUPG_PT_end = time.time()
                SUPG_PT[scale]['success'].append(int(precision >= pt))
                SUPG_PT[scale]['CR'].append(recall)
                SUPG_PT[scale]['cost'].append(cost)
                SUPG_PT[scale]['overhead'].append(SUPG_PT_end - SUPG_PT_start)

                # PQE-PT
                PQE_PT_start = time.time()
                precision, recall, _ = test_PQE_PT(oracle_dist, proxy_dist, bd=cost, t=t, prob=prob, pt=pt)
                PQE_PT_end = time.time()
                PQE_PT[scale]['success'].append(int(precision >= pt))
                PQE_PT[scale]['CR'].append(recall)
                PQE_PT[scale]['cost'].append(cost)
                PQE_PT[scale]['overhead'].append(PQE_PT_end - PQE_PT_start)

        # --------------------------------------------------------------------

        for _ in range(repeat):
            # CSE-RT
            CSE_RT_start = time.time()
            precision, recall, RT_cost = test_CSE_RT(oracle_dist, ranks, t=t, delta=delta, prob=rt_prob, rt=rt,
                                                     mode='approx_m1')
            CSE_RT_end = time.time()
            CSE_RT_stats['success'].append(int(recall >= rt))
            CSE_RT_stats['CR'].append(precision)
            CSE_RT_stats['cost'].append(RT_cost)
            CSE_RT_stats['overhead'].append(CSE_RT_end - CSE_RT_start)

        avg_RT_cost = np.mean(CSE_RT_stats['cost'][i * repeat:(i + 1) * repeat])

        for scale in scale_list:
            cost = min(ceil(avg_RT_cost * (1 + scale / 100)), len(oracle_dist))
            # SUPG-RT
            SUPG_RT_start = time.time()
            try:
                precision, recall, _, _ = SUPG(oracle_dist, proxy_dist, t, rt, prob, cost=cost, query_type='RT')
            except ZeroDivisionError:
                precision = recall = 0
            SUPG_RT_end = time.time()
            SUPG_RT[scale]['success'].append(int(recall >= rt))
            SUPG_RT[scale]['CR'].append(precision)
            SUPG_RT[scale]['cost'].append(cost)
            SUPG_RT[scale]['overhead'].append(SUPG_RT_end - SUPG_RT_start)

            # PQE-RT
            PQE_RT_start = time.time()
            precision, recall, _ = test_PQE_RT(oracle_dist, proxy_dist, bd=cost, t=t, prob=prob, rt=rt)
            PQE_RT_end = time.time()
            PQE_RT[scale]['success'].append(int(recall >= rt))
            PQE_RT[scale]['CR'].append(precision)
            PQE_RT[scale]['cost'].append(cost)
            PQE_RT[scale]['overhead'].append(PQE_RT_end - PQE_RT_start)

    method_list = ['PQE', 'CSE', 'SUPG']
    backup_res = [method_list, scale_list, PQE_PT, CSE_PT_stats, SUPG_PT,
                  PQE_RT, CSE_RT_stats, SUPG_RT, prob]
    Path("./results/CMPR/").mkdir(parents=True, exist_ok=True)
    pickle.dump(backup_res, open('results/CMPR/' + fname + '.pkl', 'wb'))


def exp_runningtime(oracle, proxy, index, is_sample2test, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname=''):
    delta = 1 - sqrt(prob)
    rt_prob = prob / (1 - delta)
    CSE_PT_stats = defaultdict(list)
    CSE_RT_stats = defaultdict(list)

    PQE_PT_stats = defaultdict(list)
    PQE_RT_stats = defaultdict(list)
    SUPG_PT_stats = defaultdict(list)
    SUPG_RT_stats = defaultdict(list)

    sample2test_PT_stats = defaultdict(list)
    sample2test_RT_stats = defaultdict(list)

    def extend_stats(old_stat, new_stat):
        for metric in ['success', 'CR', 'cost', 'overhead']:
            old_stat[metric].extend(new_stat[metric])

    def binary_search(method, q_type, cr_target):
        L = 1
        R = len(oracle_dist)
        r = defaultdict(list)
        while L < R:
            r = defaultdict(list)

            cost = floor((L + R) / 2)
            for _ in range(repeat):
                start_time = time.time()
                if method == 'SUPG' and q_type == 'PT':
                    try:
                        prec, recl, _, _ = SUPG(oracle_dist, proxy_dist, t, pt, prob, cost=cost, query_type='PT')
                    except ZeroDivisionError:
                        prec = recl = 0
                elif method == 'SUPG' and q_type == 'RT':
                    try:
                        prec, recl, _, _ = SUPG(oracle_dist, proxy_dist, t, rt, prob, cost=cost, query_type='RT')
                    except ZeroDivisionError:
                        prec = recl = 0
                elif method == 'PQE' and q_type == 'PT':
                    prec, recl, _ = test_PQE_PT(oracle_dist, proxy_dist, bd=cost, t=t, prob=prob, pt=pt)
                elif method == 'PQE' and q_type == 'RT':
                    prec, recl, _ = test_PQE_RT(oracle_dist, proxy_dist, bd=cost, t=t, prob=prob, rt=rt)
                elif method == 'sample2test' and q_type == 'PT':
                    sample2test_ranks = preprocess_ranks(proxy_dist)
                    prec, recl, _ = test_sample2test_PT(oracle_dist, sample2test_ranks, bd=cost, t=t, pt=pt)
                elif method == 'sample2test' and q_type == 'RT':
                    sample2test_ranks = preprocess_ranks(proxy_dist)
                    prec, recl, _ = test_sample2test_RT(oracle_dist, sample2test_ranks, bd=cost, t=t, rt=rt)
                else:
                    prec = recl = -1
                    print('unknown method and query type', method, q_type)

                end_time = time.time()
                if q_type == 'PT':
                    r['success'].append(int(prec >= pt))
                    r['CR'].append(recl)
                else:
                    r['success'].append(int(recl >= rt))
                    r['CR'].append(prec)
                r['cost'].append(cost)
                r['overhead'].append(end_time - start_time)

            avg_cr = np.mean(r['CR'])
            avg_success = np.mean(r['success'])
            if avg_success >= prob and avg_cr > cr_target:
                R = cost
            else:
                L = cost + 1

        return r

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)

        for _ in range(repeat):
            # CSE-PT
            CSE_PT_start = time.time()
            precision, recall, PT_cost = test_CSE_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='approx_m1')
            CSE_PT_end = time.time()
            CSE_PT_stats['success'].append(int(precision >= pt))
            CSE_PT_stats['CR'].append(recall)
            CSE_PT_stats['cost'].append(PT_cost)
            CSE_PT_stats['overhead'].append(CSE_PT_end - CSE_PT_start)

            # CSE-RT
            CSE_RT_start = time.time()
            precision, recall, RT_cost = test_CSE_RT(oracle_dist, ranks, t=t, delta=delta, prob=rt_prob, rt=rt,
                                                     mode='approx_m1')
            CSE_RT_end = time.time()
            CSE_RT_stats['success'].append(int(recall >= rt))
            CSE_RT_stats['CR'].append(precision)
            CSE_RT_stats['cost'].append(RT_cost)
            CSE_RT_stats['overhead'].append(CSE_RT_end - CSE_RT_start)

        avg_CSE_PT_CR = np.mean(CSE_PT_stats['CR'][i * repeat:(i + 1) * repeat])
        avg_CSE_RT_CR = np.mean(CSE_RT_stats['CR'][i * repeat:(i + 1) * repeat])

        if not is_sample2test:
            # find comparable SUPG-PT
            res = binary_search(method='SUPG', q_type='PT', cr_target=avg_CSE_PT_CR)
            extend_stats(SUPG_PT_stats, res)

            # find comparable SUPG-RT
            res = binary_search(method='SUPG', q_type='RT', cr_target=avg_CSE_RT_CR)
            extend_stats(SUPG_RT_stats, res)

            # find comparable PQE-PT
            res = binary_search(method='PQE', q_type='PT', cr_target=avg_CSE_PT_CR)
            extend_stats(PQE_PT_stats, res)

            # find comparable PQE-RT
            res = binary_search(method='PQE', q_type='RT', cr_target=avg_CSE_RT_CR)
            extend_stats(PQE_RT_stats, res)
        else:
            # find comparable sample2test-PT
            res = binary_search(method='sample2test', q_type='PT', cr_target=avg_CSE_PT_CR)
            extend_stats(sample2test_PT_stats, res)

        # find comparable sample2test-RT
        res = binary_search(method='sample2test', q_type='RT', cr_target=avg_CSE_RT_CR)
        extend_stats(sample2test_RT_stats, res)

    price_l = np.arange(10) / 100

    if is_sample2test:
        backup_res = [price_l, sample2test_PT_stats, sample2test_RT_stats, prob]
        Path("./results/runningtime_sample2test/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/runningtime_sample2test/' + fname + '.pkl', 'wb'))
    else:
        backup_res = [price_l, PQE_PT_stats, CSE_PT_stats, SUPG_PT_stats,
                      PQE_RT_stats, CSE_RT_stats, SUPG_RT_stats, prob]
        Path("./results/runningtime/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/runningtime/' + fname + '.pkl', 'wb'))


def exp_scalability_test(is_sample2test, pt=0.9, rt=0.9, t=0.9, prob=0.9,fname='', n_query=50):
    proxy, oracle = load_data(name=fname)
    scale_list = (np.arange(4) + 1) / 4
    subset_sizes = [int(scale * len(oracle)) for scale in scale_list]
    delta = 1 - sqrt(prob)
    rt_prob = prob / (1 - delta)

    def init_stats():
        stats = dict()
        for _ in subset_sizes:
            stats[_] = defaultdict(list)

        return stats

    PQE_PT_stats = init_stats()
    CSE_PT_stats = init_stats()
    SUPG_PT_stats = init_stats()
    sample2test_PT_stats = init_stats()
    PQE_RT_stats = init_stats()
    CSE_RT_stats = init_stats()
    SUPG_RT_stats = init_stats()
    sample2test_RT_stats = init_stats()

    for subset_size in subset_sizes:
        subset = np.random.choice(len(oracle), size=subset_size, replace=False)
        subset_oracle = oracle[subset]
        subset_proxy = proxy[subset]

        np.random.seed(1)
        index = np.random.choice(range(subset_size), size=n_query, replace=False)

        for i in range(len(index)):
            proxy_dist, oracle_dist = preprocess_dist(subset_oracle, subset_proxy, subset_oracle[[index[i]]])
            ranks = preprocess_ranks(proxy_dist)

            for _ in range(repeat):
                # CSE-PT
                CSE_PT_start = time.time()
                precision, recall, PT_cost = test_CSE_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='approx_m1')
                CSE_PT_end = time.time()
                CSE_PT_stats[subset_size]['success'].append(int(precision >= pt))
                CSE_PT_stats[subset_size]['CR'].append(recall)
                CSE_PT_stats[subset_size]['cost'].append(PT_cost)
                CSE_PT_stats[subset_size]['overhead'].append(CSE_PT_end - CSE_PT_start)

                if is_sample2test:
                    # sample2test-PT
                    sample2test_PT_start = time.time()
                    precision, recall, _ = test_sample2test_PT(oracle_dist, ranks, bd=PT_cost, t=t, pt=pt)
                    sample2test_PT_end = time.time()
                    sample2test_PT_stats[subset_size]['success'].append(int(precision >= pt))
                    sample2test_PT_stats[subset_size]['CR'].append(recall)
                    sample2test_PT_stats[subset_size]['cost'].append(PT_cost)
                    sample2test_PT_stats[subset_size]['overhead'].append(sample2test_PT_end - sample2test_PT_start)
                else:
                    # SUPG-PT
                    SUPG_PT_start = time.time()
                    try:
                        precision, recall, _, _ = SUPG(oracle_dist, proxy_dist, t, pt, prob, cost=PT_cost, query_type='PT')
                    except ZeroDivisionError:
                        precision = recall = 0
                    SUPG_PT_end = time.time()
                    SUPG_PT_stats[subset_size]['success'].append(int(precision >= pt))
                    SUPG_PT_stats[subset_size]['CR'].append(recall)
                    SUPG_PT_stats[subset_size]['cost'].append(PT_cost)
                    SUPG_PT_stats[subset_size]['overhead'].append(SUPG_PT_end - SUPG_PT_start)

                    # PQE-PT
                    PQE_PT_start = time.time()
                    precision, recall, _ = test_PQE_PT(oracle_dist, proxy_dist, bd=PT_cost, t=t, prob=prob, pt=pt)
                    PQE_PT_end = time.time()
                    PQE_PT_stats[subset_size]['success'].append(int(precision >= pt))
                    PQE_PT_stats[subset_size]['CR'].append(recall)
                    PQE_PT_stats[subset_size]['cost'].append(PT_cost)
                    PQE_PT_stats[subset_size]['overhead'].append(PQE_PT_end - PQE_PT_start)

                # --------------------------------------------------------------------

                # CSE-RT
                CSE_RT_start = time.time()
                precision, recall, RT_cost = test_CSE_RT(oracle_dist, ranks, t=t, delta=delta, prob=rt_prob, rt=rt,
                                                         mode='approx_m1')
                CSE_RT_end = time.time()
                CSE_RT_stats[subset_size]['success'].append(int(recall >= rt))
                CSE_RT_stats[subset_size]['CR'].append(precision)
                CSE_RT_stats[subset_size]['cost'].append(RT_cost)
                CSE_RT_stats[subset_size]['overhead'].append(CSE_RT_end - CSE_RT_start)

                if is_sample2test:
                    # sample2test-RT
                    sample2test_RT_start = time.time()
                    precision, recall, _ = test_sample2test_RT(oracle_dist, ranks, bd=RT_cost, t=t, rt=rt)
                    sample2test_RT_end = time.time()
                    sample2test_RT_stats[subset_size]['success'].append(int(recall >= rt))
                    sample2test_RT_stats[subset_size]['CR'].append(precision)
                    sample2test_RT_stats[subset_size]['cost'].append(RT_cost)
                    sample2test_RT_stats[subset_size]['overhead'].append(sample2test_RT_end - sample2test_RT_start)
                else:
                    # SUPG-RT
                    SUPG_RT_start = time.time()
                    try:
                        precision, recall, _, _ = SUPG(oracle_dist, proxy_dist, t, rt, prob, cost=RT_cost, query_type='RT')
                    except ZeroDivisionError:
                        precision = recall = 0
                    SUPG_RT_end = time.time()
                    SUPG_RT_stats[subset_size]['success'].append(int(recall >= rt))
                    SUPG_RT_stats[subset_size]['CR'].append(precision)
                    SUPG_RT_stats[subset_size]['cost'].append(RT_cost)
                    SUPG_RT_stats[subset_size]['overhead'].append(SUPG_RT_end - SUPG_RT_start)

                    # PQE-RT
                    PQE_RT_start = time.time()
                    precision, recall, _ = test_PQE_RT(oracle_dist, proxy_dist, bd=RT_cost, t=t, prob=prob, rt=rt)
                    PQE_RT_end = time.time()
                    PQE_RT_stats[subset_size]['success'].append(int(recall >= rt))
                    PQE_RT_stats[subset_size]['CR'].append(precision)
                    PQE_RT_stats[subset_size]['cost'].append(RT_cost)
                    PQE_RT_stats[subset_size]['overhead'].append(PQE_RT_end - PQE_RT_start)

    if is_sample2test:
        backup_res = [subset_sizes, sample2test_PT_stats, sample2test_RT_stats, prob]
        Path("./results/scalability_sample2test/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/scalability_sample2test/' + fname + '.pkl', 'wb'))
    else:
        backup_res = [subset_sizes, PQE_PT_stats, CSE_PT_stats, SUPG_PT_stats,
                      PQE_RT_stats, CSE_RT_stats, SUPG_RT_stats, prob]
        Path("./results/scalability/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/scalability/' + fname + '.pkl', 'wb'))


def exp_compare_PQE_PQA(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=False):
    rt_k_success = defaultdict(list)
    rt_k_precis = defaultdict(list)
    pt_k_success = defaultdict(list)
    pt_k_recall = defaultdict(list)
    norm_scale_list = np.array([0.01, 0.2, 0.4, 0.6, 0.8, 1])

    for i in range(len(index)):
        for j in norm_scale_list:
            proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
            topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=j, t=t)
            sync_oracle = oracle_dist
            true_ans = np.where(sync_oracle <= t)[0]

            # PQA-PT-sync
            _, _, k_star = test_PQA_PT(sync_oracle, phi, topk, t=t, prob=prob, pt=pt)

            if k_star == 0:
                pt_k_success[j].append(1)
                pt_k_recall[j].append(0)
            else:
                k_hat = k_star
                ans = topk[:k_hat]

                true_pos = len(np.intersect1d(ans, true_ans))
                precision = true_pos / len(ans)
                recall = true_pos / len(true_ans)

                pt_k_success[j].append(int(precision >= pt))
                pt_k_recall[j].append(recall)

            # PQA-RT-sync
            _, _, k_star = test_PQA_RT(sync_oracle, phi, topk, t=t, prob=prob, rt=rt)

            k_hat = k_star
            ans = topk[:k_hat]

            true_pos = len(np.intersect1d(ans, true_ans))
            precision = true_pos / len(ans)
            recall = true_pos / len(true_ans)

            rt_k_success[j].append(int(recall >= rt))
            rt_k_precis[j].append(precision)

    if not is_precompile:
        backup_res = [norm_scale_list, pt_k_recall, pt_k_success, rt_k_precis, rt_k_success, prob]
        Path("./results/COMP_PQA_PQE/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/COMP_PQA_PQE/' + fname + '.pkl', 'wb'))


def exp_compare_CSA_CSE(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=False):
    rt_success = defaultdict(list)
    rt_precis = defaultdict(list)
    pt_success = defaultdict(list)
    pt_recall = defaultdict(list)
    scale_list = np.array([-75, -50, -25, 0, 25, 50, 75])

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)
        true_ans = np.where(oracle_dist <= t)[0]
        K_rt = floor(len(true_ans) * (1 - rt)) + 1
        K_pt = find_K_pt(ranks, oracle_dist, t=t, pt=pt)

        # CSC-RT
        for _ in range(repeat):
            for scale in scale_list:
                K_scaled = min(ceil(K_rt * (1 + scale / 100)), len(oracle))
                precision, recall, _, _, _ = test_CSC_RT(oracle_dist, ranks, t=t, prob=prob, rt=rt, mode='exact',
                                                         K=K_scaled)
                success = int(recall >= rt)
                rt_success[scale].append(success)
                rt_precis[scale].append(precision)

        # CSC-PT
        for _ in range(repeat):
            for scale in scale_list:
                K_scaled = min(ceil(K_pt * (1 + scale / 100)), len(oracle))
                precision, recall, _, _, _, _, _, _ = test_CSC_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt,
                                                                  mode='exact',
                                                                  K=K_scaled)
                success = int(precision >= pt)
                pt_success[scale].append(success)
                pt_recall[scale].append(recall)

    if not is_precompile:
        backup_res = [scale_list, pt_recall, pt_success, rt_precis, rt_success, prob]
        Path("./results/COMP_CSC_CSE/").mkdir(parents=True, exist_ok=True)
        pickle.dump(backup_res, open('results/COMP_CSC_CSE/' + fname + '.pkl', 'wb'))


def exp_compare_sample2test_CSE(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname=''):
    delta = 1 - sqrt(prob)
    rt_prob = prob / (1 - delta)
    scale_list = np.array([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
    CSE_PT_stats = defaultdict(list)
    CSE_RT_stats = defaultdict(list)

    sample2test_PT = dict()
    sample2test_RT = dict()

    for scale in scale_list:
        sample2test_PT[scale] = defaultdict(list)
        sample2test_RT[scale] = defaultdict(list)

    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        ranks = preprocess_ranks(proxy_dist)

        for _ in range(repeat):
            # C-PT
            CSE_PT_start = time.time()
            precision, recall, PT_cost = test_CSE_PT(oracle_dist, ranks, t=t, prob=prob, pt=pt, mode='approx_m1')
            CSE_PT_end = time.time()
            CSE_PT_stats['success'].append(int(precision >= pt))
            CSE_PT_stats['CR'].append(recall)
            CSE_PT_stats['cost'].append(PT_cost)
            CSE_PT_stats['overhead'].append(CSE_PT_end - CSE_PT_start)

        avg_PT_cost = np.mean(CSE_PT_stats['cost'][i * repeat:(i + 1) * repeat])

        for scale in scale_list:
            cost = min(ceil(avg_PT_cost * (1 + scale / 100)), len(oracle_dist))
            for _ in range(repeat):
                # sample2test-PT
                sample2test_PT_start = time.time()
                precision, recall, _ = test_sample2test_PT(oracle_dist, ranks, bd=cost, t=t, pt=pt)
                sample2test_PT_end = time.time()
                sample2test_PT[scale]['success'].append(int(precision >= pt))
                sample2test_PT[scale]['CR'].append(recall)
                sample2test_PT[scale]['cost'].append(cost)
                sample2test_PT[scale]['overhead'].append(sample2test_PT_end - sample2test_PT_start)

        # --------------------------------------------------------------------

        for _ in range(repeat):
            # CSE-RT
            CSE_RT_start = time.time()
            precision, recall, RT_cost = test_CSE_RT(oracle_dist, ranks, t=t, delta=delta, prob=rt_prob, rt=rt,
                                                     mode='approx_m1')
            CSE_RT_end = time.time()
            CSE_RT_stats['success'].append(int(recall >= rt))
            CSE_RT_stats['CR'].append(precision)
            CSE_RT_stats['cost'].append(RT_cost)
            CSE_RT_stats['overhead'].append(CSE_RT_end - CSE_RT_start)

        avg_RT_cost = np.mean(CSE_RT_stats['cost'][i * repeat:(i + 1) * repeat])

        for scale in scale_list:
            cost = min(ceil(avg_RT_cost * (1 + scale / 100)), len(oracle_dist))
            for _ in range(repeat):
                # sample2test-RT
                sample2test_RT_start = time.time()
                precision, recall, _ = test_sample2test_RT(oracle_dist, ranks, bd=cost, t=t, rt=rt)
                sample2test_RT_end = time.time()
                sample2test_RT[scale]['success'].append(int(recall >= rt))
                sample2test_RT[scale]['CR'].append(precision)
                sample2test_RT[scale]['cost'].append(cost)
                sample2test_RT[scale]['overhead'].append(sample2test_RT_end - sample2test_RT_start)

    method_list = ['sample2test', 'CSE']
    backup_res = [method_list, scale_list, sample2test_PT, CSE_PT_stats,
                  sample2test_RT, CSE_RT_stats, prob]
    Path("./results/COMP_sample2test_CSE/").mkdir(parents=True, exist_ok=True)
    pickle.dump(backup_res, open('results/COMP_sample2test_CSE/' + fname + '.pkl', 'wb'))


def exp_compare_topk_PQA(oracle, proxy, index, pt=0.9, rt=0.9, t=0.9, prob=0.9, fname=''):
    rt_k_success = defaultdict(list)
    rt_k_precis = defaultdict(list)
    pt_k_success = defaultdict(list)
    pt_k_recall = defaultdict(list)
    topk_precision = defaultdict(list)
    topk_recall = defaultdict(list)
    topk_cost = defaultdict(list)
    scale_list = np.array([0.1])

    for i in range(len(index)):
        proxy_dist, _ = preprocess_dist(oracle, proxy, oracle[[index[i]]])

        for j in scale_list:
            # topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=j, t=t)
            sync_oracle = preprocess_sync(proxy_dist, j)

            # # PQA-PT-sync
            # precision, recall, _ = test_PQA_PT(sync_oracle, phi, topk, t=t, prob=prob, pt=pt)
            # pt_k_success[j].append(int(precision >= pt))
            # pt_k_recall[j].append(recall)
            #
            # precision, recall, _ = test_PQA_RT(sync_oracle, phi, topk, t=t, prob=prob, rt=rt)
            # rt_k_success[j].append(int(recall >= rt))
            # rt_k_precis[j].append(precision)

            precision, recall, cost = test_topk(oracle_dist=sync_oracle, proxy_dist=proxy_dist, scale=j, t=t, prob=prob)
            topk_recall[j].append(recall)
            topk_precision[j].append(precision)
            topk_cost[j].append(cost)

    backup_res = [scale_list, pt_k_recall, pt_k_success, rt_k_precis, rt_k_success, topk_recall, topk_precision, topk_cost, prob, pt, rt]
    Path("./results/COMP_topk_PQA/").mkdir(parents=True, exist_ok=True)
    pickle.dump(backup_res, open('results/COMP_topk_PQA/' + fname + '.pkl', 'wb'))


def pre_compile():
    s_proxy, s_oracle = load_data(name='voc')
    np.random.seed(1)
    s = np.random.choice(len(s_oracle), size=200, replace=False)
    s_oracle = s_oracle[s]
    s_proxy = s_proxy[s]

    exp_PQA_maximal_CR(s_oracle, s_proxy, [0], pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=True)
    exp_CSC_minimal_cost(s_oracle, s_proxy, [0], pt=0.9, rt=0.9, t=0.9, prob=0.9, fname='', is_precompile=True)

    print('precompile done!')


pre_compile()
if __name__ == '__main__':
    Pt = Rt = 0.95
    Prob = 0.9
    Dist_t = 0.9

    # num_query = 100
    # start_time = time.time()
    # for Fname in ['voc']:
    #     Proxy, Oracle = load_data(name=Fname)
    #     np.random.seed(0)
    #     Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)
    #     exp_compare_topk_PQA(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #     # exp_compare_sample2test_CSE(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #     # exp_compare_PQE_PQA(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #     # exp_compare_CSA_CSE(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #
    # end_time = time.time()
    # print('execution time is %.2fs' % (end_time - start_time))


    # # 200 * 1 trials
    # num_query = 200
    # for Fname in ['voc', 'icd9_eICU']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco'
    #     Proxy, Oracle = load_data(name=Fname)
    #     np.random.seed(0)
    #     Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)
    #     exp_PQA_maximal_CR(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #
    # 50 * 10 trials
    start_time = time.time()
    num_query = 50
    # for Fname in ['voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco'
    #     Proxy, Oracle = load_data(name=Fname)
    #
    #     if Fname == 'coco':
    #         np.random.seed(1)
    #         Samples = np.random.choice(len(Oracle), size=8000, replace=False)
    #         Oracle = Oracle[Samples]
    #         Proxy = Proxy[Samples]
    #         Fname = Fname + '8000'
    #
    #     np.random.seed(0)
    #     Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)
    #
    #     # if Fname in ['voc', 'icd9_eICU']:
    #     #     exp_CSC_minimal_cost(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #     #     exp_comprehensive_comparison(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #
    #     # exp_CSC_cpu_overhead(Oracle, Proxy, Index, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)
    #     exp_runningtime(Oracle, Proxy, Index, is_sample2test=True, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname=Fname)

    exp_scalability_test(is_sample2test=True, pt=Pt, rt=Rt, t=Dist_t, prob=Prob, fname='coco', n_query=num_query)

    end_time = time.time()
    print('execution time is %.2fs' % (end_time - start_time))
