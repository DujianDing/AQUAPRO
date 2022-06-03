import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")


def f_map(f):
    # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000'v
    if f == 'voc':
        return 'VOC'
    if f == 'icd9_eICU':
        return 'eICU'
    if f == 'icd9_mimic':
        return 'MIMIC-III'
    if f == 'jackson10000.csv':
        return 'night-street'
    if f == 'coco8000':
        return 'COCO(small)'
    if f == 'coco':
        return 'COCO'


def _compact_plot_PQA(f, f_path):
    # sns.set_context("talk")
    roll_datasets = list()
    roll_types = list()
    roll_scales = list()
    roll_success = list()
    roll_cr = list()

    s_list, pt_k_recall, pt_k_success, rt_k_precis, rt_k_success, prob = pickle.load(open(f_path, "rb"))
    pt_0_success = float(np.mean(pt_k_success[0]))
    rt_0_success = float(np.mean(rt_k_success[0]))
    pt_0_cr = float(np.mean(pt_k_recall[0]))
    rt_0_cr = float(np.mean(rt_k_precis[0]))
    print('PQA-RT(%s): w/ k^*, success %.2f, cr %.2f' % (f, rt_0_success, rt_0_cr))
    print('PQA-PT(%s): w/ k^*, success %.2f, cr %.2f' % (f, pt_0_success, pt_0_cr))

    for _ in s_list:
            trials = len(pt_k_recall[_])

            roll_datasets.extend([f] * trials)
            roll_types.extend(['RT'] * trials)
            roll_scales.extend([_] * trials)
            roll_success.extend(rt_k_success[_])
            roll_cr.extend(rt_k_precis[_])

            roll_datasets.extend([f] * trials)
            roll_types.extend(['PT'] * trials)
            roll_scales.extend([_] * trials)
            roll_success.extend(pt_k_success[_])
            roll_cr.extend(pt_k_recall[_])

    data = pd.DataFrame({'datasets': roll_datasets, 'query_type': roll_types,
                         'scales': roll_scales, 'CR': roll_cr, 'success': roll_success})

    def twin_lineplot(x, y, z, color, **kwargs):
        ax1 = plt.gca()
        color = 'tab:blue'
        ax1.set_ylabel('CR', color=color)  # we already handled the x-label with ax1
        sns.lineplot(x=x, y=y, color=color, **kwargs, ax=ax1, ci=90)
        # ax1.plot([0], [np.mean(cr_list[0])], color=color, marker='*', markersize=15)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0.15, 0.85])

        ax2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('success prob.', color=color)
        ax2.axhline(y=prob, color='k', linestyle='dashdot')
        sns.lineplot(x=x, y=z, color=color, **kwargs, ax=ax2, ci=None)
        # ax2.plot([0], [np.mean(success_list[0])], color=color, marker='*', markersize=15)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([-0.05, 1.05])

    g = sns.FacetGrid(data, row='query_type', height=2, aspect=1.5)
    g.map(twin_lineplot, 'scales', 'CR', 'success', color='r')
    g.axes[0, 0].set_title('PQA-RT(%s)' % f)
    g.axes[1, 0].set_title('PQA-PT(%s)' % f)
    g.axes[1, 0].set_xlabel(r'$k^*$ perturbation (%)')
    plt.show()


def _plot_CSA(p, m_list, query_type, f, cost_list, success_list):
    # sns.set_style("ticks")
    sns.set_context("talk")
    roll_methods = list()
    roll_success = list()
    roll_cost = list()
    for i in range(len(m_list)):
        trials = len(cost_list[m_list[i]])
        roll_success.extend(success_list[m_list[i]])
        roll_cost.extend(cost_list[m_list[i]])
        roll_methods.extend([m_list[i]]*trials)
    data = pd.DataFrame({'cost': roll_cost, 'success': roll_success, 'methods': roll_methods})

    fig, ax1 = plt.subplots()
    if query_type == 'PT':
        ax1.set_title('CSC-%s (%s)' % (query_type, f))
    else:
        ax1.set_title('CSC-%s (%s)' % (query_type, f))
    ax1.set_xlabel('methods')
    color = 'olivedrab'
    sns.barplot(x="methods", y="cost", data=data, color=color, ax=ax1, ci=None)
    ax1.set_ylabel('#oracle calls', color=color)  # we already handled the x-label with ax1
    width_scale = 0.5
    for bar in ax1.containers[0]:
        bar.set_width(bar.get_width() * width_scale)

    ax1.bar_label(ax1.containers[0], fmt='%d', size=11)
    if f == 'eICU':
        ax1.set_ylim([0, 8500])     # eICU: 8500
    elif f == 'VOC':
        ax1.set_ylim([0, 5000])  # VOC: 5000
    else:
        print('not recognized f file!')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=20)
    # ax1.set_yscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    sns.barplot(x="methods", y="success", data=data, color=color, ax=ax2, ci=None)
    ax2.axhline(y=p, color='k', linestyle='dashdot')
    ax2.set_ylabel('success prob.', color=color)
    for bar in ax2.containers[0]:
        x = bar.get_x()
        w = bar.get_width()
        bar.set_x(x + w * (1 - width_scale))
        bar.set_width(w * width_scale)

    #     ax2.axhline(y=np.mean(success_list['all']), color='g', linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0, 1.09])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    if query_type == 'PT':
        print('average cost (ALL):', np.mean(np.append(cost_list['K0'], cost_list['exact'])))
        print('average success prob (ALL):', np.mean(np.append(success_list['K0'], success_list['exact'])))
        freq_k0 = len(cost_list['K0'])
        print('K=0 frequency:', freq_k0)
        if freq_k0 > 0:
            print('average cost (K=0)', (np.sum(cost_list['K0'])) / freq_k0)
        print('average cost (exact):', np.mean(cost_list['exact']))
        print('average cost (approx_s1):', np.mean(cost_list['approx_s1']))
        print('average cost (approx_m1):', np.mean(cost_list['approx_m1']))


def _plot_CSA_overhead(query_type, f_l, path_prefix, exp_n):
    sns.set_context("talk")
    roll_cost = list()
    roll_datasets = list()
    roll_methods = list()
    for f in f_l:
        f_path = path_prefix + exp_n + '/' + f + '.pkl'
        f = f_map(f)
        if query_type == 'PT':
            m_list, _, overhead_list = pickle.load(open(f_path, "rb"))
        else:
            m_list, overhead_list, _ = pickle.load(open(f_path, "rb"))
        speedup = int(np.mean(overhead_list['exact']) / np.mean(overhead_list['approx_m1']))
        print('%s-%s, OH(approx_m1) / OH(exact)=%d' % (f, query_type, speedup))

        for i in range(len(m_list)):
            trials = len(overhead_list[m_list[i]])
            roll_cost.extend(overhead_list[m_list[i]])
            roll_methods.extend([m_list[i]]*trials)
            roll_datasets.extend([f]*trials)
    data = pd.DataFrame({'cost': roll_cost, 'methods': roll_methods, 'datasets': roll_datasets})

    fig, ax1 = plt.subplots()
    ax1.set_title('CSC-%s' % query_type)
    ax1.set_xlabel('methods')
    # color = 'darkgreen'
    sns.barplot(x="datasets", y="cost", data=data, hue='methods', ax=ax1, ci=None)
    ax1.set_ylabel('CPU overhead (s)')
    # ax1.set_ylim([0.5, 0.58])
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x', rotation=20)
    ax1.set_yscale('log')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def _plot_CSE_PQE(p, m_l, query_type, f, pqe, cse, supg):
    sns.set_context("talk")
    roll_methods = list()
    roll_success = list()
    roll_cr = list()

    trials = len(pqe[0]['CR'])

    roll_success.extend(pqe[0]['success'])
    roll_cr.extend(pqe[0]['CR'])
    roll_methods.extend(['PQE'] * trials)

    roll_cr.extend(cse['CR'])
    roll_success.extend(cse['success'])
    roll_methods.extend(['CSE'] * len(cse['CR']))

    roll_cr.extend(supg[0]['CR'])
    roll_success.extend(supg[0]['success'])
    roll_methods.extend(['SUPG'] * trials)

    print('%s-%s, SUPG success %.2f, CR(PQE)-CR(SUPG) %.2f, CR(CSE)-CR(SUPG) %.2f' %
          (f, query_type, float(np.mean(supg[0]['success'])), float(np.mean(pqe[0]['CR']) - np.mean(supg[0]['CR'])),
           float(np.mean(cse['CR']) - np.mean(supg[0]['CR']))))

    data = pd.DataFrame({'CR': roll_cr, 'success': roll_success, 'methods': roll_methods})

    fig, ax1 = plt.subplots()
    ax1.set_title('%s (%s)' % (query_type, f))
    ax1.set_xlabel('methods')
    color = 'tab:blue'
    sns.barplot(x="methods", y="CR", data=data, color=color, ax=ax1, ci=None)
    ax1.set_ylabel('CR', color=color)  # we already handled the x-label with ax1
    width_scale = 0.5
    for bar in ax1.containers[0]:
        bar.set_width(bar.get_width() * width_scale)

    # ax1.bar_label(ax1.containers[0], fmt='%d', size=11)
    # ax1.set_ylim([0, 5000])     # eICU: 8500, VOC: 5000
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.tick_params(axis='x', rotation=20)
    # ax1.set_yscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    sns.barplot(x="methods", y="success", data=data, color=color, ax=ax2, ci=None)
    ax2.axhline(y=p, color='k', linestyle='dashdot')
    ax2.set_ylabel('success prob.', color=color)
    for bar in ax2.containers[0]:
        x = bar.get_x()
        w = bar.get_width()
        bar.set_x(x + w * (1 - width_scale))
        bar.set_width(w * width_scale)

    #     ax2.axhline(y=np.mean(success_list['all']), color='g', linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_ylim([0, 1.09])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def _plot_COMP(prob, m_l, query_type, f, s_l, pqe, cse, supg):
    sns.set_context("talk")
    roll_success = list()
    roll_cr = list()
    roll_methods = list()
    roll_indices = list()
    roll_scales = list()
    x_axis = np.arange(len(s_l))
    print('%s-%s, cse cr %.2f' % (f, query_type, float(np.mean(cse['CR']))))
    print('%s-%s, supg -50%% success %.2f, 50%% success %.2f' % (f, query_type,
                                                                 float(np.mean(supg[s_l[0]]['success'])),
                                                                 float(np.mean(supg[s_l[-1]]['success']))))
    print('%s-%s, supg -50%% CR %.2f, 50%% CR %.2f' % (f, query_type,
                                                       float(np.mean(supg[s_l[0]]['CR'])),
                                                       float(np.mean(supg[s_l[-1]]['CR']))))
    print('%s-%s, pqe -50%% success %.2f, 50%% success %.2f' % (f, query_type,
                                                                float(np.mean(pqe[s_l[0]]['success'])),
                                                                float(np.mean(pqe[s_l[-1]]['success']))))
    print('%s-%s, pqe -50%% CR %.2f, 50%% CR %.2f' % (f, query_type,
                                                      float(np.mean(pqe[s_l[0]]['CR'])),
                                                      float(np.mean(pqe[s_l[-1]]['CR']))))
    for i in range(len(s_l)):
        trials = len(pqe[s_l[i]]['CR'])
        roll_success.extend(pqe[s_l[i]]['success'])
        roll_cr.extend(pqe[s_l[i]]['CR'])
        roll_methods.extend(['PQE'] * trials)
        roll_indices.extend([x_axis[i]] * trials)
        roll_scales.extend([s_l[i]] * trials)

        roll_cr.extend(supg[s_l[i]]['CR'])
        roll_success.extend(supg[s_l[i]]['success'])
        roll_methods.extend(['SUPG'] * trials)
        roll_indices.extend([x_axis[i]] * trials)
        roll_scales.extend([s_l[i]] * trials)

        print('scale %d, CR[PQE]-CR[SUPG] %.2f' % (s_l[i], float(np.mean(pqe[s_l[i]]['CR'])-np.mean(supg[s_l[i]]['CR']))))

    data = pd.DataFrame({'indices': roll_indices, 'CR': roll_cr, 'success': roll_success,
                         'methods': roll_methods, 'scales': roll_scales})

    fig, ax1 = plt.subplots()
    ax1.set_title('%s: (%s)' % (query_type, f))
    ax1.set_xlabel('budget perturbation (%)')
    color = 'tab:blue'
    ax1.set_ylabel('CR', color=color)  # we already handled the x-label with ax1
    sns.lineplot(x="scales", y="CR", data=data, style='methods',
                 color=color, ax=ax1, ci=90, markers=True, markersize=10)
    ax1.axhline(y=np.mean(cse['CR']), color='k', linestyle=':', label='CSE')
    ax1.legend()
    ax1.axhline(y=np.mean(cse['CR']), color=color, linestyle=':')
    # sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    sns.move_legend(ax1, loc='lower right')
    ax1.set_ylim([0, 1.05])
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_xticks(x_axis)
    # ax1.set_xticklabels(s_l)
    # ax1.set_yscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('success prob.', color=color)
    ax2.axhline(y=prob, color='k', linestyle='-.')
    sns.lineplot(x="scales", y="success", data=data, style='methods',
                 color=color, ax=ax2, ci=None, markers=True, markersize=10)
    ax2.legend_.remove()
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1.05])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def _plot_running_time(prc_l, query_type, f, pqe, cse, supg):
    # sns.set_context("talk")
    roll_overhead = list()
    roll_methods = list()
    roll_indices = list()
    prc_l = [0.3, 0.45, 0.75, 1.5, 2.5]
    x_axis = np.arange(len(prc_l))
    for i in range(len(prc_l)):
        trials = len(pqe['overhead'])
        roll_overhead.extend(np.array(pqe['overhead']) + prc_l[i] * np.array(pqe['cost']))
        roll_methods.extend(['PQE'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

        roll_overhead.extend(np.array(cse['overhead']) + prc_l[i] * np.array(cse['cost']))
        roll_methods.extend(['CSE'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

        roll_overhead.extend(np.array(supg['overhead']) + prc_l[i] * np.array(supg['cost']))
        roll_methods.extend(['SUPG'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

    data = pd.DataFrame({'indices': roll_indices, 'overhead': roll_overhead, 'methods': roll_methods})

    fig, ax1 = plt.subplots()
    ax1.set_title('%s: overall running time (%s)' % (query_type, f))
    ax1.set_xlabel('seconds per oracle call')
    ax1.set_ylabel('running time (s)')
    sns.lineplot(x="indices", y="overhead", data=data, hue='methods', style='methods',
                 ax=ax1, ci=None, markers=True, markersize=10)
    # ax1.set_ylim([0.5, 0.58])
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(prc_l)
    # ax1.set_yscale('log')
    ax1.legend()

    fig.tight_layout()
    plt.show()


def _print_running_time(query_type, f_l, path_prefix, exp_n):
    oracle_price = {'VOC': 20, 'COCO(small)': 20, 'MIMIC-III': 900, 'eICU': 900, 'night-street': 0.4}
    proxy_price = {'VOC': 0.05, 'COCO(small)': 0.05, 'MIMIC-III': 0.001, 'eICU': 0.001, 'night-street': 0.02}
    data_size = {'VOC': 4952, 'COCO(small)': 8000, 'MIMIC-III': 4244, 'eICU': 8236, 'night-street': 10000}     # COCO:40137

    print('%s  DATASET      PQE      CSE      SUPG' % query_type)
    for f in f_l:
        f_path = path_prefix + exp_n + '/' + f + '.pkl'
        f = f_map(f)
        if query_type == 'PT':
            _, pqe, cse, supg, _, _, _, _ = pickle.load(open(f_path, "rb"))
        else:
            _, _, _, _, pqe, cse, supg, _ = pickle.load(open(f_path, "rb"))

        pqe_cost = data_size[f] * proxy_price[f] + np.mean(pqe['cost']) * oracle_price[f] + np.mean(pqe['overhead'])
        cse_cost = data_size[f] * proxy_price[f] + np.mean(cse['cost']) * oracle_price[f] + np.mean(cse['overhead'])
        supg_cost = data_size[f] * proxy_price[f] + np.mean(supg['cost']) * oracle_price[f] + np.mean(supg['overhead'])
        print('    %s      %.2f      %.2f      %.2f' % (f, pqe_cost/3600, cse_cost/3600, supg_cost/3600))

    print('#Oracle-%s' % query_type)
    for f in f_l:
        f_path = path_prefix + exp_n + '/' + f + '.pkl'
        f = f_map(f)
        if query_type == 'PT':
            _, pqe, cse, supg, _, _, _, _ = pickle.load(open(f_path, "rb"))
        else:
            _, _, _, _, pqe, cse, supg, _ = pickle.load(open(f_path, "rb"))
        print('    %s      %d        %d        %d' % (f, int(np.mean(pqe['cost'])), int(np.mean(cse['cost'])),
                                                      int(np.mean(supg['cost']))))


def _plot_scalability_test(s_l, query_type, f, pqe, cse, supg):
    sns.set_context("talk")
    roll_overhead = list()
    roll_methods = list()
    roll_indices = list()
    x_axis = np.arange(len(s_l))
    for i in range(len(s_l)):
        trials = len(pqe[s_l[i]]['overhead'])
        roll_overhead.extend(pqe[s_l[i]]['overhead'])
        roll_methods.extend(['PQE'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

        roll_overhead.extend(cse[s_l[i]]['overhead'])
        roll_methods.extend(['CSE'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

        roll_overhead.extend(supg[s_l[i]]['overhead'])
        roll_methods.extend(['SUPG'] * trials)
        roll_indices.extend([x_axis[i]] * trials)

        print('%s-%d, PQE time %.2f, SUPG time %.4f, CSE time %.4f' % (query_type, s_l[i],
                                                                       float(np.mean(pqe[s_l[i]]['overhead'])),
                                                                       float(np.mean(supg[s_l[i]]['overhead'])),
                                                                       float(np.mean(cse[s_l[i]]['overhead']))))

    data = pd.DataFrame({'indices': roll_indices, 'overhead': roll_overhead, 'methods': roll_methods})
    hue_order = ['PQE', 'CSE', 'SUPG']
    fig, ax1 = plt.subplots()
    ax1.set_title('%s: (%s)' % (query_type, f))
    ax1.set_xlabel('subset sizes')
    ax1.set_ylabel('CPU overheads (s)')
    sns.lineplot(x="indices", y="overhead", data=data, hue='methods', style='methods', hue_order=hue_order,
                 ax=ax1, ci=90, markers=True, markersize=10)
    # ax1.set_ylim([0.5, 0.58])
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(s_l)
    # ax1.set_yscale('log')
    ax1.legend()

    fig.tight_layout()
    plt.show()
    print(query_type)
    print('  SIZE        COST          SUCCESS             CR  ')
    print('                        PQE   CSE  SUPG    PQE  CSE  SUPG')
    for s in s_l:
        print(' %d       %.2f       %.2f  %.2f  %.2f    %.2f  %.2f  %.2f ' %
              (s, float(np.mean(cse[s]['cost'])), float(np.mean(pqe[s]['success'])), float(np.mean(cse[s]['success'])),
               float(np.mean(supg[s]['success'])),
               float(np.mean(pqe[s]['CR'])), float(np.mean(cse[s]['CR'])), float(np.mean(supg[s]['CR']))))
    print('')


if __name__ == '__main__':
    exp_name = 'PQA'
    file_path_prefix = './results/'

    if exp_name == 'PQA':
        for fname in ['voc', 'icd9_eICU']:
            file_path = file_path_prefix + exp_name + '/' + fname + '.pkl'
            fname = f_map(fname)
            _compact_plot_PQA(f=fname, f_path=file_path)
    elif exp_name == 'CSC':
        for fname in ['voc', 'icd9_eICU']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000'
            file_path = file_path_prefix + exp_name + '/' + fname + '.pkl'
            fname = f_map(fname)
            mode_list, pt_cost, pt_success, rt_cost, rt_success, prob = pickle.load(open(file_path, "rb"))
            _plot_CSA(p=prob, m_list=mode_list, query_type='RT', f=fname, cost_list=rt_cost, success_list=rt_success)
            _plot_CSA(p=prob, m_list=mode_list, query_type='PT', f=fname, cost_list=pt_cost, success_list=pt_success)
    elif exp_name == 'overhead':
        file_list = ['voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000']
        _plot_CSA_overhead(query_type='RT', f_l=file_list, path_prefix=file_path_prefix, exp_n=exp_name)
        _plot_CSA_overhead(query_type='PT', f_l=file_list, path_prefix=file_path_prefix, exp_n=exp_name)
    elif exp_name == 'CMPR':
        for fname in ['voc', 'icd9_eICU']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000'
            file_path = file_path_prefix + exp_name + '/' + fname + '.pkl'
            fname = f_map(fname)
            method_list, scale_list, PQE_PT, CSE_PT_stats, SUPG_PT, PQE_RT, CSE_RT_stats, SUPG_RT, prob = pickle.load(open(file_path, "rb"))
            _plot_CSE_PQE(p=prob, m_l=method_list, query_type='RT', f=fname, pqe=PQE_RT, cse=CSE_RT_stats, supg=SUPG_RT)
            _plot_CSE_PQE(p=prob, m_l=method_list, query_type='PT', f=fname, pqe=PQE_PT, cse=CSE_PT_stats, supg=SUPG_PT)
            _plot_COMP(prob=prob, m_l=method_list, query_type='RT', f=fname, s_l=scale_list,
                       pqe=PQE_RT, cse=CSE_RT_stats, supg=SUPG_RT)
            _plot_COMP(prob=prob, m_l=method_list, query_type='PT', f=fname, s_l=scale_list,
                       pqe=PQE_PT, cse=CSE_PT_stats, supg=SUPG_PT)
    elif exp_name == 'runningtime':
        file_list = ['voc', 'coco8000', 'icd9_mimic', 'icd9_eICU', 'jackson10000.csv']
        _print_running_time(query_type='RT', f_l=file_list, path_prefix=file_path_prefix, exp_n=exp_name)
        _print_running_time(query_type='PT', f_l=file_list, path_prefix=file_path_prefix, exp_n=exp_name)
        for fname in []:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000'
            file_path = file_path_prefix + exp_name + '/' + fname + '.pkl'
            fname = f_map(fname)
            price_l, PQE_PT_stats, CSE_PT_stats, SUPG_PT_stats, PQE_RT_stats, CSE_RT_stats, SUPG_RT_stats, prob = pickle.load(open(file_path, "rb"))
            _plot_running_time(prc_l=price_l, query_type='RT', f=fname, pqe=PQE_RT_stats, cse=CSE_RT_stats,
                               supg=SUPG_RT_stats)
            _plot_running_time(prc_l=price_l, query_type='PT', f=fname, pqe=PQE_PT_stats, cse=CSE_PT_stats,
                               supg=SUPG_PT_stats)
    elif exp_name == 'scalability':
        for fname in ['coco']:  # 'voc', 'icd9_eICU', 'icd9_mimic', 'jackson10000.csv', 'coco8000'
            file_path = file_path_prefix + exp_name + '/' + fname + '.pkl'
            fname = f_map(fname)
            subset_sizes, PQE_PT_stats, CSE_PT_stats, SUPG_PT_stats, PQE_RT_stats, CSE_RT_stats, SUPG_RT_stats, prob = pickle.load(open(file_path, "rb"))
            _plot_scalability_test(subset_sizes, 'RT', f=fname, pqe=PQE_RT_stats, cse=CSE_RT_stats, supg=SUPG_RT_stats)
            _plot_scalability_test(subset_sizes, 'PT', f=fname, pqe=PQE_PT_stats, cse=CSE_PT_stats, supg=SUPG_PT_stats)
