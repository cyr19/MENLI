import copy

import pandas as pd
import numpy as np
from collections import defaultdict
from mt_metrics_eval import data as me_data
from wmt import load_evaluation_data_1516, load_evaluation_data_17, read_human_scores
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import pickle
import torch
import seaborn as sns
import pprint
from types import SimpleNamespace
from summ import SummEval, RealSumm
import os
plt.style.use('seaborn-dark')

# MT
datasets_dict = {
        'ref': {'adv': ['paws_back', 'paws_para', 'wmt20_google-de', 'xpaws-de'],
                'mt-seg': ['wmt15', 'wmt16', 'wmt17','wmt20_mqm', 'wmt21.news_mqm'],
                'mt-sys': ['wmt21.news', 'wmt20'],
                },
        'src': {'adv': ['wmt20_google-de', 'xpaws-de', 'xpaws-fr', 'xpaws-zh', 'xpaws-ja'],
                'mt-seg': ['wmt15', 'wmt16', 'wmt17','wmt20_mqm', 'wmt21.news_mqm'],
                'mt-sys': ['wmt21.news', 'wmt20']
                }
    }

metrics_dict = {
    'ref': ['BARTScore_bart-large-cnn', 'BARTScore_bart-large-cnn+para_bi', 'BERTScore_roberta-large_L17_idf_version=0.3.11(hug_trans=4.17.0)',
            'BLEURT_BLEURT-20', 'COMET_wmt20-comet-da', 'MoverScore_bert_mnli_1-gram_idf(True)',
            'SentSim_new_BERTScore_ref', 'SentSim_new_WMD_ref'],
    'src': ['COMET_wmt21-comet-qe-mqm', 'XMoverScore_CLP', 'XMoverScore_UMD',
            'SentSim_new_BERTScore_src', 'SentSim_new_WMD_src']
}

# Summarization
datasets_dict_sum = {
    'adv': ['summ_google', 'Rank19'],
    'sum': ['summ', 'realsumm']
}

sum_metrics_dict = {
    'ref': ['BARTScore_bart-large-cnn', 'BARTScore_bart-large-cnn+para_bi', 'BERTScore_roberta-large_L17_idf_version=0.3.11(hug_trans=4.17.0)',
            'MoverScore_bert_mnli_1-gram_idf(True)', 'DiscoScore_DS_Focus_NN'],
    'src': ['BARTScore_bart-large-cnn', 'SUPERT_default']
}

nli_dict = {
    'ref': ['NLI1Score_monolingual', 'NLI2Score_monolingual'],
    'src': ['NLI1Score_crosslingual(xlm-roberta-base+2)', 'NLI2Score_crosslingual(mDeBERTa-v3-base-mnli-xnli+0)']
}

def calculate_accuracy_and_kendall(scores, scores_ad):
    num_hit = np.sum([scores[i] > scores_ad[i] for i in range(len(scores))])
    num_miss = np.sum([scores[i] < scores_ad[i] for i in range(len(scores))])
    accuracy = float(num_hit) / float(len(scores))
    kendall = float((num_hit - num_miss)) / float((num_hit + num_miss))
    return accuracy, kendall

def load_metric_scores(path, prob='e', adv=True):
    if adv:
        data = pd.read_csv(path)
        if 'NLI' in path:
            data = data[data['prob'] == prob]
        r = defaultdict(lambda: defaultdict(list))
        errors = sorted(set(data['error']))
        for error in errors:
            tmp = data[data['error'] == error].sort_values(by='id')
            r[error]['score'] = tmp['hyp_score'].tolist()
            r[error]['score_adv'] = tmp['hyp_adv_score'].tolist()
    else:
        with open(path, 'rb') as f:
            r = pickle.load(f)
    return r


def combine_nli(nli_metric, dataset, adv=True, agg='max', use_article=False, fluency=False):
    suffix = '' if not use_article else "_use_article"
    if adv:
        if dataset != 'Rank19':
            f = '_fluency' if fluency else ''
            e_data_rh = load_metric_scores('../results/adv_scores{}/{}_{}_rh{}.csv'.format(f, dataset, nli_metric, suffix), adv=True, prob='e')
            e_data_hr = load_metric_scores('../results/adv_scores{}/{}_{}_hr{}.csv'.format(f, dataset,nli_metric, suffix), adv=True, prob='e')
            n_data_rh = load_metric_scores('../results/adv_scores{}/{}_{}_rh{}.csv'.format(f, dataset, nli_metric, suffix), adv=True,
                                           prob='n')
            n_data_hr = load_metric_scores('../results/adv_scores{}/{}_{}_hr{}.csv'.format(f, dataset, nli_metric, suffix), adv=True,
                                           prob='n')
            c_data_rh = load_metric_scores('../results/adv_scores{}/{}_{}_rh{}.csv'.format(f, dataset, nli_metric, suffix), adv=True,
                                           prob='c')
            c_data_hr = load_metric_scores('../results/adv_scores{}/{}_{}_hr{}.csv'.format(f, dataset, nli_metric, suffix), adv=True,
                                           prob='c')

            r = defaultdict(lambda: defaultdict( lambda: defaultdict(dict)))
            for direction in ['rh', 'hr', 'avg']:
                if direction == 'rh':
                    e_data, n_data, c_data = e_data_rh, n_data_rh, c_data_rh
                elif direction == 'hr':
                    e_data, n_data, c_data = e_data_hr, n_data_hr, c_data_hr
                else:
                    e_data, n_data, c_data = defaultdict(lambda : defaultdict(list)),\
                                             defaultdict(lambda: defaultdict(list)),\
                                                defaultdict(lambda: defaultdict(list))
                    for error in e_data_rh.keys():
                        for score in ['score', 'score_adv']:
                            e_data[error][score] = [(s1+s2)/2 for s1, s2 in zip(e_data_rh[error][score], e_data_hr[error][score])]
                            n_data[error][score] = [(s1+s2)/2 for s1, s2 in zip(n_data_rh[error][score], n_data_hr[error][score])]
                            c_data[error][score] = [(s1+s2)/2 for s1, s2 in zip(c_data_rh[error][score], c_data_hr[error][score])]
                            assert len(e_data[error][score]) == len(e_data_rh[error][score])
                # e
                r[direction]['e'] = e_data
                for error in e_data_rh.keys():
                    # -c
                    r[direction]['-c'][error]['score'] = [-c for c in c_data[error]['score']]
                    r[direction]['-c'][error]['score_adv'] = [-c for c in c_data[error]['score_adv']]

                    # e-n
                    r[direction]['e-n'][error]['score'] = [e-n for e,n in zip(e_data[error]['score'], n_data[error]['score'])]
                    r[direction]['e-n'][error]['score_adv'] = [e-n for e,n in zip(e_data[error]['score_adv'], n_data[error]['score_adv'])]

                    # e-c
                    r[direction]['e-c'][error]['score'] = [e - c for e, c in
                                                           zip(e_data[error]['score'], c_data[error]['score'])]
                    r[direction]['e-c'][error]['score_adv'] = [e - c for e, c in
                                                               zip(e_data[error]['score_adv'], c_data[error]['score_adv'])]

                    # e-n-2c
                    r[direction]['e-n-2c'][error]['score'] = [e - n - 2*c for e, n, c in
                                                           zip(e_data[error]['score'], n_data[error]['score'], c_data[error]['score'])]
                    r[direction]['e-n-2c'][error]['score_adv'] = [e - n - 2*c for e, n, c in
                                                               zip(e_data[error]['score_adv'], n_data[error]['score_adv'], c_data[error]['score_adv'])]
        elif dataset == 'Rank19':
            # Rank19
            rh_data = load_metric_scores(f'../results/adv_scores/Rank19_{nli_metric}_rh.pkl', adv=False)
            hr_data = load_metric_scores(f'../results/adv_scores/Rank19_{nli_metric}_hr.pkl', adv=False)
            e_data_rh = rh_data['e']
            e_data_hr = hr_data['e']
            n_data_rh = rh_data['n']
            n_data_hr = hr_data['n']
            c_data_rh = rh_data['c']
            c_data_hr = hr_data['c']
            r = defaultdict(lambda: defaultdict(dict))
            for direction in ['rh', 'hr', 'avg']:
                if direction == 'rh':
                    e_data, n_data, c_data = e_data_rh, n_data_rh, c_data_rh
                elif direction == 'hr':
                    e_data, n_data, c_data = e_data_hr, n_data_hr, c_data_hr
                else:
                    e_data, n_data, c_data = {}, {}, {}
                    e_data['scores'] = [np.mean([s1,s2]) for s1, s2 in zip(e_data_rh['scores'], e_data_hr['scores'])]
                    e_data['scores_in'] = [np.mean([s1,s2]) for s1, s2 in zip(e_data_rh['scores_in'], e_data_hr['scores_in'])]
                    n_data['scores'] = [np.mean([s1, s2]) for s1, s2 in zip(n_data_rh['scores'], n_data_hr['scores'])]
                    n_data['scores_in'] = [np.mean([s1, s2]) for s1, s2 in zip(n_data_rh['scores_in'], n_data_hr['scores_in'])]
                    c_data['scores'] = [np.mean([s1, s2]) for s1, s2 in zip(c_data_rh['scores'], c_data_hr['scores'])]
                    c_data['scores_in'] = [np.mean([s1, s2]) for s1, s2 in zip(c_data_rh['scores_in'], c_data_hr['scores_in'])]
                # e
                r[direction]['e'] = e_data

                for k in ['scores', 'scores_in']:
                    # -c
                    r[direction]['-c'][k] = [-c for c in c_data[k]]
                    # e-n
                    r[direction]['e-n'][k] = [e-n for e,n in zip(e_data[k], n_data[k])]
                    # e-c
                    r[direction]['e-c'][k] = [e-c for e,c in zip(e_data[k], c_data[k])]
                    # e-n-2c
                    r[direction]['e-n-2c'][k] = [e - n - 2*c for e, n, c in zip(e_data[k], n_data[k], c_data[k])]

        else:
            raise ValueError('No such dataset.')

    else:
        # MT
        if 'wmt' in dataset:
            level = 'sys' if dataset in ['wmt20', 'wmt21.news'] else 'seg'

            e_data_rh = load_metric_scores('../results/{}_{}_scores/{}_rh_e_scores.pkl'.format(dataset, level, nli_metric), adv=False)
            e_data_hr = load_metric_scores('../results/{}_{}_scores/{}_hr_e_scores.pkl'.format(dataset, level, nli_metric), adv=False)
            n_data_rh = load_metric_scores('../results/{}_{}_scores/{}_rh_n_scores.pkl'.format(dataset, level, nli_metric), adv=False)
            n_data_hr = load_metric_scores('../results/{}_{}_scores/{}_hr_n_scores.pkl'.format(dataset, level, nli_metric), adv=False)
            c_data_rh = load_metric_scores('../results/{}_{}_scores/{}_rh_c_scores.pkl'.format(dataset, level, nli_metric), adv=False)
            c_data_hr = load_metric_scores('../results/{}_{}_scores/{}_hr_c_scores.pkl'.format(dataset, level, nli_metric), adv=False)

            if level == 'seg' and 'mqm' not in dataset:
                r = defaultdict(lambda: defaultdict(dict))
            else:
                r = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            for direction in ['rh', 'hr', 'avg']:
                if direction == 'rh':
                    e_data, n_data, c_data = e_data_rh, n_data_rh, c_data_rh
                elif direction == 'hr':
                    e_data, n_data, c_data = e_data_hr, n_data_hr, c_data_hr
                else:
                    if level == 'seg' and 'mqm' not in dataset:
                        e_data, n_data, c_data = {}, {}, {}
                        for lp, scores in e_data_rh.items():
                            e_data[lp] = [(s1+s2)/2 for s1, s2, in zip(e_data_rh[lp], e_data_hr[lp])]
                            n_data[lp] = [(s1+s2)/2 for s1, s2, in zip(n_data_rh[lp], n_data_hr[lp])]
                            c_data[lp] = [(s1+s2)/2 for s1, s2, in zip(c_data_rh[lp], c_data_hr[lp])]
                    else:
                        e_data, n_data, c_data = defaultdict(dict), defaultdict(dict), defaultdict(dict)
                        for lp, system_scores in e_data_rh.items():
                            for system, scores in system_scores.items():
                                e_data[lp][system] = [(s1+s2)/2 for s1, s2, in zip(e_data_rh[lp][system], e_data_hr[lp][system])]
                                n_data[lp][system] = [(s1 + s2) / 2 for s1, s2, in zip(n_data_rh[lp][system], n_data_hr[lp][system])]
                                c_data[lp][system] = [(s1 + s2) / 2 for s1, s2, in zip(c_data_rh[lp][system], c_data_hr[lp][system])]

                if level == 'seg' and 'mqm' not in dataset:
                    # e
                    r[direction]['e'] = e_data
                    for lp, _ in e_data.items():
                        # -c
                        r[direction]['-c'][lp] = [-c for c in c_data[lp]]
                        # e-n
                        r[direction]['e-n'][lp] = [e-n for e,n in zip(e_data[lp], n_data[lp])]
                        # e-c
                        r[direction]['e-c'][lp] = [e-c for e,c in zip(e_data[lp], c_data[lp])]
                        # e-n-2c
                        r[direction]['e-n-2c'][lp] = [e - n - 2 * c for e, n, c in zip(e_data[lp], n_data[lp], c_data[lp])]
                else:
                    r[direction]['e'] = e_data
                    for lp, system_scores in e_data.items():
                        for system, _ in system_scores.items():
                            # -c
                            r[direction]['-c'][lp][system] = [-c for c in c_data[lp][system]]
                            # e-n
                            r[direction]['e-n'][lp][system] = [e - n for e, n in zip(e_data[lp][system], n_data[lp][system])]
                            # e-c
                            r[direction]['e-c'][lp][system] = [e - c for e, c in zip(e_data[lp][system], c_data[lp][system])]
                            # e-n-2c
                            r[direction]['e-n-2c'][lp][system] = [e - n - 2 * c for e, n, c in
                                                         zip(e_data[lp][system], n_data[lp][system], c_data[lp][system])]
        else:  # sum
            if dataset in ['summ', 'realsumm']:
                if dataset == 'realsumm' or use_article:
                    agg = 'None'
                e_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_e_{agg}{suffix}.pkl", adv=False)
                e_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_e_{agg}{suffix}.pkl", adv=False)
                n_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_n_{agg}{suffix}.pkl", adv=False)
                n_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_n_{agg}{suffix}.pkl", adv=False)
                c_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_c_{agg}{suffix}.pkl", adv=False)
                c_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_c_{agg}{suffix}.pkl", adv=False)
                r = defaultdict(lambda: defaultdict(list))
                for direction in ['rh', 'hr', 'avg']:
                    if direction == 'rh':
                        e_data, n_data, c_data = e_data_rh, n_data_rh, c_data_rh
                    elif direction == 'hr':
                        e_data, n_data, c_data = e_data_hr, n_data_hr, c_data_hr
                    else:
                        e_data = [(s1+s2)/2 for s1,s2 in zip(e_data_rh, e_data_hr)]
                        n_data = [(s1+s2)/2 for s1,s2 in zip(n_data_rh, n_data_hr)]
                        c_data = [(s1+s2)/2 for s1,s2 in zip(c_data_rh, c_data_hr)]
                    r[direction]['e'] = e_data
                    r[direction]['-c'] = [-c for c in c_data]
                    r[direction]['e-n'] = [e-n for e, n in zip(e_data, n_data)]
                    r[direction]['e-c'] = [e-c for e, c in zip(e_data, c_data)]
                    r[direction]['e-n-2c'] = [e-n-2*c for e, n, c in zip(e_data, n_data, c_data)]


            else:
                raise ValueError('No such dataset.')

    return r


def print_accuracy(metric_hash, dataset, errors, acc):
    cols = ['metric', 'dataset', 'measurement'] + errors + ['average']
    cols = ','.join(cols) + '\n'

    accs = [str(acc[k]) for k in errors]

    values = [metric_hash, dataset, 'accuracy'] + accs + \
             [str(np.mean(list(acc.values())))]

    values = ','.join(values) + '\n'
    print(cols+values)

def print_pearson(metric, pearson_dict):
    col = ','.join(['metric', 'correlation']+list(pearson_dict.keys())) + ',avg\n'
    s = col
    scores = list(pearson_dict.values())
    scores.append(np.mean(list(pearson_dict.values())))
    scores = [str(i) for i in scores]
    s += ','.join([metric, 'pearson']+scores)+'\n'
    print(s)

def evaluate(metric_data, dataset, adv=True, ifprint=False):
    r = copy.deepcopy(metric_data)
    if adv:
        if dataset != 'Rank19':
            if 'rh' in metric_data.keys():
                for direction, d_data in metric_data.items():
                    for strategy, s_data in d_data.items():
                        for error, data in s_data.items():
                            r[direction][strategy][error] = calculate_accuracy_and_kendall(data['score'], data['score_adv'])[0]
                        metric_hash = '{}_{}'.format(direction, strategy)
                        erros = list(r[direction][strategy].keys())
                        if ifprint:
                            print_accuracy(metric_hash, dataset, erros, r[direction][strategy])
            else:
                # for combined metric evaluation r.keys: nli weight
                for w, w_data in r.items():
                    for error, data in w_data.items():
                        r[w][error] = calculate_accuracy_and_kendall(data['score'], data['score_adv'])[0]
                    if ifprint:
                        print_accuracy(str(w), dataset, list(r[w].keys()), r[w])
        else:
            if 'rh' in metric_data.keys():
                for direction, d_data in metric_data.items():
                    for strategy, s_data in d_data.items():
                        r[direction][strategy] = calculate_accuracy_and_kendall(s_data['scores'], s_data['scores_in'])[0]
                        metric_hash = '{}_{}'.format(direction, strategy)
                        if ifprint:
                            print(f'metric,accuracy\n{metric_hash},{r[direction][strategy]}\n')
            else:
                for w, w_data in r.items():
                    r[w] = calculate_accuracy_and_kendall(w_data['scores'], w_data['scores_in'])[0]
                    if ifprint:
                        print(f'metric,accuracy\n{w},{r[w]}\n')
    else:
        # MT
        if 'wmt' in dataset:
            level = 'seg' if dataset not in ['wmt20', 'wmt21.news'] else 'sys'
            # for nli metric evaluation r.keys: rh...
            if 'rh' in metric_data.keys():
                # wmt15-17
                if dataset not in ['wmt20', 'wmt21.news', 'wmt20_mqm', 'wmt21.news_mqm']:
                    human_data = read_human_scores() if dataset == 'wmt17' else None
                    for direction, d_data in metric_data.items():
                        for strategy, s_data in d_data.items():
                            for lp, scores in s_data.items():
                                if dataset == 'wmt17':
                                    _, _, _, human = load_evaluation_data_17(human_data, lp)
                                else:
                                    _, _, _, human = load_evaluation_data_1516(lp, int('20'+dataset[-2:]))
                                r[direction][strategy][lp] = pearsonr(scores, human)[0]
                            metric_hash = '{}_{}'.format(direction, strategy)
                            if ifprint:
                                print_pearson(metric_hash, r[direction][strategy], dataset)

                # wmt20-21
                else:
                    for direction, d_data in metric_data.items():
                        for strategy, s_data in d_data.items():
                            for lp, system_scores in s_data.items():
                                evs = me_data.EvalSet(dataset, lp) if 'mqm' not in dataset else me_data.EvalSet(dataset.split('_mqm')[0], lp)
                                if level == 'sys':
                                    # system-level
                                    gold_scores = evs.Scores('sys', 'wmt-z')
                                    system_scores = {system: [np.mean(scores)] for system, scores in system_scores.items()}
                                    corr = evs.Correlation(gold_scores, system_scores, list(system_scores.keys()))
                                    r[direction][strategy][lp] = corr.Pearson()[0]
                                else:
                                    # segment-level
                                    if lp != 'zh-en':
                                        continue
                                    gold_scores = evs.Scores('seg', 'mqm')
                                    sys_names = set(gold_scores) - evs.human_sys_names
                                    corr = evs.Correlation(gold_scores, system_scores, sys_names)
                                    r[direction][strategy][lp] = corr.Pearson()[0]
                            metric_hash = '{}_{}'.format(direction, strategy)
                            if ifprint:
                                print_pearson(metric_hash, r[direction][strategy], dataset)
            else:
                # for combined metric evaluation r.keys: nli weight
                human_data = read_human_scores() if dataset == 'wmt17' else None
                if dataset in ['wmt15', 'wmt16', 'wmt17']:
                    for w, lp_data in r.items():
                        for lp, scores in lp_data.items():
                            if dataset == 'wmt17':
                                _, _, _, human = load_evaluation_data_17(human_data, lp)
                            else:
                                _, _, _, human = load_evaluation_data_1516(lp, int('20' + dataset[-2:]))
                            r[w][lp] = pearsonr(scores, human)[0]
                        if ifprint:
                            print_pearson(str(w), r[w], dataset)
                else:
                    # system-level
                    for w, lp_data in metric_data.items():
                        for lp, system_scores in lp_data.items():
                            evs = me_data.EvalSet(dataset, lp) if 'mqm' not in dataset else me_data.EvalSet(dataset.split('_mqm')[0], lp)
                            if level == 'sys':
                                # system-level
                                gold_scores = evs.Scores('sys', 'wmt-z')
                                system_scores = {system: [np.mean(scores)] for system, scores in system_scores.items()}
                                corr = evs.Correlation(gold_scores, system_scores, sys_names)
                                r[w][lp] = corr.Pearson()[0]
                            else:
                                # segment-level
                                if lp != 'zh-en':
                                    continue
                                gold_scores = evs.Scores('seg', 'mqm')
                                sys_names = set(gold_scores) - evs.human_sys_names
                                corr = evs.Correlation(gold_scores, system_scores, sys_names)
                                r[w][lp] = corr.Pearson()[0]
                        if ifprint:
                            print_pearson(str(w), r[w], dataset)
        else:
            if 'rh' in metric_data.keys():  # for nli metrics
                if dataset == 'summ':
                    args = SimpleNamespace(**{'dataset': dataset, 'data_dir': 'datasets', 'aggregate': None})
                    eval = SummEval(args=args, load_doc=False)
                    data_df = eval.load_data_summ()
                    r = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                    for direction in metric_data.keys():
                        for strategy in metric_data['rh'].keys():
                            for c in ['coherence', 'consistency', 'fluency', 'relevance']:
                                metric_scores = metric_data[direction][strategy]
                                data_df['metric_scores'] = metric_scores
                                data = data_df.groupby('system').mean()
                                r[c][direction][strategy] = kendalltau(list(data['metric_scores']), list(data[f'expert_{c}']))[
                                    0]
                            coh, con, flu, rev = r['coherence'][direction][strategy], r['consistency'][direction][strategy], \
                                                 r['fluency'][direction][strategy], r['relevance'][direction][strategy]
                            avg = np.mean([coh, con, flu, rev])
                            if ifprint:
                                print('metric, dataset, coherence, consistency, fluency, relevance, avg')
                                print(f"{direction}-{strategy}, {dataset}, {coh}, {con}, {flu}, {rev}, {avg}\n")
                elif dataset == 'realsumm':
                    r = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                    args = SimpleNamespace(**{'dataset': dataset, 'data_dir': 'datasets', 'aggregate': None})
                    eval = RealSumm(args=args, load_doc=False)
                    data_df = eval.load_data_realsum()
                    for direction in metric_data.keys():
                        for strategy in metric_data['rh'].keys():
                            metric_scores = metric_data[direction][strategy]
                            data_df['metric_scores'] = metric_scores
                            doc_ids = set(data_df['doc_id'])
                            corr_list = []
                            for doc_id in doc_ids:
                                doc_data = data_df[data_df.doc_id == doc_id]
                                metric_scores = doc_data['metric_scores']
                                human_scores = doc_data['human_score']
                                corr, p = pearsonr(metric_scores, human_scores)
                                if p <= 0.05:
                                    corr_list.append(corr)
                            r['summary'][direction][strategy] = np.mean(corr_list)
                            data = data_df.groupby('system').mean()
                            corr = pearsonr(data['human_score'], data['metric_scores'])[0]
                            r['system'][direction][strategy] = corr
                            if ifprint:
                                print("direction,strategy,dataset,level,pearson\n")
                                print(f"{direction},{strategy},{dataset},summary,{r['summary'][direction][strategy]}")
                                print(f"{direction},{strategy},{dataset},system,{r['system'][direction][strategy]}")
                else:
                    print(dataset)
                    raise ValueError('No such dataset.')
            else:  # for combined metrics key:w1
                if dataset == 'summ':
                    args = SimpleNamespace(**{'dataset': dataset, 'data_dir': 'datasets', 'aggregate': None})
                    eval = SummEval(args=args, load_doc=False)
                    data_df = eval.load_data_summ()
                    r = defaultdict(dict)
                    for w in metric_data.keys():
                        metric_scores = metric_data[w]
                        data_df['metric_scores'] = metric_scores
                        data = data_df.groupby('system').mean()
                        for c in ['coherence', 'consistency', 'fluency', 'relevance']:
                            r[w][c] = kendalltau(list(data['metric_scores']), list(data[f'expert_{c}']))[0]
                        coh, con, flu, rev = r[w]['coherence'], r[w]['consistency'], \
                                             r[w]['fluency'], r[w]['relevance']
                        avg = np.mean([coh, con, flu, rev])
                        if ifprint:
                            print('metric, dataset, coherence, consistency, fluency, relevance, avg')
                            print(f"{w}, {dataset}, {coh}, {con}, {flu}, {rev}, {avg}\n")
                elif dataset == 'realsumm':
                    args = SimpleNamespace(**{'dataset': dataset, 'data_dir': 'datasets', 'aggregate': None})
                    eval = RealSumm(args=args, load_doc=False)
                    data_df = eval.load_data_realsum()
                    r = defaultdict(dict)
                    for w in metric_data.keys():
                        metric_scores = metric_data[w]
                        data_df['metric_scores'] = metric_scores
                        doc_ids = set(data_df['doc_id'])
                        corr_list = []
                        for doc_id in doc_ids:
                            doc_data = data_df[data_df.doc_id == doc_id]
                            metric_scores = doc_data['metric_scores']
                            human_scores = doc_data['human_score']
                            corr, p = pearsonr(metric_scores, human_scores)
                            if p <= 0.05:
                                corr_list.append(corr)
                        r['summary'][w] = np.mean(corr_list)
                        data = data_df.groupby('system').mean()
                        corr = pearsonr(data['human_score'], data['metric_scores'])[0]
                        r['system'][w] = corr
                        if ifprint:
                            print("nli_weight,dataset,level,pearson\n")
                            print(f"{w},{dataset},summary,{np.mean(corr_list)}")
                            print(f"{w},{dataset},system,{corr}")

                else:
                    raise ValueError('No such dataset.')
    return r

def nomalize_scores(scores, norm='min_max'):
    if isinstance(scores, tuple):
        assert isinstance(scores[0], list)
        assert isinstance(scores[1], list)
        all_scores = scores[0]+scores[1]
        if norm=='min_max':
            normalized_scores = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
        else:
            normalized_scores = [(x - np.mean(all_scores))/np.std(all_scores) for x in all_scores]
        assert len(scores[0]) == len(normalized_scores[:len(scores[0])])
        return normalized_scores[:len(scores[0])], normalized_scores[len(scores[0]):]
    elif isinstance(scores, dict):
        all_scores = []
        positions = {}
        start = 0
        for k, v in scores.items():
            if isinstance(v, torch.Tensor):
                v = list(v.numpy())
            v = list(v)
            all_scores += v
            positions[k] = (start, start+len(v))
            start += len(v)
        if norm == 'min_max':
            normalized_scores = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
        else:
            normalized_scores = [(x - np.mean(all_scores)) / np.std(all_scores) for x in all_scores]
        r = {}
        for k in scores.keys():
            r[k] = normalized_scores[positions[k][0]: positions[k][1]]
            assert len(r[k]) == len(scores[k])
        return r
    else:
        if norm == 'min_max':
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            return [(x - np.mean(scores))/np.std(scores) for x in scores]

def add_scores(scores1, scores2, w1, w2):
    return [w1 * s1 + w2 * s2 for s1, s2 in zip(scores1, scores2)]

def combine_nli_and_metric(nli_metric, metric, dataset, method=add_scores, adv=True, norm='min_max', agg='max', use_article=False, direction='avg', strategy='e', fluency=False):
    if adv:
        if dataset != 'Rank19':
            # load nli data
            nli_data = combine_nli(nli_metric, dataset, adv=adv, use_article=use_article, fluency=fluency)
            nli_data = nli_data[direction][strategy]
            # load metric data
            suffix = '' if not use_article else '_use_article'
            f = '_fluency' if fluency else ''
            metric_data = load_metric_scores('../results/adv_scores{}/{}_{}{}.csv'.format(f, dataset, metric, suffix), adv=adv)
            r = defaultdict(lambda: defaultdict(dict))
            for error, metric_scores in metric_data.items():
                nli_scores = nli_data[error]
                nli_scores['score'], nli_scores['score_adv'] = nomalize_scores((nli_scores['score'], nli_scores['score_adv']), norm=norm)
                metric_scores['score'], metric_scores['score_adv'] = nomalize_scores((metric_scores['score'], metric_scores['score_adv']), norm=norm)

                for w1 in np.arange(0, 1.1, 0.1):
                    # w1 nli weight; w2 metric weight
                    #w1 = w1
                    w2 = 1-w1
                    w1 = float('%.1f' % w1)
                    w2 = float('%.1f' % w2)
                    combined_scores = method(nli_scores['score'], metric_scores['score'], w1, w2)
                    combined_scores_adv = method(nli_scores['score_adv'], metric_scores['score_adv'], w1, w2)

                    assert len(combined_scores) == len(combined_scores_adv), '{}-{}'.format(len(combined_scores), len(combined_scores_adv))
                    r[w1][error] = {'score': combined_scores, 'score_adv': combined_scores_adv}
        else:
            # Rank19
            r = defaultdict(dict)
            nli_data = combine_nli(nli_metric, dataset, adv=True)
            nli_data = nli_data[direction][strategy]
            metric_data = load_metric_scores(f'../results/adv_scores/Rank19_{metric}.pkl', adv=False)
            nli_scores, metric_scores = {}, {}
            nli_scores['scores'], nli_scores['scores_in'] = nomalize_scores((list(nli_data['scores']), list(nli_data['scores_in'])), norm=norm)
            metric_scores['scores'], metric_scores['scores_in'] = nomalize_scores((metric_data['scores'], metric_data['scores_in']), norm=norm)

            for w1 in np.arange(0, 1.1, 0.1):
                w2 = 1-w1
                w1 = float('%.1f' % w1)
                w2 = float('%.1f' % w2)
                combined_scores = method(nli_scores['scores'], metric_scores['scores'], w1, w2)
                combined_scores_in = method(nli_scores['scores_in'], metric_scores['scores_in'], w1, w2)
                assert len(combined_scores) == len(combined_scores_in), '{}-{}-{}'.format(w1, len(combined_scores), len(combined_scores_in))
                r[w1] = {'scores': combined_scores, 'scores_in': combined_scores_in}

    else:
        # MT
        if 'wmt' in dataset:
            level_file = 'seg' if dataset not in ['wmt21.news', 'wmt20'] else 'sys'
            nli_data = combine_nli(nli_metric, dataset, adv=adv)
            nli_data = nli_data[direction][strategy]
            metric_data = load_metric_scores('../results/{}_{}_scores/{}_scores.pkl'.format(dataset, level_file, metric), adv=adv)

            r = defaultdict(dict)
            if level_file == 'seg' and 'mqm' not in dataset:
                for lp, metric_scores in metric_data.items():
                    nli_scores = nli_data[lp]
                    nli_scores = nomalize_scores(nli_scores, norm=norm)
                    if isinstance(metric_scores, torch.Tensor):
                        metric_scores = metric_scores.numpy()
                    metric_scores = nomalize_scores(metric_scores, norm=norm)
                    for w1 in np.arange(0, 1.1, 0.1):
                        # w1 nli weight; w2 metric weight
                        w2 = 1 - w1
                        w1 = float('%.1f' % w1)
                        w2 = float('%.1f' % w2)
                        combined_scores = method(nli_scores, metric_scores, w1, w2)
                        r[w1][lp] = combined_scores
            else:
                #
                r = defaultdict(lambda: defaultdict(dict))
                for lp, system_data in metric_data.items():
                    # normalize scores per lp
                    tmp_metric_data = nomalize_scores(system_data, norm=norm)
                    tmp_nli_data = nomalize_scores(nli_data[lp], norm=norm)
                    for system in system_data.keys():
                        nli_scores = tmp_nli_data[system]
                        metric_scores = tmp_metric_data[system]
                        assert len(nli_scores) == len(metric_scores)
                        for w1 in np.arange(0, 1.1, 0.1):
                            # w1 nli weight; w2 metric weight
                            w2 = 1 - w1
                            w1 = float('%.1f' % w1)
                            w2 = float('%.1f' % w2)
                            combined_scores = method(nli_scores, metric_scores, w1, w2)
                            r[w1][lp][system] = combined_scores
        else: # sum
            suffix = '' if not use_article else '_use_article'
            if dataset in ['summ', 'realsumm']:
                nli_data = combine_nli(nli_metric, dataset, adv=False, agg=agg, use_article=use_article)
                if dataset == 'summ':
                    metric_data = load_metric_scores(f"../results/{dataset}_scores/{metric}_{agg}{suffix}.pkl", adv=False)
                else:
                    metric_data = load_metric_scores(f"../results/{dataset}_scores/{metric}_None{suffix}.pkl",
                                                     adv=False)
                nli_data = nli_data[direction][strategy]
                r = {}
                nli_scores = nomalize_scores(nli_data, norm=norm)
                metric_scores = nomalize_scores(metric_data, norm=norm)
                assert len(nli_scores) == len(metric_scores)
                for w1 in np.arange(0, 1.1, 0.1):
                    w1 = float('%.1f' % w1)
                    w2 = float('%.1f' % (1 - w1))
                    if 'Disco' in metric:
                        metric_scores = [-s for s in metric_scores]
                    combined_scores = method(nli_scores, metric_scores, w1, w2)
                    r[w1] = combined_scores

            else:
                raise ValueError('No such dataset.')
    return r


def calculate_relative_change(x, y):
    return (y/x - 1) * 100


def prepare_improvements_df_mt(setup='all', norm='min_max'):
    setup_ori = setup
    if setup == 'all':
        setups = ['ref', 'src']
    else:
        setups = [setup]
    metric_r, nli_weight, dataset, type, improvement, ori, current, set_up = [], [], [], [], [], [], [], []
    ori_nli, improvement_nli = [], []
    for setup in setups:
        metrics = metrics_dict[setup]
        nlis = nli_dict[setup]
        mt_datasets = datasets_dict[setup]['mt-seg'] + datasets_dict[setup]['mt-sys']
        adv_datasets = datasets_dict[setup]['adv']

        for nli in nlis:
            for metric in metrics:
                # mt
                for mt_dataset in mt_datasets:
                    if mt_dataset in datasets_dict[setup]['mt-sys'] and 'WMD' in metric:
                        continue
                    metric_data = combine_nli_and_metric(nli, metric, mt_dataset, adv=False, norm=norm)
                    results = evaluate(metric_data, mt_dataset, adv=False)
                    ori_r = np.mean(list(results[0.0].values()))
                    ori_r_nli = np.mean(list(results[1.0].values()))
                    for w, r in results.items():
                        if w == 0.0 or w == 1.0:
                            continue
                        combined_r = np.mean(list(r.values()))
                        im = calculate_relative_change(ori_r, combined_r)
                        im_nli = calculate_relative_change(ori_r_nli, combined_r)
                        metric_r.append(metric)
                        nli_weight.append(w)
                        dataset.append(mt_dataset)
                        type.append('mt-seg')
                        improvement.append(im)
                        improvement_nli.append(im_nli)
                        ori.append(ori_r)
                        ori_nli.append(ori_r_nli)
                        current.append(combined_r)
                        set_up.append(setup)
                # adv
                for adv_dataset in adv_datasets:
                    if 'COMET' in metric and adv_dataset not in ['wmt20_google-de', 'xpaws-de'] and setup == 'ref':
                        continue
                    metric_data = combine_nli_and_metric(nli, metric, adv_dataset, adv=True, norm=norm)
                    results = evaluate(metric_data, adv_dataset, adv=True)
                    # original metric results
                    ori_r = np.mean(list(results[0.0].values()))
                    ori_r_nli = np.mean(list(results[1.0].values()))
                    for w, r in results.items():
                        if w == 0.0 or w == 1.0:
                            continue
                        combined_r = np.mean(list(r.values()))
                        im = calculate_relative_change(ori_r, combined_r)
                        im_nli = calculate_relative_change(ori_r_nli, combined_r)
                        metric_r.append(metric)
                        nli_weight.append(w)
                        dataset.append(adv_dataset)
                        type.append('adv')
                        improvement.append(im)
                        improvement_nli.append(im_nli)
                        ori.append(ori_r)
                        ori_nli.append(ori_r_nli)
                        current.append(combined_r)
                        set_up.append(setup)
        assert len(nli_weight) == len(metric_r)
    data = pd.DataFrame()
    data['setup'] = set_up
    data['metric'] = metric_r
    data['nli_weight'] = nli_weight
    data['dataset'] = dataset
    data['type'] = type
    data['improvement'] = improvement
    data['improvement_nli'] = improvement_nli
    data['ori'] = ori
    data['ori_nli'] = ori_nli
    data['current'] = current
    print(data)
    data.to_csv('../results/{}_improvements_data_mt.csv'.format(setup_ori), index=False)
    return data


def prepare_improvement_table_data(setup='all', norm='min_max', level='seg'):
    setup_ori = setup
    if setup == 'all':
        setups = ['ref', 'src']
    else:
        setups = [setup]
    data = defaultdict(list)
    for setup in setups:
        metrics = metrics_dict[setup]
        nlis = nli_dict[setup]

        mt_datasets = datasets_dict[setup]['mt-{}'.format(level)]
        adv_datasets = datasets_dict[setup]['adv']

        for nli in nlis:
            for metric in metrics:
                # mt
                for mt_dataset in mt_datasets:
                    if mt_dataset in datasets_dict[setup]['mt-sys'] and 'WMD' in metric:
                        continue
                    metric_data = combine_nli_and_metric(nli, metric, mt_dataset, adv=False, norm=norm)
                    results = evaluate(metric_data, mt_dataset, adv=False)
                    ori_r = np.mean(list(results[0.0].values()))
                    for w, r in results.items():
                        if w == 1.0:
                            continue
                        combined_r = np.mean(list(r.values()))
                        im = calculate_relative_change(ori_r, combined_r)
                        if w == 0.0:
                            data['type'].append('MT')
                            data['dataset'].append(mt_dataset)
                            data['metric'].append(metric)
                            data['ori'].append(ori_r)
                            continue
                        data['combined-'+str(w)].append(combined_r)
                        data['improvement-'+str(w)].append(im)

                # adv
                for adv_dataset in adv_datasets:
                    if 'COMET' in metric and adv_dataset not in ['wmt20_google-de', 'xpaws-de'] and setup == 'ref':
                        continue
                    metric_data = combine_nli_and_metric(nli, metric, adv_dataset, adv=True, norm=norm)
                    results = evaluate(metric_data, adv_dataset, adv=True)
                    ori_r = np.mean(list(results[0.0].values()))
                    for w, r in results.items():
                        if w == 1.0:
                            continue
                        combined_r = np.mean(list(r.values()))
                        im = calculate_relative_change(ori_r, combined_r)
                        if w == 0.0:
                            data['type'].append('Adv.')
                            data['dataset'].append(adv_dataset)
                            data['metric'].append(metric)
                            data['ori'].append(ori_r)
                            continue
                        data['combined-' + str(w)].append(combined_r)
                        data['improvement-' + str(w)].append(im)
    data = pd.DataFrame.from_dict(data)
    print(data)
    data.to_csv('../results/{}_improvement_table_data_{}_mt.csv'.format(setup_ori, level), index=False)
    return data


def plot_trade_off(nli1_data, nli2_data, title, setup, level):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(8, 5))
    metrics = list(nli1_data['adv'].keys())
    # nli1
    ax1 = fig.add_subplot(1, 2, 1)
    for i, metric in enumerate(metrics):
        x = nli1_data['adv'][metric]
        y = nli1_data['mt'][metric]

        ax1.scatter(x, y, s=5)
        ax1.scatter(x[0], y[0], marker='*', color='r')
        ax1.scatter(x[-1], y[-1], marker='x', color='r')
        if i == 0:
            ax1.scatter(x[0], y[0], marker='*', color='r', label='Ori. Metric')
            ax1.scatter(x[-1], y[-1], marker='x', color='r', label='NLI Metric')
        # label
        if 'SentSim' not in metric:
            if 'XMover' not in metric and 'BART' not in metric:
                metric = metric.split('_')[0]
            else:
                if 'XMover' in metric:
                    metric = 'XMover({})'.format('CLP' if 'CLP' in metric else 'UMD')
                else:
                    metric = 'BARTScore-P' if metric == 'BARTScore_bart-large-cnn' else 'BARTScore-F'
        else:
            if 'BERT' in metric:
                metric = 'SentSim(BERTS)'
            else:
                metric = 'SentSim(WMD)'

        ax1.plot(x, y, linewidth=0.8, label=metric)

    ax1.set_ylabel('WMT(pearson)')
    ax1.set_xlabel('Adv.(accuracy)')
    ax1.set_title('NLI-R') if setup == 'ref' else ax1.set_title('XNLI-R')

    #nli2
    ax2 = fig.add_subplot(1, 2, 2)

    for i, metric in enumerate(metrics):
        x = nli2_data['adv'][metric]
        y = nli2_data['mt'][metric]

        ax2.scatter(x, y, s=5)
        ax2.scatter(x[0], y[0], marker='*', color='r')
        ax2.scatter(x[-1], y[-1], marker='x', color='r')
        if i == 0:
            ax2.scatter(x[0], y[0], marker='*', color='r', label='Ori. Metric')
            ax2.scatter(x[-1], y[-1], marker='x', color='r', label='NLI Metric')

        # label
        if 'SentSim' not in metric:
            if 'XMover' not in metric and 'BART' not in metric:
                metric = metric.split('_')[0]
            else:
                if 'XMover' in metric:
                    metric = 'XMover({})'.format('CLP' if 'CLP' in metric else 'UMD')
                else:
                    metric = 'BARTScore-P' if metric == 'BARTScore_bart-large-cnn' else 'BARTScore-F1'
        else:
            if 'BERT' in metric:
                metric = 'SentSim(BERTS)'
            else:
                metric = 'SentSim(WMD)'

        ax2.plot(x, y, linewidth=0.8, label=metric)

    ax2.set_ylabel('WMT(pearson)')
    ax2.set_xlabel('Adv.(accuracy)')
    ax2.set_title('NLI-D') if setup == 'ref' else ax2.set_title('XNLI-D')

    if level == 'seg':
        ax1.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.show()
    #plt.savefig('../results/plot/{}-{}-{}-{}.png'.format(title, setup, level, norm), dpi=300, bbox_inches='tight')

def append_dict(dicts):
    r = defaultdict(list)
    for w in dicts[0].keys():
        for d in dicts:
            r[w].append(np.mean(list(d[w].values())))
    r = [np.mean(v) for _, v in r.items()]
    return r


def get_matrix(nli, dataset, adv=True, agg='max', use_article=False):
    print(f"{nli}--{dataset}")
    metric_data = combine_nli(nli, dataset, adv=adv, agg=agg, use_article=use_article)
    acc_dict = evaluate(metric_data, dataset, adv=adv, ifprint=False)
    if dataset not in ['summ', 'realsumm']:
        directions = list(acc_dict.keys()) if not use_article else ['rh']
        strategies = list(acc_dict['rh'].keys())
        aa = np.zeros((len(directions) + 1, len(strategies) + 1))
        for i in range(len(directions) + 1):
            for j in range(len(strategies) + 1):
                if j == len(strategies):
                    aa[i][j] = np.mean(aa[i, :-1])
                elif i == len(directions):
                    aa[i][j] = np.mean(aa[:-1, j])
                else:
                    if dataset != 'Rank19':
                        aa[i][j] = np.mean([v * 100 for _, v in acc_dict[directions[i]][strategies[j]].items()])
                    else:
                        aa[i][j] = acc_dict[directions[i]][strategies[j]] * 100
        matrix = aa
        s = f"\n{nli} -- {dataset} -- {agg} -- {use_article}\n"
        for i in range(len(aa)):
            s += ','.join(['%.1f' % a for a in aa[i, :]]) + '\n'

    elif dataset == 'summ':
        directions = list(acc_dict['coherence'].keys()) if not use_article else ['rh']
        strategies = list(acc_dict['coherence']['rh'].keys())
        matrix = {}
        s = ''
        for c in acc_dict.keys():
            aa = np.zeros((len(directions) + 1, len(strategies) + 1))
            for i in range(len(directions) + 1):
                for j in range(len(strategies) + 1):
                    if j == len(strategies):
                        aa[i][j] = np.mean(aa[i, :-1])
                    elif i == len(directions):
                        aa[i][j] = np.mean(aa[:-1, j])
                    else:
                        aa[i][j] = acc_dict[c][directions[i]][strategies[j]] * 100
            matrix[c] = aa
            s += f"\n{nli}-{agg}-{dataset}-{c}\n"
            for i in range(len(aa)):
                s += '\t'.join(['%.1f' % a for a in aa[i, :]])+'\n'

    elif dataset == 'realsumm':
        directions = list(acc_dict['system'].keys()) if not use_article else ['rh']
        strategies = list(acc_dict['system']['rh'].keys())
        matrix = {}
        s = ''
        for level in acc_dict.keys():
            aa = np.zeros((len(directions) + 1, len(strategies) + 1))
            for i in range(len(directions) + 1):
                for j in range(len(strategies) + 1):
                    if j == len(strategies):
                        aa[i][j] = np.mean(aa[i, :-1])
                    elif i == len(directions):
                        aa[i][j] = np.mean(aa[:-1, j])
                    else:
                        aa[i][j] = acc_dict[level][directions[i]][strategies[j]] * 100
            matrix[level] = aa
            s += f"\n{nli}-{dataset}-{level}\n"
            for i in range(len(aa)):
                s += ','.join(['%.1f' % a for a in aa[i, :]]) + '\n'
    else:
        raise ValueError(f'No such dataset: {dataset}.')

    if nli in ['NLI1Score_monolingual', 'NLI2Score_monolingual']:
        directions = ['r\u2192h', 'r\u2190h', 'r\u2194h']
    else:
        directions = ['s\u2192h', 's\u2190h', 's\u2194h']

    return matrix, directions, strategies

def get_best_position(matrix):
    return np.unravel_index(matrix.argmax(), matrix.shape)

def get_all_matrices(task = 'mt'):
    matrix_dict = defaultdict(lambda: defaultdict(list))
    if task == 'mt':
        for k1 in ['ref', 'src']:
            metrics = nli_dict[k1]
            for k2 in ['adv', 'mt-seg']:
                datasets = datasets_dict[k1][k2]
                adv = True if k2 == 'adv' else False
                for m in metrics:
                    for d in datasets:
                        print(f"{m}--{d}")
                        matrix, _, _ = get_matrix(m, d, adv=adv, agg='max')
                        if d in ['summ']:
                            matrix_2, _, _ = get_matrix(m, d, adv=adv, agg='mean')
                        if isinstance(matrix, np.ndarray):
                            matrix_dict[k1][k2].append(matrix)
                        else:
                            matrix_dict[k1][k2] += list(matrix.values())+list(matrix_2.values())
                assert len(matrix_dict[k1][k2]) == len(datasets_dict[k1][k2])*2 or \
                       len(matrix_dict[k1][k2]) == len(datasets_dict[k1][k2]) * 16
    # sum
    elif task == 'sum':
        metrics = nli_dict['ref']
        for use_article in [True, False]:
            # adv
            for dataset in datasets_dict_sum['adv']:
                if dataset == 'Rank19' and not use_article:
                    continue
                for m in metrics:
                        matrix, _, _ = get_matrix(m, dataset, adv=True, use_article=use_article)
                        matrix_dict[use_article]['adv'].append(matrix)
            # sum
            for dataset in datasets_dict_sum['sum']:
                for m in metrics:
                    if dataset == 'summ':
                        for agg in ['mean', 'max']:
                            if agg == 'max' and use_article:
                                continue
                            matrix, _, _ = get_matrix(m, dataset, adv=False, use_article=use_article, agg=agg)
                            for k, v in matrix.items():
                                matrix_dict[use_article]['sum'].append(v)
                    elif dataset == 'realsumm':
                        matrix, _, _ = get_matrix(m, dataset, adv=False, use_article=use_article)
                        for k, v in matrix.items():
                            matrix_dict[use_article]['sum'].append(v)
    return matrix_dict

def get_winning_matrices():
    all_matrices = get_all_matrices()
    r = defaultdict(lambda: defaultdict(dict))
    for k1 in ['ref', 'src']:
        for k2 in ['adv', 'mt-seg']:
            ma = np.zeros((3, 5))
            for matrix in all_matrices[k1][k2]:
                i, j = get_best_position(matrix)
                ma[i][j] += 1
            r[k1][k2]['matrix'] = ma
    return r

def print_wining_matrix(matrices):
    r = np.zeros((3, 5))
    for m in matrices:
        i, j = get_best_position(m)
        r[i][j] += 1
    print(r)
    return r

def make_table_sum(direction='avg', strategy='e', agg='max', use_article=False):
    final = defaultdict(dict)

    # sum
    for dataset in datasets_dict_sum['sum']+datasets_dict_sum['adv']:
        if dataset == 'Rank19' and not use_article:
            continue
        for i, metric in enumerate(sum_metrics_dict['ref'] if not use_article else sum_metrics_dict['src']):
            adv = True if dataset in datasets_dict_sum['adv'] else False

            metric_data = combine_nli_and_metric('NLI1Score_monolingual', metric, dataset, adv=adv, agg=agg, use_article=use_article,
                                                 direction=direction, strategy=strategy)
            r = evaluate(metric_data, dataset, adv=adv)
            if 0.0 in r.keys():
                if dataset in ['summ', 'Rank19']:
                    final[dataset][metric] = r[0.0]
                else:
                    final[dataset][metric] = np.mean(list(r[0.0].values()))
            else:
                final[dataset][metric] = {'summary': r['summary'][0.0], 'system': r['system'][0.0]}
            if i == 0:
                for nli in nli_dict['ref']:
                    metric_data = combine_nli(nli, dataset, adv=adv, agg=agg, use_article=use_article)
                    r = evaluate(metric_data, dataset, adv=adv)
                    if dataset == 'summ':
                        final[dataset][nli] = {'coherence': r['coherence'][direction][strategy],
                                               'consistency': r['consistency'][direction][strategy],
                                               'fluency': r['fluency'][direction][strategy],
                                               'relevance': r['relevance'][direction][strategy]}
                    elif dataset == 'summ_google':
                        final[dataset][nli] = np.mean(list(r[direction][strategy].values()))
                    elif dataset == 'realsumm':
                        final[dataset][nli] = {'summary': r['summary'][direction][strategy], 'system': r['system'][direction][strategy]}
                    elif dataset == 'Rank19':
                        final[dataset][nli] = r[direction][strategy]
                    else:
                        raise ValueError(f'No such dataset: {dataset}')
    pprint.pprint(final, width=1)
    return final

def plot_sum_trade_off(setup = 'ref', con=False):
    plt.style.use('seaborn-darkgrid')

    nli1, nli2 = defaultdict(dict), defaultdict(dict)
    if con:
        path = '../results/improvement_table_data_sum_con.csv'
    else:
        path = '../results/improvement_table_data_sum.csv'

    if os.path.exists(path):
        data = pd.read_csv(path)
    else:
        data = None
    tmp = defaultdict(list)

    if setup == 'ref':
        # ref sum
        if con:
            combined_results = pd.read_csv(f'../results/tables/combined-on-sum-realsumm-ref-avg-e.csv')
        else:
            combined_results = pd.read_csv(f'../results/tables/combined-on-sum-realsumm-ref-hr-e-c.csv')
        for metric in sum_metrics_dict['ref']:
            metric_data = combined_results[combined_results.metric == metric]
            nli1[metric]['summary'] = metric_data[(metric_data.level == 'summary') & (metric_data.nli == 'NLI1Score_monolingual')]['litepyramid'].values
            nli2[metric]['summary'] = metric_data[(metric_data.level == 'summary') & (metric_data.nli == 'NLI2Score_monolingual')]['litepyramid'].values
            nli1[metric]['system'] = metric_data[(metric_data.level == 'system') & (metric_data.nli == 'NLI1Score_monolingual')]['litepyramid'].values
            nli2[metric]['system'] = metric_data[(metric_data.level == 'system') & (metric_data.nli == 'NLI2Score_monolingual')]['litepyramid'].values

            for nli in nli_dict['ref']:
                for level in ['summary', 'system']:
                    ori = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['litepyramid']
                    ori_nli = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['litepyramid']
                    assert len(ori) == 1
                    ori = ori.values[0]
                    ori_nli = ori_nli.values[0]
                    for w in np.arange(0.1, 1.0, 0.1):
                        w = float('%.1f' % w)
                        current = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == w)]['litepyramid']
                        assert len(current) == 1
                        current = current.values[0]
                        improvement = (current-ori) / abs(ori) * 100
                        improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                        tmp['setup'].append(setup)
                        tmp['metric'].append(metric)
                        tmp['nli'].append(nli)
                        tmp['nli_weight'].append(w)
                        tmp['dataset'].append('realsumm')
                        tmp['type'].append(f"sum-{level}")
                        tmp['improvement'].append(improvement)
                        tmp['improvement_nli'].append(improvement_nli)
                        tmp['ori'].append(ori)
                        tmp['ori_nli'].append(ori_nli)
                        tmp['current'].append(current)

        if con:
            combined_results = pd.read_csv(f'../results/tables/combined-on-sum-summ-ref-avg-e.csv')
        else:
            combined_results = pd.read_csv(f'../results/tables/combined-on-sum-summ-ref-hr-e-c.csv')
        for metric in sum_metrics_dict['ref']:
            metric_data = combined_results[combined_results.metric == metric]
            nli1[metric]['system-summ'] = metric_data[(metric_data.aggregation == 'mean') & (metric_data.nli == 'NLI1Score_monolingual')]['avg'].values
            nli2[metric]['system-summ'] = metric_data[(metric_data.aggregation == 'mean') & (metric_data.nli == 'NLI2Score_monolingual')]['avg'].values

            for nli in nli_dict['ref']:
                for agg in ['max', 'mean']:
                    ori = metric_data[(metric_data.aggregation == agg) & (metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['avg']
                    ori_nli = metric_data[(metric_data.aggregation == agg) & (metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['avg']
                    assert len(ori) == 1
                    ori = ori.values[0]
                    ori_nli = ori_nli.values[0]
                    for w in np.arange(0.1, 1.0, 0.1):
                        w = float('%.1f' % w)
                        current = metric_data[(metric_data.aggregation == agg) & (metric_data.nli == nli) & (metric_data.nli_weight == w)]['avg']
                        assert len(current) == 1
                        current = current.values[0]
                        improvement = (current-ori) / abs(ori) * 100
                        improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                        tmp['setup'].append(setup)
                        tmp['metric'].append(metric)
                        tmp['nli'].append(nli)
                        tmp['nli_weight'].append(w)
                        tmp['dataset'].append(f'summeval-{agg}')
                        tmp['type'].append(f"sum-system")
                        tmp['improvement'].append(improvement)
                        tmp['improvement_nli'].append(improvement_nli)
                        tmp['ori'].append(ori)
                        tmp['ori_nli'].append(ori_nli)
                        tmp['current'].append(current)

        # adv
        if con:
            combined_results = pd.read_csv(f'../results/tables/combined-on-adv-ref-avg-e.csv')
        else:
            combined_results = pd.read_csv(f'../results/tables/combined-on-adv-ref-hr-e-c.csv')

        for metric in sum_metrics_dict['ref']:
            metric_data = combined_results[combined_results.metric == metric]
            nli1[metric]['adv'] = metric_data[metric_data.nli == 'NLI1Score_monolingual']['avg'].values
            nli2[metric]['adv'] = metric_data[metric_data.nli == 'NLI2Score_monolingual']['avg'].values

            for nli in nli_dict['ref']:
                ori = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['avg']
                ori_nli = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['avg']
                assert len(ori) == 1
                ori = ori.values[0]
                ori_nli = ori_nli.values[0]
                for w in np.arange(0.1, 1.0, 0.1):
                    w = float('%.1f' % w)
                    current = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == w)]['avg']
                    assert len(current) == 1
                    current = current.values[0]
                    improvement = (current-ori) / abs(ori) * 100
                    improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                    tmp['setup'].append(setup)
                    tmp['metric'].append(metric)
                    tmp['nli'].append(nli)
                    tmp['nli_weight'].append(w)
                    tmp['dataset'].append('summ_google')
                    tmp['type'].append(f"adv")
                    tmp['improvement'].append(improvement)
                    tmp['improvement_nli'].append(improvement_nli)
                    tmp['ori'].append(ori)
                    tmp['ori_nli'].append(ori_nli)
                    tmp['current'].append(current)

        nlis = [nli1, nli2]
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        metric_map = {
            'BARTScore_bart-large-cnn': 'BARTS-P',
            'BARTScore_bart-large-cnn+para_bi': 'BARTS-F',
        }
        for i in range(len(ax)):
            if i == 0:
                ax[i].set_ylabel('SummEval-system (Kendall)')
            elif i == 1:
                ax[i].set_ylabel('RealSumm-system (Pearson)')
            else:
                ax[i].set_ylabel('RealSumm-summary (Pearson)')

            ax[i].set_xlabel('Adv.(acuuracy)')

            for j, metric in enumerate(sum_metrics_dict['ref']):
                xs = nli1[metric]['adv']
                if i == 0:
                    ys_system_summ = nlis[0][metric]['system-summ']
                elif i == 1:
                    ys_system_summ = nlis[0][metric]['system']
                else:
                    ys_system_summ = nlis[0][metric]['summary']

                if metric in metric_map.keys():
                    metric = metric_map[metric]
                else:
                    metric = metric.split('_')[0]
                ax[i].plot(xs, ys_system_summ, linestyle='solid', label=metric, alpha=0.8, linewidth=0.8, marker='.', ms=3, zorder=1)

                ax[i].scatter(xs[0], ys_system_summ[0], marker='*', color='r', zorder=2)
                ax[i].scatter(xs[-1], ys_system_summ[-1], marker='x', color='r', zorder=2)

                if j == len(sum_metrics_dict['ref'])-1:
                    ax[i].scatter(xs[0], ys_system_summ[0], marker='*', color='r', label='Ori. Metric', zorder=2)
                    ax[i].scatter(xs[-1], ys_system_summ[-1], marker='x', color='r', label='NLI Metric', zorder=2)

        ax[2].legend()
        '''
        if con:
            plt.savefig('../results/plot/summ-ref-nli1-con.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('../results/plot/summ-ref-nli1.png', dpi=300, bbox_inches='tight')
        '''
        plt.show()

    else:
        # src sum
        combined_results = pd.read_csv(f'../results/tables/combined-on-sum-realsumm-src-rh--c.csv')
        for metric in sum_metrics_dict['src']:
            metric_data = combined_results[combined_results.metric == metric]
            nli1[metric]['summary'] = \
            metric_data[(metric_data.level == 'summary') & (metric_data.nli == 'NLI1Score_monolingual')][
                'litepyramid'].values
            nli2[metric]['summary'] = \
            metric_data[(metric_data.level == 'summary') & (metric_data.nli == 'NLI2Score_monolingual')][
                'litepyramid'].values
            nli1[metric]['system'] = \
            metric_data[(metric_data.level == 'system') & (metric_data.nli == 'NLI1Score_monolingual')][
                'litepyramid'].values
            nli2[metric]['system'] = \
            metric_data[(metric_data.level == 'system') & (metric_data.nli == 'NLI2Score_monolingual')][
                'litepyramid'].values

            for nli in nli_dict['ref']:
                for level in ['summary', 'system']:
                    ori = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['litepyramid']
                    ori_nli = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['litepyramid']
                    assert len(ori) == 1
                    ori = ori.values[0]
                    ori_nli = ori_nli.values[0]
                    for w in np.arange(0.1, 1.0, 0.1):
                        w = float('%.1f' % w)
                        current = metric_data[(metric_data.level == level) & (metric_data.nli == nli) & (metric_data.nli_weight == w)]['litepyramid']
                        assert len(current) == 1
                        current = current.values[0]
                        improvement = (current-ori) / abs(ori) * 100
                        improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                        tmp['setup'].append(setup)
                        tmp['metric'].append(metric)
                        tmp['nli'].append(nli)
                        tmp['nli_weight'].append(w)
                        tmp['dataset'].append('realsumm')
                        tmp['type'].append(f"sum-{level}")
                        tmp['improvement'].append(improvement)
                        tmp['improvement_nli'].append(improvement_nli)
                        tmp['ori'].append(ori)
                        tmp['ori_nli'].append(ori_nli)
                        tmp['current'].append(current)

        combined_results = pd.read_csv(f'../results/tables/combined-on-sum-summ-src-rh--c.csv')
        for metric in sum_metrics_dict['src']:
            metric_data = combined_results[combined_results.metric == metric]
            nli1[metric]['system-summ'] = \
            metric_data[(metric_data.nli == 'NLI1Score_monolingual')][
                'avg'].values
            nli2[metric]['system-summ'] = \
            metric_data[(metric_data.nli == 'NLI2Score_monolingual')][
                'avg'].values

            for nli in nli_dict['ref']:
                ori = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['avg']
                ori_nli = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['avg']
                assert len(ori) == 1
                ori = ori.values[0]
                ori_nli = ori_nli.values[0]
                for w in np.arange(0.1, 1.0, 0.1):
                    w = float('%.1f' % w)
                    current = metric_data[(metric_data.nli == nli) & (metric_data.nli_weight == w)]['avg']
                    assert len(current) == 1
                    current = current.values[0]
                    improvement = (current-ori) / abs(ori) * 100
                    improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                    tmp['setup'].append(setup)
                    tmp['metric'].append(metric)
                    tmp['nli'].append(nli)
                    tmp['nli_weight'].append(w)
                    tmp['dataset'].append(f'summeval')
                    tmp['type'].append(f"sum-system")
                    tmp['improvement'].append(improvement)
                    tmp['improvement_nli'].append(improvement_nli)
                    tmp['ori'].append(ori)
                    tmp['ori_nli'].append(ori_nli)
                    tmp['current'].append(current)

        # adv
        combined_results = pd.read_csv(f'../results/tables/combined-on-adv-src-rh--c.csv')
        for metric in sum_metrics_dict['src']:
            metric_data = combined_results[(combined_results.metric == metric)]
            summ_data = metric_data[(metric_data.dataset == 'summ_google') & (metric_data.nli == 'NLI1Score_monolingual')]['avg'].values
            rank_data = metric_data[(metric_data.dataset == 'Rank19') & (metric_data.nli == 'NLI1Score_monolingual')]['avg'].values
            nli1[metric]['adv'] = [np.mean([s1, s2]) for s1, s2 in zip(summ_data, rank_data)]

            for nli in nli_dict['ref']:
                for dataset in ['Rank19', 'summ_google']:
                    ori = metric_data[(metric_data.dataset == dataset) & (metric_data.nli == nli) & (metric_data.nli_weight == 0.0)]['avg']
                    ori_nli = metric_data[(metric_data.dataset == dataset) & (metric_data.nli == nli) & (metric_data.nli_weight == 1.0)]['avg']
                    assert len(ori) == 1
                    ori = ori.values[0]
                    ori_nli = ori_nli.values[0]
                    for w in np.arange(0.1, 1.0, 0.1):
                        w = float('%.1f' % w)
                        current = metric_data[(metric_data.dataset == dataset) & (metric_data.nli == nli) & (metric_data.nli_weight == w)]['avg']
                        assert len(current) == 1
                        current = current.values[0]
                        improvement = (current-ori) / abs(ori) * 100
                        improvement_nli = (current-ori_nli) / abs(ori_nli) * 100
                        tmp['setup'].append(setup)
                        tmp['metric'].append(metric)
                        tmp['nli'].append(nli)
                        tmp['nli_weight'].append(w)
                        tmp['dataset'].append(dataset)
                        tmp['type'].append(f"adv")
                        tmp['improvement'].append(improvement)
                        tmp['improvement_nli'].append(improvement_nli)
                        tmp['ori'].append(ori)
                        tmp['ori_nli'].append(ori_nli)
                        tmp['current'].append(current)

        nlis = [nli1, nli2]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        metric_map = {
            'BARTScore_bart-large-cnn': 'BARTS-FN'
        }
        for i in range(len(ax)):
            if i == 0:
                ax[i].set_ylabel('SummEval-system (Kendall)')
            elif i == 1:
                ax[i].set_ylabel('RealSumm-system (Pearson)')
            else:
                ax[i].set_ylabel('RealSumm-summary (Pearson)')

            ax[i].set_xlabel('Adv.(acuuracy)')

            for j, metric in enumerate(sum_metrics_dict['src']):
                xs = nli1[metric]['adv']
                if i == 0:
                    ys_system_summ = nlis[0][metric]['system-summ']
                elif i == 1:
                    ys_system_summ = nlis[0][metric]['system']
                else:
                    ys_system_summ = nlis[0][metric]['summary']

                if metric in metric_map.keys():
                    metric = metric_map[metric]
                else:
                    metric = metric.split('_')[0]
                ax[i].plot(xs, ys_system_summ, linestyle='solid', label=metric, alpha=0.8, linewidth=0.8, marker='.',
                           ms=3, zorder=1)
                ax[i].scatter(xs[0], ys_system_summ[0], marker='*', color='r', zorder=2)
                ax[i].scatter(xs[-1], ys_system_summ[-1], marker='x', color='r', zorder=2)

                if j == len(sum_metrics_dict['ref']) - 1:
                    ax[i].scatter(xs[0], ys_system_summ[0], marker='*', color='r', label='Ori. Metric', zorder=2)
                    ax[i].scatter(xs[-1], ys_system_summ[-1], marker='x', color='r', label='NLI Metric', zorder=2)

        ax[2].legend()
        #plt.savefig('../results/plot/sum-src-nli1.png', dpi=300, bbox_inches='tight')
        plt.show()

    '''
    if data is not None:
        data = pd.concat([data, pd.DataFrame(tmp)], ignore_index=True)
        data.drop_duplicates(inplace=True)
    else:
        data = pd.DataFrame(tmp)
        print('new file')
    if con:
        data.to_csv('../results/improvement_table_data_sum_con.csv', index=False)
    else:
        data.to_csv('../results/improvement_table_data_sum.csv', index=False)
    '''
    #print(data)

def plot_sum_confidence(estimator='median', con=False):
    plt.style.use('seaborn-darkgrid')
    if estimator == 'median':
        es = np.median
    else:
        es = estimator
    if con:
        data = pd.read_csv('../results/improvement_table_data_sum_con.csv')
    else:
        data = pd.read_csv('../results/improvement_table_data_sum.csv')

    data['type'] = ['Adv.' if t == 'adv' else 'Sum.' for t in data['type']]
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=data, x='nli_weight', y='improvement', hue='type', estimator=es)
    plt.legend(loc='upper left')
    plt.ylabel('improvement(%)')
    '''
    if con:
        plt.savefig('../results/plot/confidence_interval_{}_all_sum_con.png'.format(estimator), dpi=300,
                    bbox_inches='tight')
    else:
        plt.savefig('../results/plot/confidence_interval_{}_all_sum.png'.format(estimator), dpi=300,
                    bbox_inches='tight')
    '''
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=data, x='nli_weight', y='improvement_nli', hue='type', estimator=es, sort=False)
    plt.legend(loc='upper left')
    plt.ylabel('improvement(%)')
    '''
    if con:
        plt.savefig('../results/plot/confidence_interval_{}_all_sum_nli_con.png'.format(estimator), dpi=300,
                    bbox_inches='tight')
    else:
        plt.savefig('../results/plot/confidence_interval_{}_all_sum_nli.png'.format(estimator), dpi=300,
                    bbox_inches='tight')
    '''
    plt.show()

def combined_sum(use_article=False):
    metrics = sum_metrics_dict['ref'] if not use_article else sum_metrics_dict['src']

    for direction in ['rh', 'hr', 'avg']:
        if use_article and direction != 'rh':
            continue
        for strategy in ['e', '-c', 'e-n', 'e-c', 'e-n-2c']:
            # adv
            results = defaultdict(list)
            for dataset in datasets_dict_sum['adv']:
                for metric in metrics:
                    for nli in nli_dict['ref']:
                        if not use_article and dataset == 'Rank19':
                            continue
                        metric_data = combine_nli_and_metric(nli, metric, dataset, adv=True, use_article=use_article, direction=direction, strategy=strategy)
                        r_dict = evaluate(metric_data, dataset, adv=True)
                        for w in r_dict.keys():
                            if dataset == 'summ_google':
                                r = np.mean([v for _, v in r_dict[w].items()])
                            elif dataset == 'Rank19':
                                r = r_dict[w]
                            results['dataset'].append(dataset)
                            results['metric'].append(metric)
                            results['nli'].append(nli)
                            results['nli_weight'].append(w)
                            results['avg'].append(r)

            data = pd.DataFrame(results)
            data.to_csv(f"../results/tables/combined-on-adv-{'ref' if not use_article else 'src'}-{direction}-{strategy}.csv", index=False)

            results = defaultdict(list)
            dataset = 'realsumm'
            for metric in metrics:
                for nli in nli_dict['ref']:
                    metric_data = combine_nli_and_metric(nli, metric, dataset, adv=False, agg='None', use_article=use_article, direction=direction, strategy=strategy)
                    r_dict = evaluate(metric_data, dataset, adv=False)
                    for level in ['summary', 'system']:
                        level_data = r_dict[level]
                        for w in level_data.keys():
                            results['dataset'].append(dataset)
                            results['metric'].append(metric)
                            results['nli'].append(nli)
                            results['nli_weight'].append(w)
                            results['level'].append(level)
                            results['correlation'].append('pearson')
                            results['litepyramid'].append(level_data[w])
            data = pd.DataFrame(results)
            print(data)
            data.to_csv(f"../results/tables/combined-on-sum-{dataset}-{'ref' if not use_article else 'src'}-{direction}-{strategy}.csv", index=False)

            results = defaultdict(list)
            dataset = 'summ'
            for metric in metrics:
                for nli in nli_dict['ref']:
                    aggs = ['max', 'mean'] if not use_article else ['None']
                    for agg in aggs:
                        metric_data = combine_nli_and_metric(nli, metric, dataset, adv=False, agg=agg, use_article=use_article, direction=direction, strategy=strategy)
                        r_dict = evaluate(metric_data, dataset, adv=False)

                        for w in r_dict.keys():
                            results['dataset'].append(dataset)
                            results['metric'].append(metric)
                            results['nli'].append(nli)
                            results['nli_weight'].append(w)
                            results['aggregation'].append(agg)
                            results['level'].append('system')
                            results['correlation'].append('kendall')
                            ks = []
                            for c, k in r_dict[w].items():
                                results[c].append(k)
                                ks.append(k)
                            results['avg'].append(np.mean(ks))

            data = pd.DataFrame(results)
            print(data)
            data.to_csv(f"../results/tables/combined-on-sum-{dataset}-{'ref' if not use_article else 'src'}-{direction}-{strategy}.csv", index=False)
