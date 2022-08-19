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
                    print(c_data.keys())
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
                    #print(e_data['zh-en'].keys())
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
        else: # sum
            # tac datasets
            if 'tac' in dataset:
                e_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_e.pkl", adv=False)
                e_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_e.pkl", adv=False)
                n_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_n.pkl", adv=False)
                n_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_n.pkl", adv=False)
                c_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_c.pkl", adv=False)
                c_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_c.pkl", adv=False)

                r = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
                for direction in ['rh', 'hr', 'avg']:
                    if direction == 'rh':
                        e_data, n_data, c_data = e_data_rh, n_data_rh, c_data_rh
                    elif direction == 'hr':
                        e_data, n_data, c_data = e_data_hr, n_data_hr, c_data_hr
                    else:
                        e_data, n_data, c_data = defaultdict(list), defaultdict(list), defaultdict(list)
                        for topic in e_data_rh.keys():
                            e_data[topic] = [(s1+s2)/2 for s1, s2 in zip(e_data_rh[topic], e_data_hr[topic])]
                            n_data[topic] = [(s1+s2)/2 for s1, s2 in zip(n_data_rh[topic], n_data_hr[topic])]
                            c_data[topic] = [(s1+s2)/2 for s1, s2 in zip(c_data_rh[topic], c_data_hr[topic])]
                    # e
                    r[direction]['e'] = e_data
                    for topic in e_data_rh.keys():
                        # -c
                        r[direction]['-c'][topic] = [-c for c in c_data[topic]]
                        # e-n
                        r[direction]['e-n'][topic] = [e-n for e, n in zip(e_data[topic], n_data[topic])]
                        # e-c
                        r[direction]['e-c'][topic] = [e - c for e, c in zip(e_data[topic], c_data[topic])]
                        # e-n-2c
                        r[direction]['e-n-2c'][topic] = [e - n - 2 * c for e, n, c in zip(e_data[topic], n_data[topic], c_data[topic])]
            elif dataset in ['summ', 'realsumm', 'BAGEL', 'SFHOT']:
                if dataset == 'realsumm' or use_article:
                    agg = 'None'
                #if use_article and dataset == 'summ':
                #    agg = 'mean'
                e_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_e_{agg}{suffix}.pkl", adv=False)
                e_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_e_{agg}{suffix}.pkl", adv=False)
                n_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_n_{agg}{suffix}.pkl", adv=False)
                n_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_n_{agg}{suffix}.pkl", adv=False)
                c_data_rh = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_rh_c_{agg}{suffix}.pkl", adv=False)
                c_data_hr = load_metric_scores(f"../results/{dataset}_scores/{nli_metric}_hr_c_{agg}{suffix}.pkl", adv=False)
                #print(type(e_data_rh))
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
                else:
                    for direction, d_data in metric_data.items():
                        for strategy, s_data in d_data.items():
                            for lp, system_scores in s_data.items():
                                evs = me_data.EvalSet(dataset, lp) if 'mqm' not in dataset else me_data.EvalSet(dataset.split('_mqm')[0], lp)
                                if level == 'sys':
                                    # system-level
                                    if 'mqm' not in dataset:
                                        gold_scores = evs.Scores('sys', 'wmt-z')
                                    else:
                                        if lp != 'zh-en':
                                            continue
                                        gold_scores = evs.Scores('sys', 'mqm')
                                    system_scores = {system: [np.mean(scores)] for system, scores in system_scores.items()}
                                    corr = evs.Correlation(gold_scores, system_scores, list(system_scores.keys()))
                                    r[direction][strategy][lp] = corr.Pearson()[0]
                                else:
                                    # segment-level
                                    if 'mqm' not in dataset:
                                        gold_scores = evs.Scores('seg', 'wmt-raw')
                                        corr = evs.Correlation(gold_scores, system_scores, list(system_scores.keys()))
                                        r[direction][strategy][lp] = corr.Kendalllike()[0]
                                    else:
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
                if dataset in ['wmt15','wmt16','wmt17']:
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
                                if 'mqm' not in dataset:
                                    gold_scores = evs.Scores('sys', 'wmt-z')
                                    sys_names = list(system_scores.keys())
                                else:
                                    if lp != 'zh-en':
                                        continue
                                    gold_scores = evs.Scores('sys', 'mqm')
                                    sys_names = set(gold_scores) - evs.human_sys_names
                                system_scores = {system: [np.mean(scores)] for system, scores in system_scores.items()}
                                corr = evs.Correlation(gold_scores, system_scores, sys_names)
                                r[w][lp] = corr.Pearson()[0]
                            else:
                                # segment-level
                                if 'mqm' not in dataset:
                                    gold_scores = evs.Scores('seg', 'wmt-raw')
                                    corr = evs.Correlation(gold_scores, system_scores, list(system_scores.keys()))
                                    r[w][lp] = corr.KendallLike()[0]
                                else:
                                    if lp != 'zh-en':
                                        continue
                                    gold_scores = evs.Scores('seg', 'mqm')
                                    sys_names = set(gold_scores) - evs.human_sys_names
                                    corr = evs.Correlation(gold_scores, system_scores, sys_names)
                                    r[w][lp] = corr.Pearson()[0]
                        if ifprint:
                            print_pearson(str(w), r[w], dataset)


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
            print(len(nli_data))
            print(len(nli_data['scores']))
            print(len(nli_data['scores_in']))
            metric_data = load_metric_scores(f'../results/adv_scores/Rank19_{metric}.pkl', adv=False)
            nli_scores, metric_scores = {}, {}
            nli_scores['scores'], nli_scores['scores_in'] = nomalize_scores((list(nli_data['scores']), list(nli_data['scores_in'])), norm=norm)
            print(len(nli_scores['scores_in']))
            metric_scores['scores'], metric_scores['scores_in'] = nomalize_scores((metric_data['scores'], metric_data['scores_in']), norm=norm)

            for w1 in np.arange(0, 1.1, 0.1):
                w2 = 1-w1
                w1 = float('%.1f' % w1)
                w2 = float('%.1f' % w2)
                combined_scores = method(nli_scores['scores'], metric_scores['scores'], w1, w2)
                combined_scores_in = method(nli_scores['scores_in'], metric_scores['scores_in'], w1, w2)
                print(len(nli_scores['scores_in']))
                print(len(metric_scores['scores_in']))
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
    print(metrics)
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
                        aa[i][j] = acc_dict[directions[i]][strategies[j]]
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
                        print(f"{dataset} - {m}\n{matrix}")
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
                                print(f"{dataset} - {m} - {k}\n{v}")
                                matrix_dict[use_article]['sum'].append(v)
                    elif dataset == 'realsumm':
                        matrix, _, _ = get_matrix(m, dataset, adv=False, use_article=use_article)
                        for k, v in matrix.items():
                            print(f"{dataset} - {m} - {k}\n{v}")
                            matrix_dict[use_article]['sum'].append(v)

        print(f"{len(matrix_dict[use_article]['adv'])} adv matrices")
        print(f"{len(matrix_dict[use_article]['sum'])} sum matrices")
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


