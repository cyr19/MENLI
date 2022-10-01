from mt_metrics_eval import data
import numpy as np
import os
import tqdm
import argparse
import torch
from collections import defaultdict
import json
from metrics.scorer_utils import init_scorer, scoring
import pickle

def store_scores(scores_dict, metric, dataset, output_dir='../results/'):
    output_dir = output_dir+'{}_{}_scores'.format(dataset, 'seg' if 'mqm' in dataset else 'sys')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+'/{}_scores.pkl'.format(metric), 'wb') as out:
        pickle.dump(scores_dict, out)


def print_and_save(metric, pearson_dict, dataset, setup, leaderboard_dir='../results/', save=True):
    leaderboard_path = leaderboard_dir+'{}_{}_results.csv'.format(dataset, 'seg' if 'mqm' in dataset else 'sys')
    col = ','.join(['metric', 'setup', 'correlation']+list(pearson_dict.keys())) + ',avg\n'
    if not os.path.exists(leaderboard_path):
        s = col
    else:
        s = ''
    if 'e' not in pearson_dict.keys():
        scores = list(pearson_dict.values())
        scores.append(np.mean(list(pearson_dict.values())))
        scores = [str(i) for i in scores]
        s += ','.join([metric, setup, 'pearson']+scores)+'\n'
    else:
        for v in pearson_dict.keys():
            scores = list(pearson_dict[v].values())
            scores.append(np.mean(list(pearson_dict[v].values())))
            scores = [str(i) for i in scores]
            s += ','.join([metric+'_'+v, setup, 'pearson']+scores)+'\n'
    print(s)
    if save:
        with open(leaderboard_path, 'a') as out:
            out.write(s)

def evaluate(args, scorer, lps):
    if 'NLI' not in args.metric:
        scores_dict = defaultdict(dict)
        pearson = defaultdict(dict)
    else:
        scores_dict = defaultdict(lambda: defaultdict(dict))
        pearson = defaultdict(lambda: defaultdict(dict))
    for lp in lps:
        evs = data.EvalSet(args.dataset, lp)
        # wmt20-21 mqm zh-en segment-lvel
        if args.mqm and lps == ['zh-en']:
            print('mqm....')
            ref = evs.all_refs[evs.std_ref]
            gold_scores = evs.Scores('seg', 'mqm')
        elif args.dataset == 'wmt20':
            ref = evs.all_refs[evs.std_ref]
            gold_scores = evs.Scores('sys', evs.StdHumanScoreName('sys'))
        elif args.dataset == 'wmt21.news' and not args.mqm:
            ref = evs.all_refs['refA']
            gold_scores = evs.Scores('sys', 'wmt-z')
        else:
            raise ValueError('no such data.')
        src = evs.src

        # without human systems
        sys_names = set(gold_scores) - evs.human_sys_names
        for s in tqdm.tqdm(sys_names, desc=lp):
            for r, sr, sys in zip(ref, src, evs.sys_outputs[s]):
                if '' in [r, sr, sys]:
                    print(f'reference:\n{r}\nsource:\n{sr}\nsystem:\n\n{sys}')
            print(lp)
            print('source language: {}'.format(lp.split('-')[0]))
            scores = scoring(args, scorer, refs=ref, sources=src, hyps=evs.sys_outputs[s], srcl=lp.split('-')[0])
            if not isinstance(scores, tuple):
                scores_dict[lp][s] = scores
            else:
                scores_dict['c'][lp][s], scores_dict['n'][lp][s], scores_dict['e'][lp][s] = scores

        # calculate pearson
        # system-level pearson with DA
        if not args.mqm:
            if 'e' not in scores_dict.keys():
                if isinstance(scores, torch.Tensor):
                    system_scores = {system: [np.mean(scores.numpy())] for system, scores in scores_dict[lp].items()}
                else:
                    system_scores = {system: [np.mean(scores)] for system, scores in scores_dict[lp].items()}
                corr = evs.Correlation(gold_scores, system_scores, sys_names)
                pearson[lp] = corr.Pearson()[0]
                print('Pearson for {}: {}'.format(lp, pearson[lp]))
            else:
                for v in scores_dict.keys():
                    system_scores = {system: [np.mean(scores)] for system, scores in scores_dict[v][lp].items()}
                    corr = evs.Correlation(gold_scores, system_scores, sys_names)
                    pearson[v][lp] = corr.Pearson()[0]
                print('Pearson for {}: {}(e)'.format(lp, pearson['e'][lp]))

        # segment-level Pearson with MQM
        else:
            if 'e' not in scores_dict.keys():
                corr = evs.Correlation(gold_scores, scores_dict[lp], sys_names)
                pearson[lp] = corr.Pearson()[0]
                print('Pearson for {}: {}'.format(lp, pearson[lp]))
            else:
                for v in scores_dict.keys():
                    corr = evs.Correlation(gold_scores, scores_dict[v][lp], sys_names)
                    pearson[v][lp] = corr.Pearson()[0]
                print('Pearson for {}: {}(e)'.format(lp, pearson['e'][lp]))

    return pearson, scores_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # all
    parser.add_argument('--metric', type=str, default='None')
    parser.add_argument('--model', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--not_store_scores', action='store_false')
    parser.add_argument('--cross_lingual', action='store_true')
    parser.add_argument('--lp', type=str, default='all')
    parser.add_argument('--dataset', type=str, default='wmt21.news')
    parser.add_argument('--mqm', action='store_true')

    # BARTScore
    parser.add_argument('--bidirection', action='store_true')

    # NLI1Score, NLI2Score,
    parser.add_argument('--direction', type=str, default='rh')
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--formula', type=str, default='-c')

    # XMoverScore
    parser.add_argument('--mapping', type=str, default='CLP', help='CLP or UMD')

    # SentSim
    parser.add_argument('--use_wmd', action='store_true')

    # MENLI
    parser.add_argument('--nli_weight', type=float, default=1.0)
    parser.add_argument('--combine_with', type=str, default='None')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    lp_dict = {
        'wmt21.news': ['cs-en', "de-en", 'ha-en', 'is-en', "ja-en", 'ru-en', 'zh-en'],
        'wmt20': ['cs-en', 'de-en', 'iu-en', 'ja-en', 'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en']
    }
    if args.lp != 'all':
        lps = args.lp.split(',')
    else:
        lps = lp_dict[args.dataset]

    if args.mqm:
        print('Note: when using --mqm, only evaluate on zh-en in WMT20-21 with MQM annotations at segment-level.')
        lps = ['zh-en']
    print(lps)
    scorer, metric_hash = init_scorer(args)

    pearson, scores_dict = evaluate(args, scorer, lps)

    dataset = args.dataset+'_mqm' if args.mqm else args.dataset
    print_and_save(args.metric+'_'+metric_hash, pearson, dataset, 'ref-free' if args.cross_lingual else 'ref-based')

    if not args.not_store_scores:
        if 'e' not in scores_dict.keys():
            store_scores(scores_dict, args.metric+'_'+metric_hash, dataset)
        else:
            for v in scores_dict.keys():
                store_scores(scores_dict[v], args.metric + '_' + metric_hash+'_'+v, dataset)






