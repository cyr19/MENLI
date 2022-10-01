import pickle

import pandas as pd
import numpy as np
import os
from metrics import scorer_utils
import tqdm
import argparse

from collections import defaultdict

def data_append(n, data, error, ids, metric_hash, scores, scores_ad):
    if 'NLI' not in args.metric or 'MENLI' in args.metric:
        tmp = pd.DataFrame({
            'error': [error] * n,
            'id': ids,
            'metric': ['{}({})'.format(args.metric, metric_hash)] * n,
            'model': [args.model] * n,
            'direction': [np.nan] * n,
            'prob': [np.nan] * n,
            'hyp_score': scores,
            'hyp_adv_score': scores_ad
        })
        data.append(tmp)
    else:
        scores_c, scores_n, scores_e = scores
        scores_ad_c, scores_ad_n, scores_ad_e = scores_ad

        tmp_c = pd.DataFrame({
            'error': [error] * n,
            'id': ids,
            'metric': ['{}({})'.format(args.metric, metric_hash)] * n,
            'model': [args.model] * n,
            'direction': [args.direction] * n,
            'prob': ['c'] * n,
            'hyp_score': scores_c,
            'hyp_adv_score': scores_ad_c
        })

        tmp_n = pd.DataFrame({
            'error': [error] * n,
            'id': ids,
            'metric': ['{}({})'.format(args.metric, metric_hash)] * n,
            'model': [args.model] * n,
            'direction': [args.direction] * n,
            'prob': ['n'] * n,
            'hyp_score': scores_n,
            'hyp_adv_score': scores_ad_n
        })

        tmp_e = pd.DataFrame({
            'error': [error] * n,
            'id': ids,
            'metric': ['{}({})'.format(args.metric, metric_hash)] * n,
            'model': [args.model] * n,
            'direction': [args.direction] * n,
            'prob': ['e'] * n,
            'hyp_score': scores_e,
            'hyp_adv_score': scores_ad_e
        })

        data.append(tmp_c)
        data.append(tmp_n)
        data.append(tmp_e)

    return data

def calculate_accuracy_and_kendall(scores, scores_ad):
    num_hit = np.sum([scores[i] > scores_ad[i] for i in range(len(scores))])
    num_miss = np.sum([scores[i] < scores_ad[i] for i in range(len(scores))])
    accuracy = float(num_hit) / float(len(scores))
    kendall = float((num_hit - num_miss)) / float((num_hit + num_miss))
    return accuracy, kendall

def print_and_save(metric_hash, dataset, errors, acc, ken, save=True):
    cols = ['metric', 'setup', 'dataset', 'measurement'] + errors + ['average']
    cols = ','.join(cols) + '\n'
    accs = [str(acc[k]) for k in errors]
    values = ['{}({})'.format(args.metric, metric_hash), 'ref-free' if args.cross_lingual or args.use_article else 'ref-based', dataset, 'accuracy'] + accs + \
             [str(np.mean(list(acc.values())))]
    values = ','.join(values) + '\n'
    print(cols + values)
    output_path = args.output_path
    if save:
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                f.write(cols + values)
        else:
            with open(output_path, 'a') as f:
                f.write(values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # all
    parser.add_argument('--metric', type=str, default='None')
    parser.add_argument('--model', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='datasets/adv_datasets/')
    parser.add_argument('--dataset', type=str, default='paws_ori_google')
    parser.add_argument('--output_path', type=str, default='../results/adv_test.csv')
    parser.add_argument('--not_store_scores', action='store_false')
    parser.add_argument('--cross_lingual', action='store_true')
    parser.add_argument('--use_article', action='store_true')
    parser.add_argument('--fluency', action='store_true')
    parser.add_argument('--factuality', action='store_true')
    parser.add_argument('--errors', type=str, default=None)

    # BARTScore
    parser.add_argument('--bidirection', action='store_true')

    # NLI1Score, NLI2Score,
    parser.add_argument('--direction', type=str, default='rh')

    # XMoverScore
    parser.add_argument('--mapping', type=str, default='CLP', help='CLP or UMD')

    # NLI1, NLI2
    parser.add_argument('--checkpoint', type=int, default=0)

    # SentSim_new
    parser.add_argument('--use_wmd', action='store_true')

    # MENLI
    parser.add_argument('--nli_weight', type=float, default=1.0)
    parser.add_argument('--combine_with', type=str, default='None')

    import json
    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))
    scorer, metric_hash = scorer_utils.init_scorer(args)

    if '/' in args.dataset:
        srcl = args.dataset.split('/')[1]
    else:
        srcl = None
    data_dir = args.data_dir + args.dataset + '/data.csv'
    data_df = pd.read_csv(data_dir)
    data_df.sort_values(by=['error', 'id'], inplace=True)
    grouped_data = data_df.groupby('error')
    if not args.errors:
        errors = ['add', 'drop', 'name', 'neg', 'num', 'pron', 'word_JJ', 'word_NN', 'word_VB', 'typo', 'jumble', 'subject_verb_dis']
        if args.fluency:
            errors = ['typo', 'jumble', 'subject_verb_dis']
        if args.factuality:
            errors = ['add', 'drop', 'name', 'neg', 'num', 'pron', 'word_JJ', 'word_NN', 'word_VB']
    else:
        errors = args.errors.split(',')

    if 'NLI' not in args.metric or 'MENLI' in args.metric:
        acc, kendall = {}, {}
    else:
        acc, kendall = defaultdict(dict), defaultdict(dict)
    data = []
    for error in tqdm.tqdm(errors):
        group = data_df[data_df.error == error]
        print(error)
        refs = list(group['ref'])
        hyps = list(group['hyp_para'])  # only ref-based cases need hyp-para
        # ref-free
        if args.cross_lingual or args.use_article:
            hyps_ad = list(group['hyp_adv_free'])
        # ref-based
        else:
            hyps_ad = list(group['hyp_adv_based'])
        sources = list(group['source'])
        ids = list(group['id'])
        if 'MENLI' not in args.metric:
            if args.cross_lingual or args.use_article:
                # ref-free: m(src, ref) vs. m(src, hyp-adv)
                scores = scorer_utils.scoring(args, scorer, [], refs, sources, srcl=srcl)
                scores_ad = scorer_utils.scoring(args, scorer, [], hyps_ad, sources, srcl=srcl)
            else:
                # ref-based: m(ref, hyp-para) vs. m(ref, hyp-adv)
                scores = scorer_utils.scoring(args, scorer, refs, hyps, sources, p=error)
                scores_ad = scorer_utils.scoring(args, scorer, refs, hyps_ad, sources, p=error)
        else:
            if args.cross_lingual or args.use_article:
                # ref-free: m(src, ref) vs. m(src, hyp-adv)
                nli_scores = scorer.score_nli(srcs=sources, refs=[], hyps=refs)
                nli_scores += scorer.score_nli(srcs=sources, refs=[], hyps=hyps_ad)
                metric_scores = scorer.score_metric(srcs=sources, refs=[], hyps=refs, srcl=srcl)
                metric_scores += scorer.score_metric(srcs=sources, refs=[], hyps=hyps_ad, srcl=srcl)
                all_scores = scorer.combine(metric_scores=metric_scores, nli_scores=nli_scores)
                scores = all_scores[:len(refs)]
                scores_ad = all_scores[len(refs):]
                assert len(scores) == len(scores_ad)
            else:
                nli_scores = scorer.score_nli(srcs=[], refs=refs, hyps=hyps)
                nli_scores += scorer.score_nli(srcs=[], refs=refs, hyps=hyps_ad)
                metric_scores = scorer.score_metric(srcs=[], refs=refs, hyps=hyps, srcl=srcl)
                metric_scores += scorer.score_metric(srcs=[], refs=refs, hyps=hyps_ad, srcl=srcl)
                all_scores = scorer.combine(metric_scores=metric_scores, nli_scores=nli_scores)
                scores = all_scores[:len(refs)]
                scores_ad = all_scores[len(refs):]
                assert len(scores) == len(scores_ad)

        n = len(group)
        data = data_append(n, data, error, ids, metric_hash, scores, scores_ad)
        if not isinstance(scores, tuple):
            acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)
        else:
            acc['c'][error], kendall['c'][error] = calculate_accuracy_and_kendall(scores[0], scores_ad[0])
            acc['n'][error], kendall['n'][error] = calculate_accuracy_and_kendall(scores[1], scores_ad[1])
            acc['e'][error], kendall['e'][error] = calculate_accuracy_and_kendall(scores[2], scores_ad[2])

    if not args.not_store_scores:
        print("Storing metric scores.")
        results = pd.concat(data, ignore_index=True)
        store_dir = '../results/adv_scores'
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        dataset = args.dataset.replace('/', '-')
        results.to_csv(store_dir + '/{}_{}_{}{}.csv'.format(dataset, args.metric, metric_hash, '_use_article' if args.use_article else ''), index=False)

    if 'NLI' not in args.metric or 'MENLI' in args.metric:
        print_and_save(metric_hash, args.dataset, errors, acc, kendall, save=True)
    else:
        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(v_hash, args.dataset, errors, acc[v], kendall[v], save=True)









