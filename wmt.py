import pandas as pd
import numpy as np
import os
import tqdm
import argparse
from collections import defaultdict
import json
from scipy.stats import pearsonr
import pickle

def read_human_scores():
    data = pd.read_csv('experiments/datasets/WMT17/DA-seglevel.csv', delimiter=' ')
    return data

def load_evaluation_data_1516(lp, year):
    start = 'DAseg.' if year == 2016 else ''
    # load reference
    with open('experiments/datasets/DAseg-wmt-newstest{}/{}newstest{}.reference.{}'.format(year, start, year, lp), 'r') as f:
        references = f.readlines()
    with open('experiments/datasets/DAseg-wmt-newstest{}/{}newstest{}.human.{}'.format(year, start, year, lp), 'r') as f:
        human = f.readlines()
    with open('experiments/datasets/DAseg-wmt-newstest{}/{}newstest{}.mt-system.{}'.format(year, start, year, lp), 'r') as f:
        system_outputs = f.readlines()
    with open('experiments/datasets/DAseg-wmt-newstest{}/{}newstest{}.source.{}'.format(year, start, year, lp), 'r') as f:
        source = f.readlines()
    references = [l.strip() for l in references]
    human = [float(l.strip()) for l in human]
    system_outputs = [l.strip() for l in system_outputs]
    source = [l.strip() for l in source]

    return source, references, system_outputs, human

def load_evaluation_data_17(data, lp):
    data = data[data['LP'] == lp]
    evaluation_data = defaultdict(list)
    system_data = data.groupby(by='SYSTEM')
    src, tgt = lp.split('-')

    with open('experiments/datasets/WMT17/sources/newstest2017-{}{}-src.{}'.format(src,tgt,src), 'r') as f:
            sources = f.readlines()
    with open('experiments/datasets/WMT17/references/newstest2017-{}{}-ref.{}'.format(src,tgt,tgt), 'r') as f:
            refs = f.readlines()
    sources = np.array([l.strip('\n') for l in sources])
    refs = np.array([l.strip('\n') for l in refs])

    for _, group in system_data:
        # some typos in system names for zh-en?
        system = group['SYSTEM'].values[0].split('+')[0]
        system = system.replace('CASICT-cons.5144', 'CASICT-DCU-NMT.5144')
        system = system.replace('ROCMT.5167', 'ROCMT.5183')

        path = 'experiments/datasets/WMT17/system-outputs/newstest2017/{}-{}/newstest2017.{}.{}-{}'.format(src, tgt, system, src,tgt)
        #print(group)
        assert os.path.exists(path), path

        ids = np.array([int(i-1) for i in list(group['SID'])])
        #print(ids)
        human_scores = [float(i) for i in list(group['HUMAN'])]
        with open(path, 'r') as f:
            system_outputs = f.readlines()
        system_outputs = np.array([l.strip('\n') for l in system_outputs])

        evaluation_data['source'] += list(sources[ids])
        evaluation_data['reference'] += list(refs[ids])
        evaluation_data['system_output'] += list(system_outputs[ids])
        evaluation_data['human'] += human_scores
        assert len(evaluation_data['source']) == len(evaluation_data['human'])

    assert len(evaluation_data['human']) == 560
    return evaluation_data['source'], evaluation_data['reference'], evaluation_data['system_output'], evaluation_data['human']

def store_scores(scores_dict, metric, year, output_dir='../results/'):
    output_dir = output_dir+'wmt{}_seg_scores'.format(year[-2:])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+'/{}_scores.pkl'.format(metric), 'wb') as out:
        pickle.dump(scores_dict, out)

def print_and_save(metric, pearson_dict, year, setup, leaderboard_dir = '../results/', save=False):
    leaderboard_path = leaderboard_dir+'wmt{}_seg_results.csv'.format(str(year)[-2:])
    col = ','.join(['metric', 'setup', 'correlation']+list(pearson_dict.keys())) + ',avg\n'
    if not os.path.exists(leaderboard_path):
        s = col
    else:
        s = ''
    if not isinstance(pearson_dict, defaultdict):
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

def evaluate(args, human_data, scorer, lps):
    scores_dict = {}
    pearson = {}
    for lp in tqdm.tqdm(lps):
        if args.year == 2017:
            src, ref, hyp, human = load_evaluation_data_17(human_data, lp)
        else:
            src, ref, hyp, human = load_evaluation_data_1516(lp, year=args.year)
        scores = scorer.score_all(srcs=src, refs=ref, hyps=hyp, srcl=lp.split('-')[0])
        scores_dict[lp] = scores
        pearson[lp] = pearsonr(scores, human)[0]
        print('pearson for {}: {}'.format(lp, pearson[lp]))
    return pearson, scores_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # all
    parser.add_argument('--model', type=str, default='R', help='R or D')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cross_lingual', action='store_true')
    parser.add_argument('--lp', type=str, default='all')
    parser.add_argument('--year', type=int, default=2017)
    parser.add_argument('--direction', type=str, default='avg')
    parser.add_argument('--nli_weight', type=float, default=1.0)
    parser.add_argument('--combine_with', type=str, default='None')
    parser.add_argument('--formula', type=str, default='e')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    lp_dict = {
        2015: ['cs-en', 'de-en', 'fi-en', 'ru-en'],
        2016: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en'],
        2017: ['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']
    }
    if args.lp != 'all':
        lps = args.lp.split(',')
    else:
        lps = lp_dict[args.year]


    from MENLI import MENLI
    scorer = MENLI(batch_size=args.batch_size, device=args.device, direction=args.direction, formula=args.formula,
                   nli_weight=args.nli_weight, combine_with=args.combine_with, cross_lingual=args.cross_lingual, model=args.model)
    metric_hash = scorer.metric_hash

    human_data = read_human_scores() if args.year == 2017 else None

    pearson, scores_dict = evaluate(args, human_data, scorer, lps)

    # print_and_save(metric, pearson_dict, year, setup, leaderboard_dir = '../results/')
    print_and_save(metric_hash, pearson, args.year, 'ref-free' if args.cross_lingual else 'ref-based')








