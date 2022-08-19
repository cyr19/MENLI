import pandas as pd
import numpy as np
import os
from collections import defaultdict

def read_human_scores():
    data = pd.read_csv('datasets/WMT17/DA-seglevel.csv', delimiter=' ')
    return data

def load_evaluation_data_1516(lp, year):
    start = 'DAseg.' if year == 2016 else ''
    # load reference
    with open('datasets/DAseg-wmt-newstest{}/{}newstest{}.reference.{}'.format(year, start, year, lp), 'r') as f:
        references = f.readlines()
    with open('datasets/DAseg-wmt-newstest{}/{}newstest{}.human.{}'.format(year, start, year, lp), 'r') as f:
        human = f.readlines()
    with open('datasets/DAseg-wmt-newstest{}/{}newstest{}.mt-system.{}'.format(year, start, year, lp), 'r') as f:
        system_outputs = f.readlines()
    with open('datasets/DAseg-wmt-newstest{}/{}newstest{}.source.{}'.format(year, start, year, lp), 'r') as f:
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
    with open('datasets/WMT17/sources/newstest2017-{}{}-src.{}'.format(src,tgt,src), 'r') as f:
            sources = f.readlines()
    with open('datasets/WMT17/references/newstest2017-{}{}-ref.{}'.format(src,tgt,tgt), 'r') as f:
            refs = f.readlines()
    sources = np.array([l.strip('\n') for l in sources])
    refs = np.array([l.strip('\n') for l in refs])

    for _, group in system_data:
        # some typos in system names for zh-en?
        system = group['SYSTEM'].values[0].split('+')[0]
        system = system.replace('CASICT-cons.5144', 'CASICT-DCU-NMT.5144')
        system = system.replace('ROCMT.5167', 'ROCMT.5183')

        path = 'datasets/WMT17/system-outputs/newstest2017/{}-{}/newstest2017.{}.{}-{}'.format(src, tgt, system, src,tgt)
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
