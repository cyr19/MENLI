import numpy as np
import torch
import os
def init_scorer(**metric_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_hash = 'default'
    if 'BERTScore' in metric_config['combine_with']:
        from bert_score.scorer import BERTScorer
        scorer = BERTScorer(lang='en', idf=True)
        metric_hash = scorer.hash
    elif metric_config['combine_with'] == 'MoverScore':
        from moverscore_re import MoverScorer
        scorer = MoverScorer(idf=True)
    elif metric_config['combine_with'] == 'None':
        print('NLI metric only.')
        scorer = None
    elif 'XMoverScore' in metric_config['combine_with']:
        mapping = metric_config['combine_with'].split('-')[1]
        from xmoverscore.scorer import XMOVERScorer
        scorer = XMOVERScorer(model_name='bert-base-multilingual-cased', lm_name='gpt2', device=device)
        metric_hash = mapping
    else:
        raise NotImplementedError('Metric not supported yet.')
    return scorer, metric_hash

# XMoverScore
def metric_combination(a, b, alpha):
    return alpha[0] * np.array(a) + alpha[1] * np.array(b)

def scoring(scorer, refs, hyps, sources, srcl='de', metric_config={}):
    if metric_config['combine_with'] == 'BERTScore-F':
        if scorer.idf:
            scorer.compute_idf(refs)
        scores = scorer.score(hyps, refs)[2].detach().numpy().tolist()  # F1
    elif metric_config['combine_with'] == 'MoverScore':
        scores = scorer.score(refs, hyps, refs, hyps)
    elif 'XMoverScore' in metric_config['combine_with']:
        tgt = 'en'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.BAM', allow_pickle=True)
            projection = torch.tensor(temp, dtype=torch.float).to(device)
            temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.GBDD', allow_pickle=True)
            bias = torch.tensor(temp, dtype=torch.float).to(device)
        except:
            print(f'No remapping matrices for {srcl}-{tgt}')
            projection, bias = None, None
        mapping = metric_config['metric_hash']
        scores = scorer.compute_xmoverscore(mapping, projection, bias, sources, hyps, 1,
                                            bs=64)
        lm_scores = scorer.compute_perplexity(hyps, bs=1)
        scores = metric_combination(scores, lm_scores, [1, 0.1])
    elif metric_config['combine_with'] == 'None':
        print('NLI metric only; set scores to 0.')
        scores = np.zeros(len(hyps))
    else:
        raise ValueError('Metric not supported.')
    return scores
