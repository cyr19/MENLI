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
        from menli_package.moverscore_re import MoverScorer
        scorer = MoverScorer(idf=True)
    elif metric_config['combine_with'] == 'None':
        print('NLI metric only.')
        scorer = None
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

    elif metric_config['combine_with'] == 'None':
        print('NLI metric only; set scores to 0.')
        scores = np.zeros(len(hyps))
    else:
        raise ValueError('Metric not supported.')
    return scores
