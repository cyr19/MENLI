# MENLI

This repository contains the source code for our Paper: [MENLI: Robust Evaluation Metrics from Natural Language Inference](https://arxiv.org/abs/2208.07316).


Check folder [experiments](experiments) for the updates of the current work and 
the code and data used. There are some differences from the
Arxiv version. We will publish the revised paper later.

## Example of Usage 

```angular2html
from MENLI import MENLI
scorer = MENLI(direction="rh", formula="e", nli_weight=0.2, \
                combine_with="MoverScore", model="D", cross_lingual=False)
# refs and hyps in form of list of String
scorer.score_all(refs=refs, hyps=hyps) 
```

E.g., run XNLI-D on WMT15 with

```angular2html
python wmt.py --year 2015 --cross_lingual --direction avg --formula e --model D
```

Run the combined metric with BERTScore F1 on wmt17 with
```angular2html
python wmt.py --year 2017 --combine_with BERTScore-F --nli_weight 0.2 --model R
```

We provide the combination with MoverScore, BERTScore-F1, and XMoverScore here, to combine with other metrics, just fit the code into [metric_utils.py](https://github.com/cyr19/MENLI/blob/main/metric_utils.py).

Specifically, in init_scorer() function, you need to initialize a scorer like
```angular2html
def init_scorer(**metric_config):

    from bert_score.scorer import BERTScorer
    scorer = BERTScorer(lang='en', idf=True)
    metric_hash = scorer.hash

    return scorer, metric_hash
```
Then call the metric scoring function in scoring():
```angular2html
def scoring(scorer, refs, hyps, sources):

    if scorer.idf:
        scorer.compute_idf(refs)
    scores = scorer.score(hyps, refs)[2].detach().numpy().tolist()  # F1
    # Note: the outputs of the metric should be a list.
    return scores
```


Contact person: Yanran Chen (yanran.chen@stud.tu-darmstadt.de)




