#  <img src="https://raw.githubusercontent.com/cyr19/MENLI/main/results/plots/fitness-icon-robustness-151745156.jpg"  width="6%"/> MENLI

This repository contains the code and data for our TACL paper: [MENLI: Robust Evaluation Metrics from Natural Language Inference](https://arxiv.org/abs/2208.07316).

<div align="center">
<img src="https://raw.githubusercontent.com/cyr19/MENLI/main/results/plots/figure3.png" width="40%"/>
</div>

> **Abstract**: 
> Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).



## 🚀 MENLI Benchmark

We release our [adversarial datasets](experiments/datasets/adv_datasets). Please check [here](experiments/) and the [evaluation script](experiments/adv_test.py) for
more details about how to run metrics on them.

**2023-4-11 Update: we uploaded a new version of adversarial datasets for ref-based MT evaluation, which fixes some space and case errors ([more details](experiments/datasets/adv_datasets)).**
<div align="center">
<img src="https://raw.githubusercontent.com/cyr19/MENLI/main/results/tables/table10_failure.png" width="70%"/>
</div>

## 🚀 MENLI Metrics
We provide the demo implementation of the [ensemble metrics](MENLI.py); however, the implementation is still imperfect.
### Example of Usage 

```angular2html
from MENLI import MENLI
scorer = MENLI(direction="rh", formula="e", nli_weight=0.2, \
                combine_with="MoverScore", model="D", cross_lingual=False)
# refs and hyps in form of list of String
scorer.score_all(refs=refs, hyps=hyps) 
```

E.g., run XNLI-D on WMT15:

```angular2html
python wmt.py --year 2015 --cross_lingual --direction avg --formula e --model D
```

Run the combined metric with BERTScore F1 on wmt17:
```angular2html
python wmt.py --year 2017 --combine_with BERTScore-F --nli_weight 0.2 --model R
```

We implemented the combination with MoverScore, BERTScore-F1, and XMoverScore here, to combine with other metrics, just fit the code into [metric_utils.py](https://github.com/cyr19/MENLI/blob/main/metric_utils.py).

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


## 🚀 Experiments
To reproduce the experiments conducted in this work, please check the folder [experiments](experiments).


If you use the code or data from this work, please include the following citation:

```bigquery
@article{chen_menli:2023,
    author = {Chen, Yanran and Eger, Steffen},
    title = "{MENLI: Robust Evaluation Metrics from Natural Language Inference}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {804-825},
    year = {2023},
    month = {07},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00576},
    url = {https://doi.org/10.1162/tacl\_a\_00576},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00576/2143297/tacl\_a\_00576.pdf},
}

```

If you have any questions, feel free to contact us!

Yanran Chen ([yanran.chen@stud.tu-darmstadt.de](mailto:yanran.chen@stud.tu-darmstadt.de)) and Steffen Eger ([steffen.eger@uni-bielefeld.de](mailto:steffen.eger@uni-bielefeld.de))

Check our group page ([NLLG](https://nl2g.github.io/)) for other ongoing projects!
