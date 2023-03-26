# MENLI

This repository contains the code and data for our paper: [MENLI: Robust Evaluation Metrics from Natural Language Inference](https://arxiv.org/abs/2208.07316).

> **Abstract**: 
> Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).


## MENLI Benchmark

Check the [adversarial datasets](experiments/datasets/adv_datasets) as well as the [evaluation script](experiments/adv_test.py).
For each adversarial dataset, we release a single data.csv file containing columns:

- `error` the perturbation type like "add" for addition and "num" for number error.

- `id` a unique id mapping the test case to its source instance from the original dataset. E.g., "xxxx:0" in [SE<sub>adv</sub>](experiments/datasets/adv_datasets/summ_google/data.csv) means the test case was generated from 
the document "xxxx" and its first reference summary in SummEval dataset.

- `source` the text in source language in MT or the source document in summarization; denoted as *src* in the paper.

- `ref` the reference translation or summary.
- `r` the google translate of `source` or the maximally similar reference summary to `source` in SummEval; the anchor text of `hyp_adv_free`.
- `hyp_para` the paraphrase of `ref`; denoted as *cand<sub>para</sub>* in the paper.
- `hyp_adv_based` the perturbed text from `ref` for reference-based setup; denoted as *cand<sub>adv</sub>* in the paper.
- `hyp_adv_free` the perturbed text from `r` for reference-free setup; denoted as *cand<sub>adv</sub>* in the paper.

For **reference-based** metrics, we expect *m(ref,cand<sub>para</sub>) > m(ref,cand<sub>adv</sub>)*, while for **reference-free** metrics, we expect *m(src,ref) > m(src,cand<sub>adv</sub>)*.
We refer to our paper for more details and the reasons of the setup.

## MENLI Metrics
We provide the demo implementation of the [ensemble metrics](MENLI.py), which is, however, still imperfect.
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


## Experiments
To reproduce the experiments conducted in this work, please check the folder [experiments](experiments).


If you use the code or data from this work, please kindly cite us:
```angular2html
@misc{chen2022menli,
      title={MENLI: Robust Evaluation Metrics from Natural Language Inference}, 
      author={Yanran Chen and Steffen Eger},
      year={2022},
      eprint={2208.07316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you have any questions, feel free to contact us!

Yanran Chen ([yanran.chen@stud.tu-darmstadt.de](mailto:yanran.chen@stud.tu-darmstadt.de)) and Steffen Eger ([steffen.eger@uni-bielefeld.de](mailto:steffen.eger@uni-bielefeld.de))

Check our [group page](https://nl2g.github.io/) for other ongoing projects!
