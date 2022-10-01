# Experiments

This folder contains the code and data for experiments such as the evaluation scripts used in this work.

## About the metric implementation:
In this work, we experimented with [MoverScore](https://arxiv.org/abs/1909.02622), [BERTScore](https://arxiv.org/abs/1904.09675), 
[BARTScore](https://arxiv.org/abs/2106.11520), [SentSim](https://aclanthology.org/2021.naacl-main.252/), 
[XMoverScore](https://aclanthology.org/2020.acl-main.151/), [COMET](https://arxiv.org/abs/2009.09025) and 
[BLEURT](https://arxiv.org/abs/2004.04696) for MT, while for 
summarization, besides MoverScore, BARTScore and BERTScore, we also tested 
[DiscoScore](https://arxiv.org/abs/2201.11176) and [SUPERT](https://arxiv.org/abs/2005.03724).

* [moverscore_re.py](metrics/moverscore_re.py) is a re-implemented version of 
[the original one](https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore.py), 
with which we can call it as an object.
* [bartscore.py](metrics/bart_score.py) is almost the [original version](https://github.com/neulab/BARTScore/blob/main/bart_score.py), 
but we add a parameter bidirection to compute the F variant, which takes the average of P and R scores.
* [sentsim_new.py](metrics/sentsim_new.py) uses the implementation from [the library](https://github.com/potamides/unsupervised-metrics/blob/master/metrics/sentsim.py) 
published by the authors of [USCORE](https://arxiv.org/abs/2202.10062),
with an additional parameter cross_lingual to enable using it in both ref-based and ref-free setting.
* [xmoverscore](metrics/xmoverscore) contains the [original code](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
with an additional line in score_utils.py (line 158) to allow use it without remapping, as we tested it 
with some language pairs for which there are no remapping matrices from the authors (e.g., ja-en and 
km-en in WMT20). We used its version reported in the original paper that leverages the final layer embedding.
* For SUPERT, we used the implementation from [SummEval](https://github.com/Yale-LILY/SummEval/tree/master/evaluation/summ_eval).
* For [COMET](https://github.com/Unbabel/COMET), [BLEURT](https://github.com/google-research/bleurt), [DiscoScore](https://github.com/aiphes/discoscore)
and [BERTScore](https://github.com/Tiiiger/bert_score), we used their original implementation.
* NLI1Score and NLI2Score are denoted as (X)NLI-R and (X)NLI-D in our paper, respectively; 
they output the three probability distribution scores in oder of [c, n, e] at one time.

The details of the settings and checkpoints used for those metrics can be found in [metrics/scorer_utils.py](metrics/scorer_utils.py). 
Generally, we used the current recommended or default checkpoints (May 2022). So COMET and BLEURT
dominating on MT in our evaluation is with no doubt, since most of the used datasets are just 
their training sets.

## Evaluation
### About the datasets:
The evaluation data is located in folder [datasets](datasets).
We uploaded the MT, summarization and adversarial datasets used in this work, except for WMT20-21,
as they are
bundled in the library [mt_metrics_eval](https://github.com/google-research/mt-metrics-eval),
which we used for all evaluation on WMT20-21 (version v2).


For each of our generated adversarial datasets, we released a single data.csv file 
with columns [error,id,source,ref,r,hyp_para,hyp_adv_based,hyp_adv_free] (see folder [datasets/adv_datasets](datasets/adv_datasets)). 

Note that the datasets we used now are different from that for the Arxiv version
(we will publish the revised paper later):

* The fluency-related phenomena were added into the adversarial datasets after the Arxiv version.
* xpaws/ja was recreated after the Arxiv version, since it accidentally contained wrong data in another language.
* We recreated the paws datasets:
  * Fixed some broken backtranslations
  * Kept the test suites almost the same for the two datasets except hyp_para this time. 
  as the hyp_para from the original PAWS dataset are more distant 
  from the ref in syntax structure compared to the ones from backtranslation, so we assume that the test 
  suites based on the original PAWS are more difficult to the metrics, which has then 
  been confirmed in our evaluation.

### About the evaluation scripts:
#### Machine Translation
We used [wmt.py](wmt.py) for WMT15-17 and [wmt_sys.py](wmt_sys.py) for WMT20-21. 

To run metrics on WMT15-17 (segment-level):
```angular2html
python wmt.py --year 2015 --metric BERTScore
python wmt.py --year 2017 --metric XMoverScore --mapping CLP --cross_lingual
```
To run metrics on WMT20-21 zh-en at segment-level:
```angular2html
python wmt_sys.py --metric BERTScore --dataset wmt21.news --mqm
```
To run metrics on WMT20-21 at system-level:
```angular2html
python wmt_sys.py --metric BERTScore --dataset wmt20
```

To run NLI1/2Score in cross-lingual environment, you need to specify the model 
name and the checkpoint index like:
```angular2html
# These two settings are what we reported in the paper.
--metric NLI1Score --cross_lingual --model xlm-roberta-base --checkpoint 2
--metric NLI2Score --cross_lingual --model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli --checkpoint 0
```
Note: 
NLI1/2Score doesn't support the selection of the pooling strategy, it always outputs c, n, e
for one direction. To compute the scores for direction hyp->ref or hyp->src, you could use
`--direction hr`. To directly use the NLI metrics with selections of pooling strategy and combination with
other metrics, you may consider [MENLI.py](../MENLI.py). 

#### Summarization
We used [summ.py](summ.py) to evaluate metrics on [SummEval]() and [RealSumm]() datasets:
```angular2html
python summ.py --dataset summ --metric DiscoScore --aggregate mean
python summ.py --dataset realsumm --metric SUPERT --use_article
```

#### Adversarial evaluation
We used [adv_test.py](adv_test.py) to evaluate the metrics on our adversarial datasets.
```angular2html
python adv_test.py --dataset paws_back_google --metric MoverScore
python adv_test.py --dataset xpaws/de --metric SentSim_new --cross_lingual
python adv_test.py --dataset summ_google --metric SUPERT --use_article
```

## Evaluation Results


###  Machine translation

Note:
* The results for XMoverScore are different from that in the Arxiv version after fixing
the bugs of incorrect use of remapping matrices for part of the language pairs.

* NLI metrics are with formula e from direction ref/src<-->hyp.


####Segment-level evaluation:

| Metric | WMT15 | WMT16 | WMT17 | WMT20 | WMT21 | AVG |
| ---  |  --- | --- |  --- |  --- |  --- |  --- |
| Reference-based
| Supervised 
| COMET | 0.832 | 0.805 | 0.834 | 0.500 | 0.412 | 0.676 |
| BLEURT  |  0.850 | 0.833|  0.842 |  0.571 |  0.442 | 0.708  |
| Unsupervised
| BLEU | 0.495 | 0.466 | 0.469 | 0.240 | 0.228 | 0.380 |
| ROUGE | 0.560 | 0.528 | 0.526 | 0.258 | 0.252 | 0.425 |
| MoverScore | 0.760 | 0.720 | 0.735 | 0.315 | 0.307 | 0.567 |
| BERTScore | 0.786 | 0.746 | 0.769 | 0.481 | 0.317 | 0.620 |
| BARTScore-P | 0.753 | 0.716 | 0.721 | 0.414 | 0.328 | 0.587 |
| BARTScore-F | 0.768 | 0.769 | 0.795 | 0.418 | 0.264 | 0.612 |
| SentSim(BERTS) | 0.815 | 0.769 | 0.795 | 0.418 | 0.264 | 0.612 |
| SentSim(WMD) | 0.802 | 0.751 | 0.779 | 0.384 | 0.320 | 0.607 |
| NLI-based
| NLI-R | 0.583 | 0.514 | 0.569 | 0.332 | 0.256 | 0.451 |
| NLI-D | 0.553 | 0.491 | 0.549 | 0.320 | 0.282 | 0.439 |
| Reference-free
| Supervised
| COMET | 0.633 | 0.611 | 0.616 | 0.744 | 0.498 | 0.620 |
| Unsupervised
| XMoverS(CLP) | 0.542 | 0.513 | 0.549 | 0.324 | 0.181 | 0.422 |
| XMoverS(UMD) | 0.508 | 0.479 | 0.509 | 0.316 | 0.189 | 0.400 |
| SentSim(BERTS) | 0.555 | 0.521 | 0.555 | 0.310 | 0.165 | 0.421 |
| SentSim(WMD) | 0.548 | 0.510 | 0.543 | 0.291 | 0.241 | 0.427 |
| NLI-based
| XNLI-R | 0.231 | 0.197 | 0.265 | 0.232 | 0.182 | 0.221 |
| XNLI-D | 0.250 | 0.180 | 0.242 | 0.012 | 0.059 | 0.149 |



####System-level evaluation:

| Metric | WMT20 | WMT21 | AVG | 
| ---  |   --- |  --- |  --- |
| Reference-based
| Supervised 
| COMET | 0.886 | 0.730 | 0.808
| BLEURT  |  0.888 | 0.727 |  0.807
| Unsupervised
| BLEU | 0.858 | 0.655 | 0.757
| ROUGE | 0.878 | 0.670 | 0.774 
| MoverScore | 0.910 | 0.703 | 0.806 
| BERTScore | 0.887 | 0.711 | 0.799 
| BARTScore-P | 0.835 | 0.686 | 0.761 
| BARTScore-F | 0.879 | 0.724 | 0.802 
| SentSim(BERTS) | 0.543 | 0.258 | 0.401
| NLI-based
| NLI-R | 0.886 | 0.627 | 0.756 | 
| NLI-D | 0.891 | 0.650 | 0.770 | 
| Reference-free
| Supervised
| COMET | 0.735 | 0.661 | 0.698 | 
| Unsupervised
| XMoverS(CLP) | 0.673 | 0.672 | 0.673 | 
| XMoverS(UMD) | 0.676 | 0.667 | 0.672 | 
| SentSim(BERTS) | 0.088 | -0.130 | -0.021 |
| NLI-based
| XNLI-R | 0.585 | 0.084 | 0.335 |  
| XNLI-D | 0.726 | 0.437 | 0.581 | 

####Adversarial evaluation:

The results can be found in [experiment.ipynb](experiments.ipynb).


### Summarization


* Reference-based evaluation:
  * NLI metrics are with formula e-c from direction hyp-->ref

|  | SummEval(mean) | | | | |RealSumm|--- |SEval|
| ---| ---|---|---|---| --- |---|---|---|
|Metric | coherence | consistency | fluency | relevance | avg | summary-level | system-level | |
|BLEU | 29.4 | 4.4 | 24.4 | 39.7 | 24.5 | 48.0 | 12.4 | 18.2
|ROUGE | 19.1 | 8.8 | -3.7 | 11.8 | 9.0 | 54.0 | 45.7 | 18.5
|MoverScore | 20.6 | 45.6 | 42.1 | 36.8 | 36.3 | 58.5 | 50.1 | 28.7
|BERTScore | 61.8 | 22.1 | 27.3 | 60.3 | 42.9 | 57.4 | 38.0 | 59.8
|BARTScore-P | 48.5 | 17.7 | 37.6 | 50.0 | 38.5 | 47.8 | 53.1 | 69.7
|BARTScore-P | 51.5 | 20.6 | 31.7 | 52.9 | 39.2 | 58.3 | 68.7 | 78.8
|DiscoScore | 67.7 | 27.9 | 53.9 | 63.2 | 53.2 | -19.9 | -6.6 | 33.4
|NLI-based |
|NLI-R | 14.7 | 63.2 | 49.4 | 27.9 | 38.8 | 52.5 | 85.6 | 86.4 
|NLI-D | 25.0 | 70.6 | 56.8 | 47.1 | 49.9 | 48.9 | 84.0 | 80.6

The results on SummEval with aggregation max can be found in [experiment.ipynb](experiments.ipynb).

* Reference-free evaluation:

|  | SummEval | | | | | RealSumm| |Adv.| ||
| ---| ---|---|---|---| --- |---|---|---|---| ---|
|Metric | coherence | consistency | fluency | relevance | avg | summary-level | system-level | SEval| Rank19 | AVG
| BARTScore-FN | 73.5 | 13.2 | 39.1 | 66.2 |48.0 | 17.8 | -2.3 | 42.7 | 79.6 | 61.2
| SUPERT | 14.7 | 60.3 | 46.5 | 27.9 | 37.4 | 52.2 | 62.6 | 29.6 | 66.8 | 48.2
|NLI-based |
|NLI-R | 22.1 | 23.5 | 39.1 | 50.0 | 33.7 |30.0 | 68.8 | 72.0 | 86.6 | 79.3 
|NLI-D | 16.2 | 64.7 | 33.2 | 32.4 | 36.6 | -7.6 | 56.8 | 62.4 | 88.5 | 75.5



## Todo:
* Update experiments.ipynb and combine.py
* Update results folder
* Update scores.zip file

Run [experiments.ipynb](https://github.com/cyr19/MENLI/blob/main/experiments/experiments.ipynb) 
to reproduce the results for MT and summarization. Before that, 
download the stored metric scores from 
[here](https://drive.google.com/file/d/11ucw-Rgyj5G8TJ1KxNowAfnQjCnyKtv2/view?usp=sharing) 
and unzip it to results/ folder (Arxiv version).






