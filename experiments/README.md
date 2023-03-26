# Experiments
This folder contains the code and data for the experiments conducted. 
To reproduce the figures and tables in the paper, please run the corresponding .py file. 
Before that, you need to unzip the metric scores from [here](../results/scores.zip), 
or you can run the evaluation scripts in this folder to get the scores.

## Metric implementation:
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
they output the three probability distribution scores in oder of [c, n, e] together.

The details of the settings and checkpoints used for those metrics can be found in [metrics/scorer_utils.py](metrics/scorer_utils.py). 
Generally, we used the currently recommended or default checkpoints (May 2022).
<!--
So COMET and BLEURT
dominating on MT in our evaluation is with no doubt, since most of the used datasets are just 
their training sets. 
-->


## Datasets:
The evaluation data is located in folder [datasets](datasets).
We uploaded the MT, summarization and adversarial datasets used in this work, except for WMT20-21,
as they are
bundled in the library [mt_metrics_eval](https://github.com/google-research/mt-metrics-eval)  (version v2),
which we used for all evaluation on WMT20-21.


For each of our generated adversarial datasets, we released a single data.csv file 
with columns [error,id,source,ref,r,hyp_para,hyp_adv_based,hyp_adv_free] (see folder [datasets/adv_datasets](datasets/adv_datasets)). 
The datasets' names there are different from that in our paper:

[``paws_back_google``](datasets/adv_datasets/paws_back_google) = PAWS<sub>back</sub>

[``paws_ori_google``](datasets/adv_datasets/paws_back_google) = PAWS<sub>ori</sub>

[``xpaws/x``](datasets/adv_datasets/xpaws) = XPAWS<sub>x</sub>

[``wmt20_google/de``](datasets/adv_datasets/xpaws) = WMT20<sub>de</sub>

[``summ_google``](datasets/adv_datasets/xpaws) = SE<sub>adv</sub>


Note that the datasets we used now are different from that for the Arxiv version
(we will publish the revised paper soon):

* The fluency-related phenomena were added into the adversarial datasets after the Arxiv version.
* xpaws/ja was recreated after the Arxiv version, since it accidentally contained wrong data in another language.
* We recreated the paws datasets:
  * Fixed some broken backtranslations
  * Kept the test suites almost the same for the two datasets except hyp_para this time. 
  as the hyp_para from the original PAWS dataset are more distant 
  from the ref in syntax structure compared to the ones from backtranslation, so we assume that the test 
  suites based on the original PAWS are more difficult to the metrics, which has then 
  been confirmed in our evaluation.

## Evaluation scripts:
To obtain comparable results with Table 8 and 9, you can use the following evaluation scripts.


### Machine Translation
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
from one direction. To compute the scores for direction hyp->ref or hyp->src, you could specify
`--direction hr`. To directly use the NLI metrics with selections of pooling strategy and combination with
other metrics, you may consider [MENLI.py](../MENLI.py). 

### Summarization
We used [summ.py](summ.py) for [SummEval](datasets/model_annotations.aligned.scored.jsonl) and [RealSumm](datasets/REALSumm) datasets:
```angular2html
python summ.py --dataset summ --metric DiscoScore --aggregate mean
python summ.py --dataset realsumm --metric SUPERT --use_article
```

### Adversarial evaluation
We used [adv_test.py](adv_test.py) to evaluate the metrics on our adversarial datasets.
```angular2html
python adv_test.py --dataset paws_back_google --metric MoverScore
python adv_test.py --dataset xpaws/de --metric SentSim_new --cross_lingual
python adv_test.py --dataset summ_google --metric SUPERT --use_article
```



