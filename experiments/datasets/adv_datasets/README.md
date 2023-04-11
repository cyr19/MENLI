We release our generated adversarial datasets here.

The names of the datasets here are slightly different from that in our paper:

[``paws_back_google``](paws_back_google) = PAWS<sub>back</sub>

[``paws_ori_google``](paws_ori_google) = PAWS<sub>ori</sub>

[``xpaws/x``](xpaws) = XPAWS<sub>x</sub>

[``wmt20_google/de``](wmt20_google/de) = WMT20<sub>de</sub>

[``summ_google``](summ_google) = SE<sub>adv</sub>


Each adversarial dataset has a single **data.csv** file containing columns:

- `error` the perturbation type like "add" for addition and "num" for number error.

- `id` a unique id mapping the test case to its source instance from the original dataset. E.g., "xxxx:0" in [SE<sub>adv</sub>](experiments/datasets/adv_datasets/summ_google/data.csv) means the test case was generated from 
the document "xxxx" and its first reference summary in SummEval dataset.

- `source` the text in source language in MT or the source document in summarization; denoted as *src* in the paper.

- `ref` the reference translation or summary.
- `r` the google translate of `source` or the maximally similar reference summary to `source` in SummEval; the anchor text of `hyp_adv_free`.
- `hyp_para` the paraphrase of `ref`; denoted as *cand<sub>para</sub>* in the paper.
- `hyp_adv_based` the perturbed text from `ref` for reference-based setup; denoted as *cand<sub>adv</sub>* in the paper.
- `hyp_adv_free` the perturbed text from `r` for reference-free setup; denoted as *cand<sub>adv</sub>* in the paper.

Examples of test suites from our benchmark are given below:
<div align="center">
<img src="https://raw.githubusercontent.com/cyr19/MENLI/main/results/tables/table2_examples.png" width="70%"/>
</div>

For **reference-based** metrics, we expect *m(ref,cand<sub>para</sub>) > m(ref,cand<sub>adv</sub>)*, while for **reference-free** metrics, we expect *m(src,ref) > m(src,cand<sub>adv</sub>)*. We have shown that many standard metrics failed on our test suites:
<div align="center">
<img src="https://raw.githubusercontent.com/cyr19/MENLI/main/results/tables/table10_failure.png" width="70%"/>
</div>

Check [our paper](https://arxiv.org/abs/2208.07316) for more details!

## 2023-4-11 Update
We uploaded a **data_fixed_strict.csv** file to each adversarial dataset folder for ref-based MT evaluation, which removes some noise (e.g., case and space errors) in `hyp_para` (column "hyp_para_fixed") and `hyp_adv` (column "hyp_adv_based_fixed").

We tested BERTScore, COMET and NLI-R on them ([results](https://docs.google.com/spreadsheets/d/1ma4ckRx1r-Y-bAuVTUUwLXZ5tlDKe-3GYI-XKrL5G1E/edit?usp=sharing)).
After fixing the issues, standard metrics' performance drops (BERTScore 65.3% -> 56.2% and COMET 67.4% -> 62.1%), while NLI-R is almost not influenced at all (84.8% -> 84.3%); this is expected,
as now the overlap between `ref` and `hyp_adv` becomes larger, which challenges the standard metrics even more, especially those matching-based metrics like BERTScore.

We will release the fixed version for ref-free MT and summarization evaluation soon.
