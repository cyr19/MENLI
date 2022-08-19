import numpy as np
import os
import torch

def init_scorer(args):
    if args.metric == 'SentSim_new':
        from sentsim_new import SentSim
        scorer = SentSim(use_wmd=args.use_wmd, cross_lingual=args.cross_lingual)
        metric_hash = scorer.hash

    elif args.metric == 'SummaCZS':
        from model_summac import SummaCZS
        if args.formula == 'e':
            scorer = SummaCZS(granularity="sentence", model_name="anli", use_ent=True, use_cont=False)
        elif args.formula == 'c':
            scorer = SummaCZS(granularity="sentence", model_name="anli", use_ent=False, use_cont=True)
        else:
            scorer = SummaCZS(granularity="sentence", model_name="anli", use_ent=True, use_cont=True)
        metric_hash = args.formula

    elif args.metric in ['QAFactEval']:
        from qafacteval import QAFactEval
        kwargs = {"cuda_device": 0, "use_lerc_quip": True, "verbose": False, "generation_batch_size": 16,
                  "answering_batch_size": 16, "lerc_batch_size": 4}
        model_folder = 'QAFactEval/models'
        scorer = QAFactEval(lerc_quip_path=f"{model_folder}/quip-512-mocha", generation_model_path=f"{model_folder}/generation/model.tar.gz",
                            answering_model_dir=f"{model_folder}/answering", lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
                            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz", **kwargs)
        metric_hash = 'default'

    elif args.metric in ['sentBLEU', 'BLEU']:
        from sacrebleu.metrics import BLEU
        scorer = BLEU()
        metric_hash = 'default'

    elif args.metric == 'Rouge':
        from rouge import Rouge
        scorer = Rouge()
        metric_hash = 'L'

    elif args.metric == 'SUPERT':
        from summ_eval.supert_metric import SupertMetric
        scorer = SupertMetric()
        metric_hash = 'default'

    elif args.metric == 'DiscoScore':
        import gdown
        from disco_score.metrics.metrics import Metrics
        if not os.path.exists('models/Conpono'):
            url = 'https://drive.google.com/drive/folders/1FE2loCSfdBbYrYk_qHg6W_PTqvA9w46T'
            gdown.download_folder(url, output='models/Conpono')

        from types import SimpleNamespace
        disco_args = SimpleNamespace(**{'model_name': 'models/Conpono', 'device': args.device, 'we': None})
        scorer = getattr(Metrics(args=disco_args), 'DS_Focus_NN')
        metric_hash = 'DS_Focus_NN'
        import nltk
        nltk.download('punkt', download_dir='/storage/ukp/work/ychen/envs/anaconda3/envs/disco/nltk_data')

    elif args.metric == 'RoMe':
        print('rome')
        from RoMe.rome import RoMe
        scorer = RoMe()
        metric_hash = 'default'

    elif args.metric == 'NLI1Score':
        from NLI1Score import NLI1Scorer
        scorer = NLI1Scorer(model=args.model, direction=args.direction,
                            device=args.device, cross_lingual=args.cross_lingual, checkpoint=args.checkpoint)
        metric_hash = scorer.hash

    elif args.metric == 'NLI2Score':
        from NLI2Score import NLI2Scorer
        scorer = NLI2Scorer(model=args.model, direction=args.direction,
                            device=args.device, cross_lingual=args.cross_lingual, checkpoint=args.checkpoint)
        metric_hash = scorer.hash

    elif args.metric == 'NLIDocScore':
        from NLIDocScore import NLIDocScorer
        scorer = NLIDocScorer(model=args.model, direction=args.direction, use_article=args.use_article, checkpoint=args.checkpoint)
        metric_hash = scorer.hash

    elif args.metric == 'BERTScore':
        from bert_score.scorer import BERTScorer
        scorer = BERTScorer(lang='en', idf=True, nthreads=4)
        metric_hash = scorer.hash

    elif args.metric == 'MoverScore':
        from moverscore_re import MoverScorer
        scorer = MoverScorer(idf=True, device=args.device, model='bert_mnli')
        metric_hash = scorer.hash

    elif args.metric == 'BARTScore':
        from bart_score import BARTScorer
        # checkpoint = 'facebook/bart-large-cnn'
        scorer = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn', bidirection=args.bidirection)
        if args.bidirection:
            print('loading finetuned model...')
            scorer.load('models/bart.pth')
        metric_hash = 'bart-large-cnn' if not args.bidirection else 'bart-large-cnn+para_bi'

    elif args.metric == 'BLEURT':
        print('bluert')
        from bleurt.score import BleurtScorer
        scorer = BleurtScorer('bleurt/BLEURT-20')
        metric_hash = 'BLEURT-20'

    elif args.metric == 'XMoverScore':
        print('xmover')
        import torch
        from xmoverscore.scorer import XMOVERScorer
        scorer = XMOVERScorer(model_name='bert-base-multilingual-cased', lm_name='gpt2', device=args.device)
        metric_hash = '{}'.format(args.mapping)

    elif args.metric == 'COMET':
        print('comet')
        from comet import download_model, load_from_checkpoint
        # checkpoint = "wmt20-comet-da"
        if args.cross_lingual:
            model_path = download_model('wmt21-comet-qe-mqm', saving_directory='models/')
            metric_hash = 'wmt21-comet-qe-mqm'
        else:
            model_path = download_model('wmt20-comet-da', saving_directory='models/')
            metric_hash = 'wmt20-comet-da'
        scorer = load_from_checkpoint(model_path)

    else:
        scorer = 'None'
        metric_hash = 'None'

    print(args.metric)
    print(metric_hash)
    return scorer, metric_hash

# XMoverScore
def metric_combination(a, b, alpha):
    return alpha[0] * np.array(a) + alpha[1] * np.array(b)


def scoring(args, scorer, refs, hyps, sources, p=None):

    if args.metric == 'BERTScore':
        if scorer.idf:
            scorer.compute_idf(refs)
        scores = scorer.score(hyps, refs)[2].detach().numpy()  # F1

    elif args.metric == 'SummaCZS':
        scores = scorer.score(sources=sources, generateds=hyps)['scores']

    elif args.metric == 'Rouge':
        hyps = [h if h!='' else ' ' for h in hyps]
        rouge_scores_dicts = scorer.get_scores(hyps, refs)
        scores = [s['rouge-l']['f'] for s in rouge_scores_dicts]

    elif args.metric == 'sentBLEU':
        scores = [scorer.sentence_score(hypothesis=h, references=[r]).score for h, r in zip(hyps, refs)]

    elif args.metric == 'SUPERT':
        results = scorer.evaluate_batch(hyps, sources, aggregate=False)
        print(results[0])
        scores = [score['supert'] for score in results]

    elif args.metric == 'DiscoScore':
        scores = [scorer(ref=[ref.lower()], sys=hyp.lower()) for hyp, ref in zip(hyps, refs)]
        print(type(scores[0]))

    elif args.metric == 'SentSim_new':
        if args.cross_lingual:
            scores = scorer.score(sources, hyps)
        else:
            scores = scorer.score(refs, hyps)

    elif args.metric == 'MoverScore':
        scores = scorer.score(refs, hyps, refs, hyps)

    elif args.metric in ['NLI1Score', 'NLI2Score', 'BARTScore']:
        if args.cross_lingual or args.use_article:
            scores = scorer.score(sources, hyps)
        else:
            scores = scorer.score(refs, hyps)

    elif args.metric == 'BLEURT':
        scores = scorer.score(references=refs, candidates=hyps)

    elif args.metric == 'COMET':
        if args.cross_lingual:
            refs = np.zeros(len(sources))
        data = [{
            'src': s,
            'mt': hyp,
            'ref': ref
        } for s, hyp, ref in zip(sources, hyps, refs)]

        scores = scorer.predict(data, batch_size=args.batch_size)[0]

    elif args.metric == 'XMoverScore':
        srcl, tgt = 'de', 'en'
        temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.BAM', allow_pickle=True)
        projection = torch.tensor(temp, dtype=torch.float).to(args.device)

        temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.GBDD',
                       allow_pickle=True)
        bias = torch.tensor(temp, dtype=torch.float).to(args.device)

        scores = scorer.compute_xmoverscore(args.mapping, projection, bias, sources, hyps, 1,
                                            bs=args.batch_size)
        lm_scores = scorer.compute_perplexity(hyps, bs=1)
        scores = metric_combination(scores, lm_scores, [1, 0.1])

    elif args.metric == 'SentSim':
        scores = scorer.score(refs, hyps, sources)

    elif args.metric == 'None':
        print('testing; randomly set scores.')
        scores = np.random.normal(size=len(refs))
    elif args.metric == 'NLItest':
        scores = np.random.normal(size=len(refs)), np.random.normal(size=len(refs)), np.random.normal(size=len(refs))
    else:
        raise ValueError('Metric not supported.')

    return scores








