import numpy as np
import torch
import os
import metrics

def init_scorer(args):
    if args.metric == 'SentSim_new':
        from metrics.sentsim_new import SentSim
        scorer = SentSim(use_wmd=args.use_wmd, cross_lingual=args.cross_lingual)
        metric_hash = scorer.hash
    elif args.metric == 'sentBLEU':
        from sacrebleu.metrics import BLEU
        scorer = BLEU()
        metric_hash = 'default'
    elif args.metric == 'Rouge':
        from rouge import Rouge
        scorer = Rouge()
        metric_hash = 'L'
    elif args.metric == 'NLI1Score':
        from metrics.NLI1Score import NLI1Scorer
        scorer = NLI1Scorer(model=args.model, direction=args.direction,
                            device=args.device, cross_lingual=args.cross_lingual, checkpoint=args.checkpoint)
        metric_hash = scorer.hash
    elif args.metric == 'NLI2Score':
        from metrics.NLI2Score import NLI2Scorer
        scorer = NLI2Scorer(model=args.model, direction=args.direction,
                            device=args.device, cross_lingual=args.cross_lingual, checkpoint=args.checkpoint)
        metric_hash = scorer.hash
    elif args.metric == 'BERTScore':
        from metrics.bert_score.scorer import BERTScorer
        scorer = BERTScorer(lang='en', idf=True, nthreads=4)
        metric_hash = scorer.hash
    elif args.metric == 'MoverScore':
        from metrics.moverscore_re import MoverScorer
        scorer = MoverScorer(idf=True, device=args.device, model='bert_mnli')
        metric_hash = scorer.hash
    elif args.metric == 'BARTScore':
        from metrics.bart_score import BARTScorer
        # checkpoint = 'facebook/bart-large-cnn'
        scorer = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn', bidirection=args.bidirection)
        if args.bidirection:
            print('loading finetuned model...')
            try:
                scorer.load('metrics/models/bart.pth')
            except:
                raise FileNotFoundError('You need to manually download this checkpoint from https://github.com/neulab/BARTScore')
        metric_hash = 'bart-large-cnn' if not args.bidirection else 'bart-large-cnn+para_bi'
    elif metrics.args.metric == 'BLEURT':
        print('bluert')
        from bleurt.score import BleurtScorer
        # checkpoint = 'bleurt/BLEURT-20'
        try:
            scorer = BleurtScorer('metrics/bleurt/BLEURT-20')
        except:
            raise FileNotFoundError('You need to manually download this checkpoint from https://github.com/google-research/bleurt')
        metric_hash = 'BLEURT-20'
    elif args.metric == 'XMoverScore':
        print('xmover')
        from metrics.xmoverscore.scorer import XMOVERScorer
        scorer = XMOVERScorer(model_name='bert-base-multilingual-cased', lm_name='gpt2', device=args.device)
        metric_hash = '{}'.format(args.mapping)
    elif args.metric == 'COMET':
        print('comet')
        from metrics.comet import download_model, load_from_checkpoint
        if args.cross_lingual:
            # checkpoint = "wmt21-comet-qe-mqm"
            model_path = download_model('wmt21-comet-qe-mqm', saving_directory='metrics/models/')
            metric_hash = 'wmt21-comet-qe-mqm'
        else:
            # checkpoint = "wmt20-comet-da"
            model_path = download_model('wmt20-comet-da', saving_directory='metrics/models/')
            metric_hash = 'wmt20-comet-da'
        scorer = load_from_checkpoint(model_path)

    # summrization specific metrics
    elif args.metric == 'SUPERT':
        print('supert')
        import nltk
        nltk.download('stopwords', download_dir='/home/ychen/nltk_data')
        from summ_eval.supert_metric import SupertMetric
        scorer = SupertMetric()
        metric_hash = 'default'

    elif args.metric == 'DiscoScore':
        print('discoscore')
        import gdown
        from disco_score.metrics.metrics import Metrics
        if not os.path.exists('metrics/models/Conpono'):
            url = 'https://drive.google.com/drive/folders/1FE2loCSfdBbYrYk_qHg6W_PTqvA9w46T'
            gdown.download_folder(url, output='metrics/models/Conpono')
        from types import SimpleNamespace
        disco_args = SimpleNamespace(**{'model_name': 'metrics/models/Conpono', 'device': args.device, 'we': None})
        scorer = getattr(Metrics(args=disco_args), 'DS_Focus_NN')
        metric_hash = 'DS_Focus_NN'
        import nltk
        #nltk.download('punkt', download_dir='/storage/ukp/work/ychen/envs/anaconda3/envs/disco/nltk_data')
        nltk.download('punkt', download_dir='/home/ychen/nltk_data')

    else:
        scorer = 'None'
        metric_hash = 'None'
    print(args.metric)
    print(metric_hash)
    return scorer, metric_hash

# XMoverScore + LM
def metric_combination(a, b, alpha):
    return alpha[0] * np.array(a) + alpha[1] * np.array(b)


def scoring(args, scorer, refs, hyps, sources, p=None, srcl='de'):
    if args.metric == 'BERTScore':
        if scorer.idf:
            scorer.compute_idf(refs)
        scores = scorer.score(hyps, refs)[2].detach().numpy()  # F1

    elif args.metric == 'Rouge':
        hyps = [h if h!='' else ' ' for h in hyps]
        rouge_scores_dicts = scorer.get_scores(hyps, refs)
        scores = [s['rouge-l']['f'] for s in rouge_scores_dicts]

    elif args.metric == 'sentBLEU':
        scores = [scorer.sentence_score(hypothesis=h, references=[r]).score for h, r in zip(hyps, refs)]

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
        tgt = 'en'
        try:
            temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.BAM', allow_pickle=True)
            projection = torch.tensor(temp, dtype=torch.float).to(args.device)
            temp = np.load('xmoverscore/mapping/layer-12/europarl-v7.' + srcl + '-' + tgt + '.2k.12.GBDD', allow_pickle=True)
            bias = torch.tensor(temp, dtype=torch.float).to(args.device)
        except:
            print(f'No remapping matrices for {srcl}-{tgt}')
            projection, bias = None, None
        scores = scorer.compute_xmoverscore(args.mapping, projection, bias, sources, hyps, 1,
                                            bs=args.batch_size)
        lm_scores = scorer.compute_perplexity(hyps, bs=1)
        scores = metric_combination(scores, lm_scores, [1, 0.1])

    # summarization specific metrics
    elif args.metric == 'SUPERT':
        results = scorer.evaluate_batch(hyps, sources, aggregate=False)
        scores = [score['supert'] for score in results]

    elif args.metric == 'DiscoScore':
        scores = [scorer(ref=[ref.lower()], sys=hyp.lower()) for hyp, ref in zip(hyps, refs)]

    elif args.metric == 'None':
        print('testing; randomly set scores.')
        scores = np.random.normal(size=len(hyps))

    elif args.metric == 'NLItest':
        scores = np.random.normal(size=len(hyps)), np.random.normal(size=len(refs)), np.random.normal(size=len(refs))
    else:
        raise ValueError('Metric not supported.')
    return scores
