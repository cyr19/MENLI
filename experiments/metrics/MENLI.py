import numpy as np
from transformers import AutoTokenizer, __version__, AutoModelForSequenceClassification
import transformers
import os
import torch
transformers.logging.set_verbosity_error()
from .metric_utils import init_scorer, scoring
import gdown

class MENLI:
    def __init__(self,
                 model='R',
                 batch_size=64,
                 device=None,
                 direction='avg',
                 formula='e',
                 cross_lingual=False,
                 src=False,
                 nli_weight=1,
                 **metric_conf
                 ):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        self.cross_lingual = cross_lingual 
        self.src = src
        self.model_type = model
        self._model, self._tokenizer = self.get_model()
        self.direction = direction
        self.formula = formula
        self.nli_weight = float("%.1f" % nli_weight)
        self.metric_config = metric_conf
        self.other_metric = metric_conf['combine_with']
        self.metric, self.metric_hash = init_scorer(**metric_conf)
        self.metric_config['metric_hash'] = self.metric_hash
        self.nli_scores = []
        self.metric_scores = []

    @property
    def hash(self):
        return f"{self.model_type}_{self.direction}_{self.formula}_cross({self.cross_lingual})_src({self.src})_{self.other_metric}({self.metric_hash})_{self.nli_weight}"

    @property
    def hash_dict(self):
        return {
            'cross_lingual': self.cross_lingual,
            'src': self.src,
            'formula': self.formula,
            'direction': self.direction,
            'other_metric': self.other_metric,
            'metric_config': self.metric_config
        }

    def collate_input_features(self, pre, hyp):
        tokenized_input_seq_pair = self._tokenizer.encode_plus(pre, hyp,
                                                               max_length=self._tokenizer.model_max_length,
                                                               return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)

        return input_ids, token_type_ids, attention_mask

    def combine_nli(self, probs, direction, formula):
        if formula == 'e':
            return probs['e']
        elif formula == '-c':
            return [-c for c in probs['c']]
        elif formula == 'e-c':
            return [e-c for e, c in zip(probs['e'], probs['c'])]
        elif formula == 'e-n':
            return [e-n for e, n in zip(probs['e'], probs['n'])]
        else:
            raise NotImplementedError(f'Formula {self.formula} not supported.')

    def score_nli(self, srcs=[], refs=[], hyps=[], direction=None, formula=None):
        direction = direction if direction is not None else self.direction
        formula = formula if formula is not None else self.formula
        print(f'computing nli scores (direction {direction} + formula {formula})...')
        probs_rh, probs_hr, probs_avg = {}, {}, {}
        if self.src or self.cross_lingual:
            refs = srcs
        with torch.no_grad():
            if direction in ['rh', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(ref, hyp)
                    logits = self._model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=None)[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                concatenated = np.concatenate(probs, 0)
                if self.model_type == 'D' and not self.cross_lingual:
                    probs_rh['e'], probs_rh['n'], probs_rh['c'] = concatenated[:, 2], concatenated[:, 1], concatenated[:, 0]
                else:
                    probs_rh['e'], probs_rh['n'], probs_rh['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]

            if direction in ['hr', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(hyp, ref)
                    logits = self._model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=None)[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                concatenated = np.concatenate(probs, 0)
                if self.model_type == 'D' and not self.cross_lingual:
                    probs_hr['e'], probs_hr['n'], probs_hr['c'] = concatenated[:, 2], concatenated[:, 1], concatenated[:, 0]
                else:
                    probs_hr['e'], probs_hr['n'], probs_hr['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]

            if direction == 'rh':
                final_score = self.combine_nli(probs_rh, direction, formula)
            elif direction == 'hr':
                final_score = self.combine_nli(probs_hr, direction, formula)
            elif direction == 'avg':
                print('averaging....')
                for p in probs_hr.keys():
                    probs_avg[p] = [(s1+s2)/2.0 for s1, s2 in zip(probs_rh[p], probs_hr[p])]
                final_score = self.combine_nli(probs_avg, direction, formula)
        final_score = list(final_score)
        self.nli_scores += final_score
        return final_score

    def score_metric(self, srcs=[], refs=[], hyps=[], srcl='de'):
        print('computing metric scores....')
        scores = scoring(scorer=self.metric, sources=srcs, refs=refs, hyps=hyps, srcl=srcl, metric_config=self.metric_config)
        self.metric_scores += list(scores)
        return scores

    def combine(self, metric_scores, nli_scores, nli_weight=None):
        if nli_weight!=None:
          print('new nli weight...')
          nli_weight = nli_weight
        else:
          nli_weight = self.nli_weight
        norm_metric_scores = self.min_max_normalize(metric_scores)
        norm_nli_scores = self.min_max_normalize(nli_scores)
        print(f'weight for metric: {1-nli_weight}')
        final_scores = [nli_weight * n + (1 - nli_weight) * m for n, m in zip(norm_nli_scores, norm_metric_scores)]
        return final_scores

    def combine_all(self):
        print('combing scores...')
        norm_metric_scores = self.min_max_normalize(self.metric_scores)
        norm_nli_scores = self.min_max_normalize(self.nli_scores)
        final_scores = [self.nli_weight*n + (1-self.nli_weight)*m for n, m in zip(norm_nli_scores, norm_metric_scores)]
        self.metric_scores, self.nli_scores = [], []
        return final_scores

    def score_all(self, srcs, refs, hyps, srcl='de'):
        if self.nli_weight == 0.0:
            return self.score_metric(srcs, refs, hyps, srcl)
        elif self.nli_weight == 1.0:
            return self.score_nli(srcs, refs, hyps)
        else:
            self.score_metric(srcs, refs, hyps, srcl)
            self.score_nli(srcs, refs, hyps)
            return self.combine_all()

    def min_max_normalize(self, scores):
        if len(scores) > 1:
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            normalized_scores = scores
        return normalized_scores

    def get_model(self):
        if self.model_type == 'R':
            if not self.cross_lingual:
                tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', use_fast=False, cache_dir='.cache')
                model = AutoModelForSequenceClassification.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', num_labels=3, cache_dir='.cache')
            else:
                tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=False, cache_dir='.cache')
                model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3, cache_dir='.cache')
                checkpoint_path = 'crosslingual_R'
                if not os.path.exists(checkpoint_path):
                    url = 'https://drive.google.com/drive/folders/1g7h1D8yfEP_s68sG6zacBQ8IgIT2QcGh?usp=sharing'
                    gdown.download_folder(url, output='crosslingual_R')
                model.load_state_dict(torch.load('crosslingual_R/model.pt', map_location=torch.device(self.device)))
        elif self.model_type == 'D':
            if self.cross_lingual:
                tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", use_fast=False, cache_dir='.cache')
                model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", num_labels=3, cache_dir='.cache')
            else:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli", use_fast=False, cache_dir='.cache')
                model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli", num_labels=3, cache_dir='.cache')
        model.eval()
        model = model.to(self.device)
        return model, tokenizer

if __name__ == '__main__':
    scorer = MENLI(direction='avg', formula='e', src=False, nli_weight=0.3, combine_with='BERTScore-F')

    refs = ['I like cats', "I don't have 100 Euros."]
    hyps = ['I like animals.', "I have more than 500 Euros."]
    scores = scorer.score_all(srcs=[], refs=refs, hyps=hyps)
    print(scores)



