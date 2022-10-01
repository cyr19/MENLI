import numpy as np
from transformers import AutoTokenizer, __version__, AutoModelForSequenceClassification
import transformers
transformers.logging.set_verbosity_error()
import os
import torch

class NLI2Scorer:

    def __init__(self,
                 model = None,
                 batch_size = 64,
                 device = None,
                 direction = 'rh',
                 cross_lingual = False,
                 checkpoint = 0
                 ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        if cross_lingual:
            self.model = model
        else:
            model = "microsoft/deberta-large-mnli"

        self.model_name = model.split('/')[1] if '/' in model else model

        self.checkpoint = checkpoint
        self._tokenizer = get_tokenizer(model)
        self._model = get_model(model, self.device, cross_lingual=cross_lingual, checkpoint=checkpoint)
        self.direction = direction
        self.cross_lingual = cross_lingual

    @property
    def hash(self):
        return 'crosslingual({}+{})_{}'.format(self.model_name, self.checkpoint, self.direction) if self.cross_lingual else 'monolingual_{}'.format(self.direction)

    def collate_input_features(self, pre, hyp):
        tokenized_input_seq_pair = self._tokenizer.encode_plus(pre, hyp,
                                                         max_length=self._tokenizer.model_max_length,
                                                         return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)
        return input_ids, token_type_ids, attention_mask

    def score(self, refs, hyps):
        probs = []
        with torch.no_grad():
            for ref, hyp in zip(refs, hyps):
                if self.direction == 'rh':
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(ref, hyp)
                else:
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(hyp, ref)

                logits = self._model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)[0]
                prob = torch.softmax(logits, 1).detach().cpu().numpy()
                probs.append(prob)
        probs = np.concatenate(probs, 0)
        if self.cross_lingual:
            return probs[:, 2], probs[:, 1], probs[:, 0]  # c, n, e
        return probs[:, 0], probs[:, 1], probs[:, 2]  #c, n, e

def get_tokenizer(model):
    model_dir = 'models/' + model
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, cache_dir='.cache')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir='.cache')
    return tokenizer

def get_model(model_name, device = 'cuda', cross_lingual=False, checkpoint=0):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, cache_dir='.cache')
    model.eval()
    model = model.to(device)
    return model






