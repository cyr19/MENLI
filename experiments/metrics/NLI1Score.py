import numpy as np
from transformers import AutoTokenizer, __version__, AutoModelForSequenceClassification
import transformers
import os
import torch
transformers.logging.set_verbosity_error()
import gdown

class NLI1Scorer:

    def __init__(self,
                 model = None,
                 batch_size = 64,
                 device = None,
                 direction = 'rh',
                 cross_lingual = False,
                 checkpoint = 0,
                 ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        if cross_lingual:
            self.model = model
        else:
            model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
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
        if self.model_name == 'xlm-roberta-large-xnli-anli':
            return probs[:, 0], probs[:, 1], probs[:, 2]

        return probs[:, 2], probs[:, 1], probs[:, 0] #c, n, e
        

def get_tokenizer(model):
    print(model)
    model_dir = 'models/' + model

    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, cache_dir='.cache')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir='.cache')
    return tokenizer

def get_model(model_name, device = None, cross_lingual=False, checkpoint=None):
    print(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, cache_dir='.cache')

    if cross_lingual and model_name == 'xlm-roberta-base':
        print('>>>')

        if checkpoint == 2:
            print('loading checkpoint 2...')
            checkpoint_path = 'models/crosslingual_R'
            if not os.path.exists(checkpoint_path):
                url = 'https://drive.google.com/drive/folders/1g7h1D8yfEP_s68sG6zacBQ8IgIT2QcGh?usp=sharing'
                gdown.download_folder(url, output=checkpoint_path)
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location=torch.device(device)))

        else:
            raise ValueError('checkpoint not found.')
    model.eval()
    model = model.to(device)
    return model



