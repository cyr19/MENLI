from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import string
from pyemd import emd
from torch import nn
from math import log
from itertools import chain

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
import os
import sys
import requests
import zipfile


class MoverScorer:
    def __init__(
            self,
            model='bert_mnli',
            batch_size = 128,
            nthread = 4,
            idf = True,
            stopwords = [],
            device = None,
            wordpiece = 1,
            remove_punctuation = True,
            n_gram = 1,
    ):

        assert model is not None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name = model
        if model == 'bert_mnli':
            USERHOME = os.path.expanduser(".")
            MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, 'bert_mnli'))
            MNLI_BERT = 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip'
            model_dir = os.path.join(MOVERSCORE_DIR)
            def download_MNLI_BERT(url, filename):
                with open(filename, 'wb') as f:
                    response = requests.get(url, stream=True)
                    total = response.headers.get('content-length')

                    if total is None:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        total = int(total)
                        for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total)
                            sys.stdout.write('\r[{}{}]'.format('-' * done, '.' * (50 - done)))
                            sys.stdout.flush()
                sys.stdout.write('\n')

            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            tarball = os.path.join(model_dir, os.path.basename(MNLI_BERT))
            rawdir = os.path.join(model_dir, 'raw')

            if not os.path.exists(tarball):
                print("Downloading %s to %s" % (MNLI_BERT, tarball))
                download_MNLI_BERT(MNLI_BERT, tarball)
                if tarball.endswith('.zip'):
                    z = zipfile.ZipFile(tarball, 'r')
                    #        z.printdir()
                    z.extractall(model_dir)
                    z.close()
                os.rename(os.path.join(model_dir, 'config.json'), os.path.join(model_dir, 'bert_config.json'))

        else:
            if os.path.exists(model):
                model_dir = model
            else:
                model_dir = model

        class BertForSequenceClassification(BertPreTrainedModel):
            def __init__(self, config, num_labels):
                super(BertForSequenceClassification, self).__init__(config)
                self.num_labels = num_labels
                self.bert = BertModel(config)
                self.dropout = nn.Dropout(config.hidden_dropout_prob)
                self.classifier = nn.Linear(config.hidden_size, num_labels)
                self.apply(self.init_bert_weights)

            def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None):
                encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                          output_all_encoded_layers=True)

                return encoded_layers, pooled_output

        self.batch_size = batch_size
        self.remove_punctuation = remove_punctuation
        self.wordpiece = wordpiece
        self.nthread = nthread
        self.n_gram = n_gram

        self._tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True, cache_dir='.cache')
        self._tokenizer.max_len = 512
        self._tokenizer.return_offsets_mapping=True

        self._model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3, cache_dir='.cache')
        self._model.eval()
        self._model.to(self.device)
        self.idf = idf
        self._stopwords = stopwords

    @property
    def hash(self):
        return '{}_{}-gram_idf({})'.format(self.model_name, self.n_gram, self.idf)

    def score(self, refs, hyps, ref_idf_sents=None, hyp_idf_sents=None):
        if self.idf:
            ref_idf_dict = get_idf_dict(ref_idf_sents, self._tokenizer, self.nthread)
            hyp_idf_dict = get_idf_dict(hyp_idf_sents, self._tokenizer, self.nthread)
        else:
            ref_idf_dict = defaultdict(lambda : 1.0)
            hyp_idf_dict = defaultdict(lambda : 1.0)
        preds = []
        for batch_start in range(0, len(refs), self.batch_size):
            batch_refs = refs[batch_start:batch_start + self.batch_size]
            batch_hyps = hyps[batch_start:batch_start + self.batch_size]

            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, self._model,
                                                                                         self._tokenizer, ref_idf_dict,
                                                                                         device=self.device)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, self._model,
                                                                                         self._tokenizer, hyp_idf_dict,
                                                                  device=self.device)

            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)

            ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

            for i in range(len(ref_tokens)):
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                           w not in set(string.punctuation) and '##' not in w and w not in self._stopwords]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                           w not in set(string.punctuation) and '##' not in w and w not in self._stopwords]

                ref_embedding_i, ref_idf_i = load_ngram(ref_ids, ref_embedding[i], ref_idf[i], self.n_gram, 1,
                                                        device=self.device)
                hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], self.n_gram, 1,
                                                        device=self.device)
                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

                distance_matrix = pairwise_distances(raw, raw)

                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

                c1[:len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i):] = hyp_idf_i

                c1 = _safe_divide(c1, np.sum(c1))
                c2 = _safe_divide(c2, np.sum(c2))
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
        return preds


#####################---utils---#####################
def truncate(tokens,tokenizer):
    if len(tokens) > tokenizer.max_len - 2:
        tokens = tokens[0:(tokenizer.max_len - 2)]
    return tokens

def process(a,tokenizer):
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a),tokenizer)+["[SEP]"]

    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True)
    return x_encoded_layers

def collate_idf(arr, tokenize, tokenizer, numericalize, idf_dict,
                pad="[PAD]", device='cpu'):
    tokens = [["[CLS]"]+truncate(tokenize(a), tokenizer)+["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cpu'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)


def load_ngram(ids, embedding, idf, n, o, device):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf
