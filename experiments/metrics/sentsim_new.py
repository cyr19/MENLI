from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import product
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineSimilarity
from torch.cuda import is_available as cuda_is_available
from datasets import load_metric
#from utils.knn import ratio_margin_align
#from utils.env import DATADIR
#from common import CommonScore
import torch
import pulp
#import logging
import numpy as np

# This code is based on https://github.com/Rain9876/Unsupervised-crosslingual-Compound-Method-For-MT
#class SentSim(CommonScore):
class SentSim:
    """
    A wrapper around the original SentSim implementation. Be careful, the used
    models were fine-tuned on parallel sentences.
    """
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_wmd=False,
        cross_lingual=False
    ):

        wordemb_model = 'xlm-roberta-base' if cross_lingual else 'roberta-large'
        sentemb_model = 'xlm-r-bert-base-nli-stsb-mean-tokens' if cross_lingual else 'roberta-large-nli-stsb-mean-tokens'
        if use_wmd:
            self.tokenizer, self.word_model = self.get_WMD_Model(wordemb_model)
            self.word_model.to(device)
            self.layers = self.layer_processing(self.word_model)
        else:
            self.word_model = wordemb_model
        self.use_wmd = use_wmd
        self.sent_model = SentenceTransformer(sentemb_model, device=device)
        self.device = device
        self.cross_lingual = cross_lingual

    @property
    def hash(self):
        return "{}_{}".format('WMD' if self.use_wmd else "BERTScore", 'src' if self.cross_lingual else 'ref')

    def _embed(self, source_sents, target_sents):
        return (
            self.sent_model.encode(source_sents, convert_to_tensor=True).cpu(),
            self.sent_model.encode(target_sents, convert_to_tensor=True).cpu())
    '''
    def align(self, source_sents, target_sents):
        logging.warn("For now SentSim sentence alignment only leverages sentence embeddings.")
        source_embeddings, target_embeddings = self._embed(source_sents, target_sents)
        indeces, scores = ratio_margin_align(source_embeddings, target_embeddings, self.k,
                self.knn_batch_size, self.device)

        sent_pairs = [(source_sents[src_idx], target_sents[tgt_idx]) for src_idx, tgt_idx in indeces]
        return sent_pairs, scores
    '''

    def score(self, source_sents, target_sents):
        cosine = self.getSentSimilarity(target_sents, source_sents)
        if self.use_wmd:
            wmd = self.compute_WMD(target_sents, source_sents, self.tokenizer, self.word_model)
            return self.combine_metrics(cosine, wmd, corr=[1, -1])
        else:
            bertscore = self.getBertScore(target_sents, source_sents, self.word_model)
            return self.combine_metrics(cosine, bertscore, corr=[1, 1])

    def combine_metrics(_, *args, **kwargs):
        assert len(args) == len(kwargs["corr"]) and len(args[0]) == len(args[1])
        output = []

        for i in range(len(args[0])):
            value = 0
            for sign, metric in zip(kwargs["corr"], args):
                assert metric[i] <= 1 and metric[i] >= 0
                if sign > 0:
                    value += np.exp(metric[i])
                else:
                    value += np.exp(1-metric[i])
            output.append(value)

        return output

    def getSentSimilarity(self, sents1, sents2):
        embed_sent1, embed_sent2 = self._embed(sents1, sents2)
        cos_sim = CosineSimilarity(dim=1)(embed_sent1,embed_sent2)
        # Normalized
        cos_sim = (cos_sim -torch.min(cos_sim))/ (torch.max(cos_sim)-torch.min(cos_sim))
        return cos_sim.numpy()

    def getBertScore(self, sents1, sents2, model):
        bert_score_metric = load_metric('bertscore', keep_in_memory=True, cache_dir='.cache')
        bert_score_metric.add_batch(predictions=sents2, references=sents1)
        score = torch.tensor(bert_score_metric.compute(model_type=model, device=self.device)["f1"])
        # Normalized Bert Score F1
        norm_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        return norm_score.tolist()

    def compute_WMD(self, hypotheses, references, tokenizer, model, embed_type=False):
        wmd = []

        for reference, hypothesis in zip(references, hypotheses):
            wmd_tmp = self.word_mover_distance(reference, hypothesis, tokenizer, model, embed_type)
            wmd.append(wmd_tmp)
        # Normalize
        wmd = [(val-min(wmd))/(max(wmd)-min(wmd)) for val in wmd]
        return np.array(wmd)

    def word_mover_distance(self, sent1, sent2, tokenizer, model, embed_type, lpFile=None):
        sent1_buckets, sent2_buckets, sent1_embedding, sent2_embedding = self.embedding_processing(sent1, sent2,
                tokenizer, model, embed_type)
        prob = self.word_mover_distance_probspec(sent1_buckets, sent2_buckets, sent1_embedding, sent2_embedding, lpFile=lpFile)
        return pulp.value(prob.objective)

    def word_mover_distance_probspec(_, sent1_buckets, sent2_buckets, sent1_embedding, sent2_embedding, lpFile=None):
        first_sent_buckets = {f"x{idx}": item[1] for idx, item in enumerate(sent1_buckets.items())}
        second_sent_buckets = {f"y{idx}": item[1] for idx, item in enumerate(sent2_buckets.items())}

        var_names = list(first_sent_buckets.keys()) + list(second_sent_buckets.keys())
        all_embedding = torch.cat([sent1_embedding, sent2_embedding])
        wordvecs = {token: embedding.detach().cpu().numpy() for token, embedding in zip(var_names, all_embedding)}
        assert len(var_names) == all_embedding.size(0)

        T = pulp.LpVariable.dicts('T_matrix', list(product(var_names, var_names)), lowBound=0)
        prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
        prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                            for token1, token2 in product(var_names, var_names)])
        for token2 in second_sent_buckets:   #constrains
            prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
        for token1 in first_sent_buckets:    #constrains
            prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets])==first_sent_buckets[token1]
        if lpFile!=None:
            prob.writeLP(lpFile)

        prob.solve(pulp.apis.PULP_CBC_CMD(msg=0))
        return prob

    def tokenize_and_truncate(self, a):
        tokens = self.tokenizer.tokenize(a)
        if len(tokens) + 2 > self.tokenizer.model_max_length:
            tokens = tokens[: self.tokenizer.model_max_length - 2]
        assert len(tokens) + 2 <= self.tokenizer.model_max_length
        return tokens

    def embedding_processing(self, sent1, sent2, tokenizer, model, embed_type=False):
        #sent1_tokens, sent2_tokens  = tokenizer.tokenize(sent1), tokenizer.tokenize(sent2)
        sent1_tokens, sent2_tokens = self.tokenize_and_truncate(sent1), self.tokenize_and_truncate(sent2)

        if embed_type:
            sent1_buckets, sent2_buckets  = self.tokens_to_fracdict(sent1_tokens), self.tokens_to_fracdict(sent2_tokens)
            sent1_embedding = model.embeddings.word_embeddings(
                    torch.tensor(tokenizer.convert_tokens_to_ids(list(sent1_buckets.keys()))))
            sent2_embedding = model.embeddings.word_embeddings(
                    torch.tensor(tokenizer.convert_tokens_to_ids(list(sent2_buckets.keys()))))
        else:
            sent1_buckets = self.tokens_to_fracdict_contextual(sent1_tokens)
            sent2_buckets = self.tokens_to_fracdict_contextual(sent2_tokens)
            sent1_id = tokenizer(sent1,return_tensors="pt")
            sent2_id = tokenizer(sent2,return_tensors="pt")

            sent1_id['input_ids'] = sent1_id['input_ids'].to(self.device)
            sent2_id['input_ids'] = sent2_id['input_ids'].to(self.device)

    #         [-8:-7] indicates Roberta-Large layer 17
    #         [-4,-3] indicates XLM Roberta-Base layer 9
            if self.cross_lingual:
                model(sent1_id['input_ids'])
                sent1_embedding = torch.mean(torch.stack(self.layers[-4:-3]).squeeze(1).permute(1,0,2), dim=1)
                model(sent2_id['input_ids'])
                sent2_embedding = torch.mean(torch.stack(self.layers[-4:-3]).squeeze(1).permute(1,0,2), dim=1)
            else:
                model(sent1_id['input_ids'])
                sent1_embedding = torch.mean(torch.stack(self.layers[-8:-7]).squeeze(1).permute(1, 0, 2), dim=1)
                model(sent2_id['input_ids'])
                sent2_embedding = torch.mean(torch.stack(self.layers[-8:-7]).squeeze(1).permute(1, 0, 2), dim=1)
        self.layers.clear()

        if sent1_embedding.size()[0] - 2 == len(sent1_tokens):
            sent1_embedding = sent1_embedding[1:-1,:] # Remove bos and eos tokens
        if sent2_embedding.size()[0] - 2 == len(sent2_tokens):
            sent2_embedding = sent2_embedding[1:-1,:] # Remove bos and eos tokens

        assert len(sent1_buckets) + len(sent2_buckets) == (sent1_embedding.size()[0] + sent2_embedding.size()[0])
        return sent1_buckets, sent2_buckets, sent1_embedding, sent2_embedding

    def tokens_to_fracdict_contextual(_, tokens):
        return {token: 1/len(tokens) for token in range(len(tokens))}

    def tokens_to_fracdict(_, tokens):
        cntdict = defaultdict(lambda : 0)

        for token in tokens:
            cntdict[token] += 1
        totalcnt = sum(cntdict.values())
        return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}

    def get_WMD_Model(_, name):
        max_length = 512 if 'base' in name else 1024
        tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=max_length)
        print(tokenizer.model_max_length)
        model = AutoModel.from_pretrained(name, return_dict=True)
        # bert_model.embeddings.word_embeddings
        model.eval()
        return tokenizer, model

    def layer_processing(_, model):
        layers = []

        for i in model.encoder.layer:
            i.register_forward_hook(lambda *args: layers.append(args[2][0]))

        return layers
