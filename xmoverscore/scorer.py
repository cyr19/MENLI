from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertModel, BertTokenizer, BertConfig
from .score_utils import word_mover_score, lm_perplexity

class XMOVERScorer:

    def __init__(
        self,
        model_name='multilingual',
        lm_name=None,
        do_lower_case=False,      
        device='cuda:0'
    ):        
        
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case, model_max_length=512, truncation=True)
        print(self.tokenizer.model_max_length)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(device)        

        if lm_name == 'gpt2':
            self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
            self.lm.to(device)
        self.device = device


    #(mapping, projection, bias, model, tokenizer, src, hyps, n_gram=1, batch_size=256, device='cuda:0')
    def compute_xmoverscore(self, mapping, projection, bias, source, translations, ngram=1, bs=32, layer=8, dropout_rate=0.3):
        return word_mover_score(mapping, projection, bias, self.model, self.tokenizer, source, translations, \
                                n_gram=ngram,  batch_size=bs, device=self.device)

    def compute_perplexity(self, translations, bs):        
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs)            
