import pandas as pd
import numpy as np
from metrics.scorer_utils import init_scorer, scoring
from collections import defaultdict
from scipy.stats import pearsonr, kendalltau, spearmanr
import os
import pickle
from mosestokenizer import *
import json
from datasets.REALSumm.scoring.utils import get_sents_from_tags

def store_scores(scores_dict, metric_name):
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_scores')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, metric_name + f"_{args.aggregate}{'_use_article' if args.use_article else ''}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(scores_dict, f)

class RealSumm:
    def __init__(self, data_path='datasets/REALSumm/scores_dicts/', scorer=None, args=None, load_doc=True):
        self.args = args
        self.scorer = scorer
        self.data_path = data_path
        self.load_doc = load_doc
        self.data = self.load_data_realsum()
        self.detokenizer = MosesDetokenizer('en')

    def evaluate(self):
        self.compute_metric_scores()
        self.correlate_with_human_scores()

    def load_data_realsum(self):
        data = defaultdict(list)
        if self.load_doc:
            with open(os.path.join(self.data_path, 'src.txt'), 'r') as f:
                all_source_docs = f.readlines()
            source_ids = [1017, 10586, 11343, 1521, 2736, 3789, 5025, 5272, 5576, 6564, 7174, 7770, 8334, 9325, 9781, 10231, 10595, 11351, 1573, 2748, 3906, 5075, 5334, 5626, 6714, 7397, 7823, 8565, 9393, 9825, 10325, 10680, 11355, 1890, 307, 4043, 5099, 5357, 5635, 6731, 7535, 7910, 8613, 9502, 10368, 10721, 1153, 19, 3152, 4303, 5231, 5420, 5912, 6774, 7547, 8001, 8815, 9555, 10537, 10824, 1173, 1944, 3172, 4315, 5243, 5476, 6048, 6784, 7584, 8054, 8997, 9590, 10542, 11049, 1273, 2065, 3583, 4637, 5244, 5524, 6094, 6976, 7626, 8306, 9086, 9605, 10563, 11264, 1492, 2292, 3621, 4725, 5257, 5558, 6329, 7058, 7670, 8312, 9221, 9709]
            source_docs = np.array(all_source_docs)[source_ids]
        for system_type in ['abs', 'ext']:
            with open(os.path.join(self.data_path, f'{system_type}.pkl'), 'rb') as f:
                obj = pickle.load(f)
            for doc_id, doc_data in obj.items():
                ref = doc_data['ref_summ']
                ref = ' '.join(get_sents_from_tags(ref, sent_start_tag='<t>', sent_end_tag='</t>'))
                system_data = doc_data['system_summaries']
                for system, system_summary in system_data.items():
                    if system == 'bart_out.txt' and system_type == 'ext':
                        system = 'bart_out_ext.txt'
                    hyp = ' '.join(get_sents_from_tags(system_summary['system_summary'], sent_start_tag='<t>', sent_end_tag='</t>')) #.replace('</t>', '').replace('<t>', '')
                    data['system_type'].append(system_type)
                    data['doc_id'].append(doc_id)
                    if self.load_doc:
                        data['doc'].append(source_docs[doc_id].replace('( cnn ) ', ''))
                    data['ref'].append(ref)
                    data['system'].append(system)
                    data['hyp'].append(hyp)
                    data['human_score'].append(system_summary['scores']['litepyramid_recall'])
        data_df = pd.DataFrame.from_dict(data).sort_values(['doc_id', 'system_type', 'system'])
        #data_df.to_csv('datasets/realsum_processed.csv', encoding='utf-8', index=False)
        return data_df

    def detokenize(self, text: str):
        words = text.split(" ")
        return self.detokenizer(words)

    def compute_metric_scores(self):
        results = defaultdict(list) if 'NLI' in self.args.metric and 'MENLI' not in args.metric else []
        refs = list(self.data['ref'])
        hyps = list(self.data['hyp'])
        srcs = list(self.data['doc']) if self.args.use_article else []
        # compute scores on the whole dataset
        scores = scoring(scorer=self.scorer, args=self.args, refs=refs, hyps=hyps, sources=srcs)
        assert len(scores) == len(self.data) or len(scores[0]) == len(self.data)
        if not isinstance(scores, tuple):
            results = scores
        else:
            assert len(scores[0]) == len(refs)
            results['e'] = scores[2]
            results['n'] = scores[1]
            results['c'] = scores[0]

        if 'NLI' not in self.args.metric or 'MENLI' in args.metric:
            self.data['metric_scores'] = results
        else:
            self.data['metric_scores_e'] = results['e']
            self.data['metric_scores_n'] = results['n']
            self.data['metric_scores_c'] = results['c']

        if not self.args.not_store_scores:
            cols = [col for col in self.data.columns if 'metric' in col]
            for col in cols:
                suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
                metric_name = self.args.metric + '_' + metric_hash + suffix
                scores = list(self.data[col])
                store_scores(scores, metric_name)
        return self.data

    def print_and_save(self, corr, metric_name, level, save=True):
        output_path = os.path.join(self.args.output_dir, f"{self.args.dataset}_results.csv")
        first_raw = "metric,model,dataset,setup,level,pearson\n"
        s = f"{metric_name},{self.args.model},{self.args.dataset},{'ref-free' if self.args.use_article else 'ref-based'},{level},{corr}\n"
        print(first_raw+s)
        mode = 'w' if not os.path.exists(output_path) else 'a'
        final_string = first_raw + s if not os.path.exists(output_path) else s
        if save:
            with open(output_path, mode) as f:
                f.write(final_string)

    def correlate_with_human_scores(self, save=True):
        cols = [col for col in self.data.columns if 'metric' in col]

        doc_ids = set(self.data['doc_id'])

        print('\ncomputing summary-level correlation...')
        for col in cols:
            corr_list = []
            suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
            metric_name = self.args.metric + '_' + metric_hash + suffix

            for doc_id in doc_ids:
                doc_data = self.data[self.data.doc_id == doc_id]
                metric_scores = doc_data[col]
                human_scores = doc_data['human_score']

                corr, p = pearsonr(metric_scores, human_scores)
                if p <= 0.05:
                    corr_list.append(corr)

            print(f"{(len(doc_ids) - len(corr_list))/float(len(doc_ids))} values ignored.")
            self.print_and_save(np.mean(corr_list), metric_name, 'summary', save=save)


        print('\ncomputing system-level correlation...')
        data = self.data.groupby('system').mean()
        for col in cols:
            suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
            metric_name = self.args.metric + '_' + metric_hash + suffix
            metric_scores = data[col]
            human_scores = data['human_score']
            corr = pearsonr(human_scores, metric_scores)[0]
            self.print_and_save(corr, metric_name, 'system', save=save)

class SummEval:
    def __init__(self, data_path='datasets/model_annotations.aligned.scored.jsonl', scorer=None, args=None, load_doc=True):
        self.data_path = data_path
        self.args = args
        self.load_doc = load_doc
        self.data = self.load_data_summ()
        self.scorer = scorer
        if args.aggregate:
            self.aggregate = np.max if args.aggregate == 'max' else np.mean
        else:
            self.aggregate = args.aggregate

    def evaluate(self):
        if not self.aggregate and not args.use_article:
            raise ValueError('Aggregation method not specified.')
        self.compute_metric_scores(aggregate=self.aggregate)
        self.correlate_with_human_scores()

    def find_source_document(self, doc_id):
        try:
            if 'cnn-test' in doc_id:
                data_dir = os.path.join(self.args.data_dir, 'cnndm/cnn/cnn/stories')
            else:
                data_dir = os.path.join(self.args.data_dir, 'cnndm/dailymail/dailymail/stories')
        except:
            raise FileNotFoundError('You need to manually download cnndm datasets from https://cs.nyu.edu/~kcho/DMQA/.')
        path = os.path.join(data_dir, doc_id.split('-')[-1]+'.story')
        with open(path, 'r') as f:
            doc = f.read()
        doc = doc.split('@highlight')[0].replace('\n', ' ').replace('(CNN)', '')
        return doc

    def load_data_summ(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            json_strings = f.readlines()

        lines = [json.loads(l) for l in json_strings]
        data = defaultdict(list)
        for l in lines:
            data['system'].append(l['model_id'])
            data['id'].append(l['id'])
            if self.load_doc:
                doc = self.find_source_document(l['id'])
                data['doc'].append(doc)
            data['hyp'].append(l['decoded'])
            data['refs'].append(l['references'])

            for anno in ['expert', 'turker']:
                if anno == 'turker':
                    continue
                for criterion in ['coherence', 'consistency', 'fluency', 'relevance']:
                    data['{}_{}'.format(anno, criterion)].append(np.mean([i[criterion] for i in l[anno+"_annotations"]]))

        data = pd.DataFrame.from_dict(data).sort_values(by=['system', 'id'])
        #data.to_csv('datasets/evalsumm_processed.csv', index=False)
        return data

    def compute_metric_scores(self, aggregate=np.mean):
        results = defaultdict(list) if 'NLI' in self.args.metric and 'MENLI' not in args.metric else []

        hyps, refs = [], []
        for h, rs in zip(self.data['hyp'], self.data['refs']):
            assert len(rs) == 11
            hyps += [h] * len(rs)
            refs += rs

        assert len(refs) == len(hyps)

        # compute scores on the whole dataset
        if self.args.use_article:
            scores = scoring(scorer=self.scorer, args=self.args, refs=[], hyps=list(self.data['hyp']), sources=list(self.data['doc']))
            if not isinstance(scores, tuple):
                assert len(scores) == len(self.data)
                results = scores
            else:
                assert len(scores[0]) == len(self.data)
                results['c'], results['n'], results['e'] = scores
        else:
            scores = scoring(scorer=self.scorer, args=self.args, refs=refs, hyps=hyps, sources=[])
            if not isinstance(scores, tuple):
                assert len(scores) == len(refs)
                results = [aggregate(scores[i * 11: i * 11 + 11]) for i in range(int(len(scores) / 11))]
            else:
                assert len(scores[0]) == len(refs)
                results['e'] = [aggregate(scores[2][i * 11: i * 11 + 11]) for i in range(int(len(scores[0]) / 11))]
                results['n'] = [aggregate(scores[1][i * 11: i * 11 + 11]) for i in range(int(len(scores[0]) / 11))]
                results['c'] = [aggregate(scores[0][i * 11: i * 11 + 11]) for i in range(int(len(scores[0]) / 11))]

        if 'NLI' not in self.args.metric or 'MENLI' in args.metric:
            self.data['metric_scores'] = results
        else:
            self.data['metric_scores_e'] = results['e']
            self.data['metric_scores_n'] = results['n']
            self.data['metric_scores_c'] = results['c']

        if not self.args.not_store_scores:
            cols = [col for col in self.data.columns if 'metric' in col]
            for col in cols:
                suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
                metric_name = self.args.metric + '_' + metric_hash + suffix
                scores = list(self.data[col])
                store_scores(scores, metric_name)
        print(self.data)
        return self.data

    def print_and_save(self, corr_dict, metric_name):
        output_path = os.path.join(self.args.output_dir, f"{self.args.dataset}_results.csv")
        first_raw = "metric,model,dataset,setup,aggregation,correlation,annotator,coherence,consistency,fluency,relevance,average\n"
        for anno in ['expert', 'turker']:
            if anno == 'turker':
                continue
            s = ''
            for cor in corr_dict[anno].keys():
                coh = corr_dict[anno][cor]['coherence']
                con = corr_dict[anno][cor]['consistency']
                flu = corr_dict[anno][cor]['fluency']
                rel = corr_dict[anno][cor]['relevance']
                avg = np.mean([coh, con, flu, rel])
                s += f"{metric_name},{self.args.model},{self.args.dataset},{'ref-free' if self.args.use_article else 'ref-based'}," \
                     f"{self.args.aggregate},{cor},{anno},{coh},{con},{flu},{rel},{avg}\n"
            print(first_raw + s)
            mode = 'w' if not os.path.exists(output_path) else 'a'
            final_string = first_raw + s if not os.path.exists(output_path) else s
            with open(output_path, mode) as f:
                f.write(final_string)

    def correlate_with_human_scores(self):
        data = self.data.groupby('system').mean()
        cols = [col for col in data.columns if 'metric' in col]
        for col in cols:
            suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
            metric_name = self.args.metric + '_' + metric_hash + suffix
            corr_dict = defaultdict(lambda: defaultdict(lambda : defaultdict(float)))
            metric_scores = list(data[col])
            for anno in ['expert', 'turker']:
                if anno == 'turker':
                    continue
                for c in ['coherence', 'consistency', 'fluency', 'relevance']:
                    human_scores = list(data[anno + '_' + c])
                    corr_dict[anno]['pearson'][c] = pearsonr(human_scores, metric_scores)[0]
                    corr_dict[anno]['kendall'][c] = kendalltau(human_scores, metric_scores)[0]
            self.print_and_save(corr_dict, metric_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # all
    parser.add_argument('--metric', type=str, default='None')
    parser.add_argument('--model', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--dataset', type=str, default='summ')
    parser.add_argument('--output_dir', type=str, default='../results/')
    parser.add_argument('--aggregate', type=str)
    parser.add_argument('--not_store_scores', action='store_false')
    parser.add_argument('--use_article', action='store_true')

    # BARTScore
    parser.add_argument('--bidirection', action='store_true')

    # NLI1Score, NLI2Score,
    parser.add_argument('--checkpoint', type=int, default=0)

    # MENLI
    parser.add_argument('--nli_weight', type=float, default=1.0)
    parser.add_argument('--combine_with', type=str, default='None')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    scorer, metric_hash = init_scorer(args)

    if args.dataset == 'summ':
        eval = SummEval(scorer=scorer, args=args, load_doc=args.use_article)
    elif args.dataset == 'realsumm':
        eval = RealSumm(scorer=scorer, args=args, load_doc=args.use_article)
    else:
        raise ValueError('No such dataset.')
    eval.evaluate()



