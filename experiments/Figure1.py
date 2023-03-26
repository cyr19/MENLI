from combine_1 import *

setup = 'ref'
nli = nli_dict[setup]

if setup == 'ref':
    metrics = ['COMET_wmt20-comet-da', 'BLEURT_BLEURT-20', 'MoverScore_bert_mnli_1-gram_idf(True)',
               'BERTScore_roberta-large_L17_idf_version=0.3.11(hug_trans=4.17.0)', 'BARTScore_bart-large-cnn',
               'BARTScore_bart-large-cnn+para_bi', 'SentSim_new_BERTScore_ref', 'SentSim_new_WMD_ref']
else:
    metrics = metrics_dict[setup]

ms = metrics + ['NLI-R', 'NLI-D'] if setup == 'ref' else metrics + ['XNLI-R', 'XNLI-D']
datasets = datasets_dict[setup]['adv']
all = np.zeros((len(ms), 13))
errors = ['add', 'drop', 'name', 'neg', 'num', 'pron', 'word_JJ', 'word_NN', 'word_VB', 'typo', 'jumble',
          'subject_verb_dis']
for d in datasets:
    r = []
    nli1_data = combine_nli(nli_metric=nli[0], dataset=d, adv=True)
    nli2_data = combine_nli(nli_metric=nli[1], dataset=d, adv=True)
    for m in metrics:
        if m == 'COMET_wmt20-comet-da' and d not in ['wmt20_google-de', 'xpaws-de']:
            r.append([0] * 13)
            continue
        metric_data = combine_nli_and_metric(nli_metric=nli[0], metric=m, dataset=d, adv=True, direction='avg',
                                             strategy='e')
        acc_dict = evaluate(metric_data, dataset=d, adv=True)[0.0]  # original metrics
        acc_list = [acc_dict[error] for error in errors]
        acc_list.append(np.mean(acc_list))
        r.append([a * 100 for a in acc_list])
    acc_dict = evaluate(nli1_data, dataset=d, adv=True)['avg']['e']
    acc_list = [acc_dict[error] for error in errors]
    acc_list.append(np.mean(acc_list))
    r.append([a * 100 for a in acc_list])

    acc_dict = evaluate(nli2_data, dataset=d, adv=True)['avg']['e']
    acc_list = [acc_dict[error] for error in errors]
    acc_list.append(np.mean(acc_list))
    r.append([a * 100 for a in acc_list])
    all = np.add(all, np.array(r))

if setup == 'ref':
    all[0, :] = [i * 2 for i in all[0, :]]
all = all / len(datasets)
s = ''
for i in range(len(all)):
    s += ','.join([str(v) for v in all[i]]) + '\n'

plot_phenomena_matrix(all, ms, 'all', setup)



