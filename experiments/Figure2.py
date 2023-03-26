from combine_1 import *


def plot_MT_trade_off_lines(setup, level):
    datasets_adv = datasets_dict[setup]['adv']
    datasets_mt = datasets_dict[setup][f'mt-{level}']
    metrics = metrics_dict[setup]
    nlis = nli_dict[setup]

    pearson_dicts_all_dataset = defaultdict(lambda: defaultdict(list))
    for dataset in datasets_mt:
        nli1_data, nli2_data = defaultdict(dict), defaultdict(dict)
        for metric in metrics:
            if 'SentSim_new_WMD' in metric and level == 'sys':
                print('skip sentsim(wmd) for system-level evaluation.')
                continue
            pearson_dicts = defaultdict(list)
            acc_dicts = defaultdict(list)
            for nli in nlis:
                print(f"Evaluate {metric}+{nli} on {dataset} (MT)...")
                mt_data = combine_nli_and_metric(nli, metric, dataset, method=add_scores, adv=False)
                pearson_dict = evaluate(mt_data, dataset, adv=False)
                pearson_dicts[nli].append(pearson_dict)
                pearson_dicts_all_dataset[metric][nli].append(pearson_dict)
                for dt_adv in datasets_adv:
                    print(f"Evaluate {metric}+{nli} on {dt_adv} (adv.)...")
                    if 'COMET' in metric and setup == 'ref' and dt_adv not in ['wmt20_google-de', 'xpaws-de']:
                        print('skip for COMET: {}'.format(dt_adv))
                        continue
                    adv_data = combine_nli_and_metric(nli, metric, dt_adv, adv=True)
                    acc_dict = evaluate(metric_data=adv_data, dataset=dt_adv, adv=True)
                    acc_dicts[nli].append(acc_dict)

            assert len(acc_dicts[nlis[0]]) == len(datasets_adv) or (len(acc_dicts[nlis[0]]) == 2 and 'COMET' in metric and setup == 'ref')
            nli1_data['adv'][metric] = append_dict(acc_dicts[nlis[0]])  # return list of float
            nli2_data['adv'][metric] = append_dict(acc_dicts[nlis[1]])

    for metric in metrics:
        if 'SentSim_new_WMD' in metric and level == 'sys':
            print('skip sentsim(wmd) for system-level evaluation.')
            continue
        nli1_data['mt'][metric] = append_dict(pearson_dicts_all_dataset[metric][nlis[0]])
        nli2_data['mt'][metric] = append_dict(pearson_dicts_all_dataset[metric][nlis[1]])

    plot_trade_off(nli1_data, nli2_data, 'ALL', setup, level=level)


# ref seg
plot_MT_trade_off_lines('ref', 'seg')

# src seg
plot_MT_trade_off_lines('src', 'seg')
