import glob
import pandas as pd


for dataset in ['paws_back_google', 'paws_ori_google', 'wmt20_google-de', 'xpaws-de', 'xpaws-fr', 'xpaws-zh', 'xpaws-ja', 'summ_google']:
    files = glob.glob(f'{dataset}*')
    for f in files:
        df = pd.read_csv(f)
        #print(df)
        # check if the metrics are consistent
        assert len(set(df['metric'])) == 1
        # check if all phenomena are included
        try:
            assert len(set(df['error'])) == 12
        except:
            print(df)
        assert set(df['error']) == set(['add', 'drop', 'name', 'neg', 'num', 'pron', 'word_JJ', 'word_NN', 'word_VB', 'typo', 'jumble', 'subject_verb_dis'])
    # todo: check if all metrics are included
    if dataset in ['paws_back_google', 'paws_ori_google']:
        print(files)
        assert len(files) == 13
    elif dataset in ['wmt20_google-de', 'xpaws-de']:
        print(files)
        assert len(files) == 23
    elif dataset in ['xpaws-fr', 'xpaws-ja', 'xpaws-zh']:
        print(files)
        assert len(files) == 9
    elif dataset in ['summ_google']:
        print(files)
        print(len(files))
        assert len(files) == 17
