from combine_1 import *
plt.style.use('seaborn-darkgrid')
seed = 0


#data = prepare_improvements_df_mt()
# run above function to get the following .csv file
data = pd.read_csv('../results/tables/all_improvement_data_mt.csv')

data['type'] = [f"Adv.({t.split('-')[-1]})" if 'adv' in t else 'Standard' for t in data['type']]
data['type'] = ['Adv.(adequacy)' if t == 'Adv.(fact)' else t for t in data['type']]
#data['type']
plt.figure(figsize=(3, 6))
sns.lineplot(data=data, x='nli_weight', y='improvement', hue='type', estimator='mean', seed=seed)
plt.legend(loc='upper left')
plt.ylabel('improvement(%)')
plt.xlim(0.1,0.9)
plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.tight_layout()
plt.savefig(f'../results/plots/figure3_mt.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

'''
combined_sum(use_article=True, error_type='all')
combined_sum(use_article=True, error_type='factuality')
combined_sum(use_article=False, error_type='all')
combined_sum(use_article=False, error_type='factuality')
data_ref = prepare_improvement_df_sum('ref')
data_src = prepare_improvement_df_sum('src')
data = pd.concat([data_src, data_ref], ignore_index=True)
'''
# run above to get the following .csv file
data = pd.read_csv('../results/tables/all_improvement_data_sum.csv')

data['type'] = [f"Adv.({t.split('-')[-1]})" if 'adv' in t else 'Standard' for t in data['type']]
data['type'] = ['Adv.(adequacy)' if t == 'Adv.(fact)' else t for t in data['type']]
plt.figure(figsize=(3, 6))
sns.lineplot(data=data, x='nli_weight', y='improvement', hue='type', estimator=np.median, seed=seed)
plt.legend(loc='upper left')
plt.ylabel('improvement(%)')
plt.xlim(0.1, 0.9)
plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.savefig(f'../results/plots/figure3_sum.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

