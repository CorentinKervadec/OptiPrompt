import os
from fc1_utils import import_fc1, filter_templates
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

model_name = 'opt-350m'
files = [f'fc1_att_data_{model_name}_t0_autoprompt-no-filter_fullvoc.pickle', 
         f'fc1_att_data_{model_name}_t0_optiprompt_fullvoc_fixetok.pickle',
         f'fc1_att_data_{model_name}_t0_rephrase_fullvoc.pickle']
datapath = '../../data/fc1_data'

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m', use_fast=False)


# import the data from the pickle files
mode = 'minimal'
data = import_fc1(datapath, files, mode=[mode])

# if you want to filter the templates
data = filter_templates(data, min_template_accuracy=10, only_best_template=False)[mode]
templates = data[['type', 'template']].drop_duplicates()

# load token stats
tokens_stats = pd.read_csv('../../unit_token_wiki_100000/unit-token-wiki-s100000-concat69_2-input-token-stats.tsv', sep='\t')
excitant_sorted = tokens_stats[['token','excitant_ratio']].sort_values('excitant_ratio', ascending=False)
excitant_sorted['decoded'] = excitant_sorted['token'].apply(lambda x: tokenizer.decode(x).replace('/n', '__MEWLINE__'))
excitant_sorted[['decoded', 'excitant_ratio']].to_csv('../../unit_token_wiki_100000/unit-token-wiki-s100000-concat69_input-excitant.tsv', sep='\t', index=False)

tokens_stats = tokens_stats.to_dict()
# 

for prompt_type in templates['type'].unique():
    print(f'[{prompt_type}]')
    this_templates = templates[templates['type']==prompt_type]['template'].to_list()
    avg_freq = []
    avg_max_act = []
    avg_excitant_ratio = []
    for tmplt in this_templates:
        tmplt_tkn = tokenizer.encode(tmplt.replace('[X]','').replace('[Y]',''))
        avg_freq.append(np.array([tokens_stats['freq'][t] for t in tmplt_tkn]).mean())
        avg_max_act.append(np.array([tokens_stats['max_act'][t] for t in tmplt_tkn]).mean())
        avg_excitant_ratio.append(np.array([tokens_stats['excitant_ratio'][t] for t in tmplt_tkn]).mean())
    avg_freq = np.mean(avg_freq)
    avg_max_act = np.mean(avg_max_act)
    avg_excitant_ratio = np.mean(avg_excitant_ratio)
    print(f'avg_freq: {avg_freq}')
    print(f'avg_max_act: {avg_max_act}')
    print(f'avg_excitant_ratio: {avg_excitant_ratio}')

    