import pickle
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

PICKLE_NAME = './unit-token-wiki-s100000-concat69.pickle'
MAX_TOKENS = 500
OUTPUT_PATH = './unit-token-wiki-s100000-concat69_2'

parser = argparse.ArgumentParser(description='OPTCorpus generation')

# Data selection
parser.add_argument('--decode_tokens', action='store_true')

args = parser.parse_args()

def load_pickles(f):
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    return data

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m', use_fast=False)

print('Loading data...')
data = load_pickles(PICKLE_NAME)

diversity = {'input':[], 'output':[]}

# extract the top token for each unit
for m in ['input', 'output']:
    with open(OUTPUT_PATH+f'-{m}.tsv', 'w') as f:
        f.write('\t'.join(['Layer', 'Unit', 'avg_act_uni', 'avg_act_true','energy', 'q99', 'Tokens'])+'\n')
        for l, data_layer in enumerate(data['unit_tokens'][m]):
            # transpose to get [n-units, n-tokens]
            data_layer = data_layer.t()
            # sort tokens
            sorted, indices = torch.sort(data_layer, descending=True, dim=1)
            # compute the 99th quantile
            q99 = torch.quantile(sorted.float(), q=0.99, dim=-1)
            nonzero_mask = (sorted != 0.0) # also discard  units-tokens with activation 0
            q99_mask = (sorted >= q99.unsqueeze(-1)) * nonzero_mask
            # unit energy
            energy = data_layer.sum(-1)
            # compute token diversity in that layer
            units_token_distribution = data_layer / (energy.unsqueeze(-1)+1e-9) # normalised by the energy = how is distributed the energy across tokens.
            aggreg_mask = torch.sum(q99_mask, dim=0) / torch.sum(q99_mask)
            ent_aggreg = torch.sum(-aggreg_mask*torch.log(aggreg_mask+1e-12))
            diversity[m].append(ent_aggreg)
            # average activation
            avg_act_uniform_dist = data_layer.mean(-1)
            avg_act_true_dist = (data_layer.float() * data['tokens-count'][m].unsqueeze(0)).mean(-1)
            # unit tokens association
            for unit_idx, unit_q99_mask in tqdm(enumerate(q99_mask), desc=f'[{m}] Layer {l}:'):
                unit_data = [avg_act_uniform_dist[unit_idx].item(),\
                             avg_act_true_dist[unit_idx].item(),\
                             energy[unit_idx].item(),\
                             q99[unit_idx].item()]
                # measure how much this unit is different from the others

                # extract the most associated tokens
                idx_q99_tokens = indices[unit_idx][unit_q99_mask]
                if args.decode_tokens:
                    q99_words = [tokenizer.decode(t) for t in idx_q99_tokens]
                else:
                    q99_words = [str(t) for t in idx_q99_tokens]
                # limit the number of tokens per unit
                q99_words = q99_words[:min(MAX_TOKENS, len(q99_words))]
                # write into the file
                line = '\t'.join([f'Layer {l}', f'Unit {unit_idx}']\
                                  + [str(i) for i in unit_data]\
                                  + q99_words) + '\n'
                f.write(line)

# todo
# compute the frequency / max activation ratio
# in order to find the tokens which are rare but very excitant
# then compute the average ratio on autoprompt vs para templates.
# something like r = max(act) / sqrt(count)

# also count the average token frequency on auto vs. para templates

# compute the average activation for each unit
# 1) Normalized by the total number of tokens.
# (under wikipedia token distribution)
# (uni_tokens * tokens_count) / tokens_count.sum()
# 2) Normalized by each tokens count individually
# (under uniform token distribution)
# unit_tokens