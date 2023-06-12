import pickle
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

PICKLE_NAME = '../../unit_token_wiki_100000/unit-token-wiki-s100000-concat69.pickle'
MAX_TOKENS = 500
OUTPUT_PATH = '../../unit_token_wiki_100000/unit-token-wiki-s100000-concat69_idonly'
REPLACE = {'\n':'__NEWLINE__',
           '\t':'__TAB__',
           '\s':'__SPACE__',
           ' ':'__SPACE__',
           '    ':'__TAB__'}
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


# for m in ['input', 'output']:
#     # tokens stats
#     with open(OUTPUT_PATH+f'-{m}-token-stats.tsv', 'w') as f:
#         f.write('\t'.join(['token', 'count', 'freq', 'q99_act', 'max_act', 'excitant_ratio']) + '\n')
#         per_token_max_act = torch.max(torch.concat(data['unit_tokens'][m], dim=-1), dim=-1).values
#         per_token_q99_act = torch.quantile(torch.concat(data['unit_tokens'][m], dim=-1).float(), q=0.99, dim=-1)
#         excitant_ratio = (torch.log(per_token_max_act) / torch.log(data['tokens-count'][m]))
#         excitant_ratio = torch.where(data['tokens-count'][m]==0, torch.zeros_like(excitant_ratio), excitant_ratio)
#         freq = data['tokens-count'][m] / data['tokens-count'][m].sum() 
#         for i, cpt in enumerate(tqdm(data['tokens-count'][m], desc=f'Tokens stats {m}')):
#             if args.decode_tokens:
#                 decoded = tokenizer.decode(i)
#                 for t,r in REPLACE.items():
#                     decoded = decoded.replace(t, r)
#             else:
#                 decoded = str(i)
#             line = '\t'.join([
#                 decoded,\
#                 str(cpt.item()),\
#                 str(freq[i].item()),\
#                 str(per_token_q99_act[i].item()),\
#                 str(per_token_max_act[i].item()),\
#                 str(excitant_ratio[i].item()),\
#                     ]) + '\n'
#             f.write(line)

for m in ['input', 'output']:
    # unit stats
    with open(OUTPUT_PATH+f'-{m}.tsv', 'w') as f:
        # f.write('\t'.join(['Layer', 'Unit', 'avg_act_uni', 'avg_act_true','energy', 'q99', 'Entropy', 'Uniqueness', 'Tokens'])+'\n')
        f.write('\t'.join(['Layer', 'Unit', 'q99', 'Tokens'])+'\n')
        for l, data_layer in enumerate(data['unit_tokens'][m]):
            # transpose to get [n-units, n-tokens]
            data_layer = data_layer.t().float()
            # sort tokens
            sorted, indices = torch.sort(data_layer, descending=True, dim=1)
            # compute the 99th quantile
            q99 = torch.quantile(sorted.float(), q=0.99, dim=-1)
            nonzero_mask = (sorted != 0.0) # also discard  units-tokens with activation 0
            q99_mask = (sorted >= q99.unsqueeze(-1)) * nonzero_mask
            # # unit energy
            # energy = data_layer.sum(-1)
            # # compute token diversity in that layer
            # units_token_distribution = data_layer / (energy.unsqueeze(-1)+1e-9) # normalised by the energy = how is distributed the energy across tokens.
            # units_token_entropy = torch.sum(-units_token_distribution*torch.log(units_token_distribution+1e-12), dim=-1)
            # # aggreg_mask = torch.sum(q99_mask, dim=0) / torch.sum(q99_mask)
            # # ent_aggreg = torch.sum(-aggreg_mask*torch.log(aggreg_mask+1e-12))
            # # diversity[m].append(ent_aggreg)
            # # average activation
            # avg_act_uniform_dist = data_layer.mean(-1)
            # avg_act_true_dist = (data_layer.float() * data['tokens-count'][m].unsqueeze(0)).mean(-1)
            # unit tokens association
            for unit_idx, unit_q99_mask in tqdm(enumerate(q99_mask), desc=f'[{m}] Layer {l}:'):
                # unit_data = [avg_act_uniform_dist[unit_idx].item(),\
                #              avg_act_true_dist[unit_idx].item(),\
                #              energy[unit_idx].item(),\
                #              q99[unit_idx].item(),\
                #              units_token_entropy[unit_idx].item(),]
                unit_data = [q99[unit_idx].item(),]
                # # measure how much this unit is different from the others
                # uniqueness = F.kl_div(units_token_distribution[unit_idx], units_token_distribution).mean()
                # unit_data.append(uniqueness.item())
                # extract the most associated tokens
                idx_q99_tokens = indices[unit_idx][unit_q99_mask]
                if args.decode_tokens:
                    q99_words = [tokenizer.decode(t) for t in idx_q99_tokens]
                else:
                    q99_words = [str(t) for t in idx_q99_tokens]
                # limit the number of tokens per unit
                q99_words = q99_words[:min(MAX_TOKENS, len(q99_words))]
                # remove '\n' from tokens
                if args.decode_tokens:
                    for t,r in REPLACE.items():
                        q99_words = [w.replace(t, r) for w in q99_words]
                # write into the file
                line = '\t'.join([f'Layer {l}', f'Unit {unit_idx}']\
                                  + [str(i) for i in unit_data]\
                                  + q99_words) + '\n'
                f.write(line)
                print(line)

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