import sys
sys.path.insert(0, '/homedtcl/ckervadec/OptiPrompt')
import argparse
import os
import random
import logging
import torch
import pickle
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_vocab

from models import build_model_by_name
from analyze_prompts import init_template
from analyze_prompts import select_pred_masked_act, compute_freq_sensibility,\
    count_activated_neurons, find_triggered_neurons
from analyze_prompts import run_fc1_extract
from analyze_prompts import draw_heatmap, heatmap_slider,\
     heatmap_pairs_button, dendrogram_average, clustermap_average, reduce_proj, \
     levenshtein, scatter_slider
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from typing import Callable


sns.set(font_scale=1.0)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')
parser.add_argument('--output_dir', type=str, default='/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/analyze', help='the output directory to store prediction results')
parser.add_argument('--common_vocab_filename', type=str, default='../data/vocab/common_vocab_opt_probing_prompts.txt', help='common vocabulary of models (used to filter triples)')

parser.add_argument('--test_data_dir', type=str, default="../data/filtered_LAMA_opt")
parser.add_argument('--eval_batch_size', type=int, default=32)

parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--output_predictions', default=True, help='whether to output top-k predictions')
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
parser.add_argument('--output_all_log_probs', action="store_true", help='whether to output all the log probabilities')

parser.add_argument('--prompt_files', type=str, default='../prompts/marco_rephrasing/relation-paraphrases_v2.txt,../prompts/LAMA_relations.jsonl,data/prompts/my-autoprompt-filter-causal-facebook-opt-350m_seed0.jsonl', help='prompt file separated by coma')
parser.add_argument('--relation', type=str, default='all', help='which relation to evaluate.')

args = parser.parse_args()

SENSIBILITY_TRESHOLD=0
TRIGGER_TRESHOLD_FREQ_RATE=0.2
# PROMPT_FILES="../prompts/marco_rephrasing/relation-paraphrases_v2.txt,../prompts/LAMA_relations.jsonl,../data/prompts/my-autoprompt-filter-causal-facebook-opt-350m_seed0.jsonl,../data/marco_rephrasing/best_rephrase_opt-350m_QS.jsonl"
PROMPT_FILES="../prompts/marco_rephrasing/relation-paraphrases_v2.txt,../data/prompts/my-autoprompt-filter-causal-facebook-opt-350m_seed0.jsonl"


# Initialize GPUs
device=torch.device(args.device)
if args.device == 'cuda':
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        logger.warning('No GPU found! exit!')
    logger.info('# GPUs: %d'%n_gpu)

elif args.device == 'mps':
    n_gpu = 1
else:
    logger.info('# Running on CPU')
    n_gpu = 0

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(args.seed)


model = build_model_by_name(args)

# Turn on the analyse mode
model.set_analyse_mode()

if args.common_vocab_filename is not None:
    vocab_subset = load_vocab(args.common_vocab_filename)
    logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
else:
    filter_indices = None
    index_list = None

if args.output_all_log_probs:
    model.k = len(vocab_subset)

if args.relation=='all':
    relation_list = RELATIONS_TEST
else:
    relation_list=[r for r in args.relation.split(',')]

# read prompt files
all_prompt_files = PROMPT_FILES.split(',')

LOAD_DATA = True
filename = f"fc1_data_{args.model_name.split('/')[-1]}_t{SENSIBILITY_TRESHOLD}_rephrase.pickle"
if not LOAD_DATA:
    all_fc1_act = run_fc1_extract(
        model, all_prompt_files, relation_list, logger, args.test_data_dir, filter_indices,
        index_list, vocab_subset, args.eval_batch_size * n_gpu, SENSIBILITY_TRESHOLD)

    with open(os.path.join(args.output_dir, filename),"wb") as f:
        pickle.dump(all_fc1_act,f)
else:
    with open(os.path.join(args.output_dir, filename),"rb") as f:
        all_fc1_act = pickle.load(f)

unique_relations = set(d['relation'] for d in all_fc1_act)
unique_prompt = set(d['prompt'] for d in all_fc1_act)
unique_layer = set(d['layer'] for d in all_fc1_act)

# Add activated count
[d.update({
    'count':count_activated_neurons(
        d['sensibility'],
        torch.logspace(start=0,end=torch.log2(d['nb_facts']), steps=10, base=2)),
}) for d in all_fc1_act]

# Measure activation overlap between prompts

# Add triggered mask
[d.update({
    'triggered':find_triggered_neurons(
        d['sensibility'],
        d['nb_facts']*TRIGGER_TRESHOLD_FREQ_RATE)
}) for d in all_fc1_act]


# Compute overlap

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist] 
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

predict_triggered_overlap = flatten([[[[{
    'relation':rel,
    'template_A':d_A['template'],
    'template_B':d_B['template'],
    'prompt_A':d_A['prompt'],
    'prompt_B':d_B['prompt'],
    'layer': l,
    'overlap':(torch.mul(d_B['triggered'],d_A['triggered']).sum()/d_A['triggered'].sum()).item()}
        for d_B in all_fc1_act if d_B['relation']==rel and d_B['layer']==l]
            for d_A in all_fc1_act if d_A['relation']==rel and d_A['layer']==l]
                for l in unique_layer]
                    for rel in unique_relations])
                    
df_count = pd.DataFrame(
    data=flatten([[dict(datum, **{'treshold':t, 'count':c}) for t,c in enumerate(datum['count'])] for datum in all_fc1_act]),
    columns=['relation', 'prompt', 'template', 'layer', 'count', 'treshold'])

df_sensibility = pd.DataFrame(
    data=all_fc1_act,
    columns=['relation', 'prompt', 'template', 'layer', 'sensibility', 'micro'])
df_sensibility['sensibility'] = df_sensibility['sensibility'].apply(
        lambda l: [x.item() for x in l]
    )
df_sensibility['micro'] = df_sensibility['micro'].apply(
        lambda x: x*100
    )



df_overlap = pd.DataFrame(
    data=predict_triggered_overlap,
    columns=['relation', 'template_A','template_B','prompt_A','prompt_B','layer','overlap'])
df_overlap['name'] = [ tA+'/'+tB for (tA, tB) in zip(df_overlap['template_A'], df_overlap['template_B']) ]
# sort by layer
df_count = df_count.sort_values(['layer'])
df_overlap = df_overlap.sort_values(['layer','template_A','template_B'])

for rel in unique_relations:
    df_rel = df_overlap[df_overlap['relation']==rel]

    fig = heatmap_slider(
        df_rel,
        x="template_A",
        y="template_B",
        z="overlap",
        s="layer",
        z_scale=(0.0, 1.0),
        title=f"[{rel}] Prompt overlap")

    fig.show()
    fig.write_html(os.path.join('..',args.output_dir,f'fc1_cmpr_promts_{rel}.html'))

# Adapted to compare few prompts accross relation and layers
fig = heatmap_pairs_button(
    df_overlap,
    x="layer",
    y="relation",
    z="overlap",
    pA="template_A",
    pB="template_B",
    z_scale=(0.0, 1.0),
    title=f"Compare A/B")

fig.show()
fig.write_html(os.path.join('..',args.output_dir,'fc1_cmpr_promts_AB.html'))

for r in unique_relations:

    df_rel = df_overlap[df_overlap['relation']==r]
    fig = dendrogram_average(
        df_rel,
        x='template_A',
        y='template_B',
        z='overlap',
        avg='layer',
    )
    fig.show()
    fig.write_html(os.path.join('..',args.output_dir,f'fc1_dendro_{r}.html'))

for r in unique_relations:
    df_rel = df_sensibility[df_sensibility['relation']==r]
    
    fig = reduce_proj(
        df_rel,
        x='template',
        z='sensibility',
        sl='layer',
        c='micro',
        title=f"[{r}] Reduce proj",
        algo='pca',
    )
    fig.show()

    fig.write_html(os.path.join('..',args.output_dir,f'fc1_reduce_proj_{r}.html'))

fig = reduce_proj(
    df_sensibility,
    x='template',
    z='sensibility',
    sl='layer',
    c='relation',
    sb='prompt',
    title=f"Reduce proj",
    algo='tsne',
    n=2,
    discrete_colors=True,
    n_neighbors=20,
    size=10,
)
fig.show()

fig.write_html(os.path.join('..',args.output_dir,f'fc1_reduce_proj_all_rel.html'))

model_name = 'facebook/opt-350m'

# plot heatmap with levenshtein distance between all pais of templates

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.add_tokens(['[X]', '[Y]'])

templates = df_sensibility['template'].unique()
tkn_templates = [tokenizer.encode(t) for t in templates]

# print(templates)

col_A = []
col_B = []
lev = []
# [[(col_A.append(templates[ia]), col_B.append(templates[ib]), lev.append(levenshtein(tA, tB))) for ib, tB in enumerate(tkn_templates)] for ia, tA in enumerate(tkn_templates)]
# df_distance = pd.DataFrame({'template_A':col_A, 'template_B':col_B, 'levenshtein':lev})


df_overlap['d_lev'] = [levenshtein(tokenizer.encode(tA), tokenizer.encode(tB)) for tA, tB in zip(df_overlap['template_A'], df_overlap['template_B'])]


fig = scatter_slider(
    df_overlap,
    x='overlap',
    y='d_lev',
    s='layer',
    t='name',
    title=f"Overlap vs. levenshtein (all relations)",)

fig.show()

fig.write_html(os.path.join('..',args.output_dir,f'fc1_overlap_vs_lev.html'))

for r in unique_relations:
    df_rel = df_overlap[df_overlap['relation']==r]
    
    fig = scatter_slider(
    df_rel,
    x='overlap',
    y='d_lev',
    s='layer',
    t='name',
    title=f"[{r}] Overlap vs. levenshtein",)

    fig.show()
    fig.write_html(os.path.join('..',args.output_dir,f'fc1_overlap_vs_lev_{r}.html'))


model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

input_rep = []

def save_input_hook() -> Callable:
    def fn(_, __, output):
        input_rep.append(output.detach().mean(1).cpu().squeeze().numpy())
    return fn

model.model.decoder.project_in.register_forward_hook(save_input_hook())

input_rep_dic = {}

for t in df_overlap['template_A'].unique():
    input = tokenizer.encode(t.replace('[X]','').replace('[Y]',''), return_tensors="pt")
    model(input)
    input_rep_dic[t]=input_rep[-1]

del input_rep

def cos_distance(A,B):
    A_dot_B = np.dot(A,B)
    A_mag = np.sqrt(np.sum(np.square(A)))
    B_mag = np.sqrt(np.sum(np.square(B)))
    dist = 1.0 - (A_dot_B / (A_mag * B_mag))
    return dist

df_overlap['d_in'] = [cos_distance(input_rep_dic[tA], input_rep_dic[tB]) for tA, tB in zip(df_overlap['template_A'], df_overlap['template_B'])]

print(df_overlap['d_in'])

fig = scatter_slider(
    df_overlap,
    x='overlap',
    y='d_in',
    s='layer',
    t='name',
    title=f"Overlap vs. input cosine (all relations)",)

fig.show()

fig.write_html(os.path.join('..',args.output_dir,f'fc1_overlap_vs_in.html'))

for r in unique_relations:
    df_rel = df_overlap[df_overlap['relation']==r]
    
    fig = scatter_slider(
    df_rel,
    x='overlap',
    y='d_in',
    s='layer',
    t='name',
    title=f"[{r}] Overlap vs. input cosine",)

    fig.show()

    fig.write_html(os.path.join('..',args.output_dir,f'fc1_overlap_vs_in_{r}.html'))
