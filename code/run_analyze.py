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

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from typing import Callable

from utils import load_vocab
from models import build_model_by_name
from analyze_prompts import *

sns.set(font_scale=1.0)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------
#
#       PARAMS
#
# --------------------

SENSIBILITY_TRESHOLD=0
TRIGGER_TRESHOLD_FREQ_RATE=0.2
LOAD_FC1=["../data/fc1/fc1_data_opt-350m_t0_autoprompt-filter.pickle",]
        #   "../data/fc1/fc1_data_opt-350m_t0_autoprompt-no-filter.pickle",
        #   "../data/fc1/fc1_data_opt-350m_t0_rephrase.pickle"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')
    parser.add_argument('--output_dir', type=str, default='/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/analyze', help='the output directory to store prediction results')
    parser.add_argument('--common_vocab_filename', type=str, default='./data/vocab/common_vocab_opt_probing_prompts.txt', help='common vocabulary of models (used to filter triples)')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
    parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
    return parser.parse_args()

def init_device(device):
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

def set_seed(seed):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist] 
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

def cos_sim(A,B):
        A_dot_B = np.dot(A,B)
        A_mag = np.sqrt(np.sum(np.square(A)))
        B_mag = np.sqrt(np.sum(np.square(B)))
        dist = (A_dot_B / (A_mag * B_mag))
        return dist

if __name__ == "__main__":

    args = parse_args()
    
    # ----- init
    init_device(args.device)
    set_seed(args.seed)
    ## model
    model = build_model_by_name(args)
    ## vocab
    vocab_subset = load_vocab(args.common_vocab_filename)
    logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.add_tokens(['[X]', '[Y]'])

    # ----- load fc1
    all_fc1_act = []
    for filename in LOAD_FC1:
        print(f'Loading {filename}...')
        with open(os.path.join(args.output_dir, filename),"rb") as f:
            all_fc1_act += pickle.load(f)

    # ----- format fc1
    print("Formatting data...")    
    unique_relations = set(d['relation'] for d in all_fc1_act)
    unique_prompt = set(d['prompt'] for d in all_fc1_act)
    unique_layer = set(d['layer'] for d in all_fc1_act)
    # Add activated count
    [d.update({
        'count':count_activated_neurons(
            d['sensibility'],
            torch.logspace(start=0,end=torch.log2(d['nb_facts']), steps=10, base=2)),
    }) for d in all_fc1_act]
    # Add triggered mask
    [d.update({
        'triggered':find_triggered_neurons(
            d['sensibility'],
            d['nb_facts']*TRIGGER_TRESHOLD_FREQ_RATE)
    }) for d in all_fc1_act]
    # Compute overlap
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

    # ----- to dataframe         
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

    # ---------------------
    #   METRICS
    # ---------------------

    # measure distance in OPT input embedding between all pais of templates
    input_rep = []
    def save_input_hook() -> Callable:
        def fn(_, __, output):
            input_rep.append(output.detach().mean(1).cpu().squeeze().numpy())
        return fn
    model.model.model.decoder.project_in.register_forward_hook(save_input_hook())
    input_rep_dic = {}
    for t in df_overlap['template_A'].unique():
        input = tokenizer.encode(t.replace('[X]','').replace('[Y]',''), return_tensors="pt")
        model.model(input)
        input_rep_dic[t]=input_rep[-1]
    del input_rep
    df_overlap['d_in'] = [cos_sim(input_rep_dic[tA], input_rep_dic[tB]) for tA, tB in zip(df_overlap['template_A'], df_overlap['template_B'])]

    # ---------------------
    #   PLOTS
    # ---------------------
    
    """
    Display all prompts -- encdoded with the neural activation -- into a tSNE-reduced 2d space
    """
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

    """
    Display distance in the OPT input embedding space vs. neural overlap
    """

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
