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
import json
from scipy.stats import pearsonr

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from typing import Callable

from utils import load_vocab
from utils import load_optiprompt, free_optiprompt
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

MODEL='opt-1.3b'
EXP_NAME=f'opt-{MODEL}-autoprompt'

SENSIBILITY_TRESHOLD=0
TRIGGER_TRESHOLD_FREQ_RATE=0.2
LOAD_FC1=[
    #f"../data/fc1/fc1_data_{MODEL}_t0_optiprompt_fullvoc_fixetok.pickle",
        # f"../data/fc1/fc1_ppl_pred_data_{MODEL}_t0_autoprompt-filter.pickle",
     #     f"../data/fc1/fc1_ppl_data_{MODEL}_t0_autoprompt-no-filter_fullvoc.pickle",
          f"../data/fc1/fc1_ppl_data_{MODEL}_t0_rephrase_fullvoc.pickle"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=f'facebook/{MODEL}', help='the huggingface model name')
    parser.add_argument('--output_dir', type=str, default='/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/analyze', help='the output directory to store prediction results')
    parser.add_argument('--common_vocab_filename', type=str, default='none', help='common vocabulary of models (used to filter triples)')
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

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def load_pred(prompt, relation):
    prompt = prompt.split('.')[0]
    samples = load_file(os.path.join('./data/eval/', prompt, 'facebook_opt-350m/%s.jsonl'%(relation)))
    # make sure that samples are in the same order
    pred = [s["topk"][0]["token"] for s in samples]
    return pred

def compute_agreement(pred_A, pred_B):
    score = float(sum([a==b for a,b in zip(pred_A, pred_B)]))/float(len(pred_A))
    return score

if __name__ == "__main__":

    args = parse_args()
    
    save_dir = os.path.join(args.output_dir, EXP_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # ----- init
    init_device(args.device)
    set_seed(args.seed)
    ## model
    model = build_model_by_name(args)
    ## vocab
    if args.common_vocab_filename!='none':
        vocab_subset = load_vocab(args.common_vocab_filename)   
    else:
        vocab_subset = list(model.inverse_vocab.keys())
    logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
    ## tokenizer
    # tokenizer.add_tokens(['[X]', '[Y]'])

    # ----- load fc1
    all_fc1_act = []
    all_preds = {}
    for filename in LOAD_FC1:
        print(f'Loading {filename}...')
        with open(os.path.join(args.output_dir, filename),"rb") as f:
            dataloaded = pickle.load(f)
            all_fc1_act += dataloaded['fc1']
            all_preds.update(dataloaded['preds'])


    # ----- format fc1
    print("Formatting data...")    
    unique_relations = set(d['relation'] for d in all_fc1_act)
    unique_prompt = set(d['prompt'] for d in all_fc1_act)
    unique_layer = set(d['layer'] for d in all_fc1_act)
    # remove prompts where [Y] is not at the end
    all_fc1_act = [p for p in all_fc1_act \
                   if (p['template'][-3:]=='[Y]' 
                       or p['template'][-4:]=='[Y]?'
                       or p['template'][-5:]=='[Y] .')]
    # Prompt name
    def get_type(prompt):
        if 'paraphrase' in prompt:
            return 'paraphrase'
        elif 'autoprompt-filter' in prompt:
            return 'autoprompt-filter'
        elif 'autoprompt-no-filter' in prompt:
            return 'autoprompt-no-filter'
        elif 'optiprompt' in prompt:
            return 'optiprompt'
    [d.update({
        'type':get_type(d['prompt'])
    }) for d in all_fc1_act]
    # # modify template for optiprompts # done during extraction
    # [d.update({
    #     'template+':f"{d['relation']}_{d['prompt'].split('seed')[-1]}_{d['template']}" if d['type']=='optiprompt' else d['template']
    # }) for d in all_fc1_act]
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
        'type': d_B['type'] if (d_B['type']==d_A['type']) else 'mixte',
        'overlap':(torch.mul(d_B['triggered'],d_A['triggered']).sum()/(d_B['triggered']+d_A['triggered']).sum()).item()}
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
        columns=['relation', 'prompt', 'template', 'layer', 'sensibility', 'micro', 'ppl', 'type'])
    df_sensibility['sensibility'] = df_sensibility['sensibility'].apply(
            lambda l: [x.item() for x in l]
        )
    df_sensibility['micro'] = df_sensibility['micro'].apply(
            lambda x: x*100
        )
    df_sensibility['log_ppl'] = np.log(df_sensibility['ppl'])
    df_overlap = pd.DataFrame(
        data=predict_triggered_overlap,
        columns=['relation', 'template_A','template_B','prompt_A','prompt_B','layer','overlap', 'type'])
    df_overlap['name'] = [ tA+'/'+tB for (tA, tB) in zip(df_overlap['template_A'], df_overlap['template_B']) ]
    # sort by layer
    df_count = df_count.sort_values(['layer'])
    df_overlap = df_overlap.sort_values(['layer','template_A','template_B'])

    # ---------------------
    #   METRICS
    # ---------------------
    df_micro = df_sensibility[['prompt','relation','micro', 'type']].set_index('prompt').drop_duplicates().groupby(['prompt', 'type'])
    avg_micro = df_micro.mean()
    std_micro = df_micro.std()
    max_micro = df_micro.max()
    avg_avg_micro = avg_micro.groupby('type').mean()
    std_avg_micro = avg_micro.groupby('type').std()
    print("Average micro: ", avg_avg_micro)
    print("Std micro: ", std_avg_micro)


    # measure distance in OPT input embedding between all pairs of templates
    input_rep = []
    def save_input_hook() -> Callable:
        def fn(_, input, output):
            input_rep.append(input[0].detach().mean(1).cpu().squeeze().numpy())
        return fn
    model.model.model.decoder.layers[0].register_forward_hook(save_input_hook())
    input_rep_dic = {}
    for idx, diter in df_overlap[['template_A', 'prompt_A', 'relation']].drop_duplicates().iterrows():
        t, p, r = diter['template_A'], diter['prompt_A'], diter['relation']
        if 'optiprompt' in p: # it is an optiprompt, we have to load the vectors
            # add optiprompts tokens to the model em4beddings
            original_vocab_size = len(list(model.tokenizer.get_vocab()))
            load_optiprompt(model, os.path.join("data/prompts/",p), original_vocab_size, r)
            flag_free_optiprompt = True
            input = model.tokenizer.encode(t.split('_')[-1].replace('[X]','').replace('[Y]',''), return_tensors="pt")
            model.model(input)
            input_rep_dic[t]=input_rep[-1]
            free_optiprompt(model, original_vocab_size)
        else:
            input = model.tokenizer.encode(t.replace('[X]','').replace('[Y]',''), return_tensors="pt")
            model.model(input)
            input_rep_dic[t]=input_rep[-1]
    del input_rep
    df_overlap['d_in'] = [cos_sim(input_rep_dic[tA], input_rep_dic[tB]) for tA, tB in zip(df_overlap['template_A'], df_overlap['template_B'])]

    # measure difference in prediction 
    df_overlap['d_out'] = [compute_agreement(all_preds[pA], all_preds[pB]) for pA, pB in zip(df_overlap['template_A'], df_overlap['template_B'])]

    # ---------------------
    #   CORRELATION
    # ---------------------

    def calculate_pvalues(df):
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
        return pvalues

    metrics_pair = ['overlap', 'd_in', 'd_out']
    metrics_solo = ['micro', 'ppl']
    control_pair = ['layer', 'relation',]
    control_solo = ['relation',]

    pair ={
        'df': df_overlap,
        'metrics':metrics_pair,
        'control':control_pair,
    }
    solo ={
        'df': df_sensibility,
        'metrics':metrics_solo,
        'control':control_solo,
    }
    

    def get_corr(df, metrics, t, ctrl, ctrl_i):
        corr = df[metrics].corr(method='pearson').unstack().reset_index()
        corr = corr.rename(columns={0:'pearson corr.'})
        corr['pval'] = calculate_pvalues(df[metrics]).unstack().reset_index()[0]
        # remove symmetrical correlation
        corr = corr[~pd.DataFrame(np.sort(corr[['level_0','level_1']], axis=1), index=corr.index).duplicated()]
        # remove correlation between the same metric
        corr = corr.loc[corr.apply(lambda r: r['level_0'] != r['level_1'], axis=1)]
        corr['label'] = [ctrl_i,]*len(corr)
        corr['control'] = [ctrl,]*len(corr)
        corr['type'] = [t,]*len(corr)
        return corr

    corr_data = {}
    for exp_name, corr_conf in {'pair':pair, 'solo':solo}.items():
        control = corr_conf['control']
        metrics = corr_conf['metrics']
        df = corr_conf['df']
        ctrl_corr = []
        for t in df['type'].unique():
            df_type = df[df['type']==t]
            for ctrl in control:
                temp = []
                for ctrl_i in df_type[ctrl].unique():
                    df_ctrl = df_type[df_type[ctrl]==ctrl_i]
                    corr = get_corr(df_ctrl, metrics, t, ctrl, ctrl_i)
                    temp.append(corr)
                temp_concat = pd.concat(temp)
                temp_concat['name'] = ['/'.join([l0, l1]) for (l0,l1) in zip(temp_concat['level_0'], temp_concat['level_1'])]
                ctrl_corr.append(temp_concat)
            # all
            corr = get_corr(df_type,metrics, t, 'all', 'all')
            corr['name'] = ['/'.join([l0, l1]) for (l0,l1) in zip(corr['level_0'], corr['level_1'])]
            ctrl_corr.append(corr)
        # all
        corr = get_corr(df,metrics, 'all', 'all', 'all')
        corr['name'] = ['/'.join([l0, l1]) for (l0,l1) in zip(corr['level_0'], corr['level_1'])]
        ctrl_corr.append(corr)
        corr_data[exp_name] = pd.concat(ctrl_corr)


    # ---------------------
    #   PLOTS
    # ---------------------
    
    """
    Display correlations
    """
    import plotly.express as px

    for corr_conf, corr_datum in corr_data.items():
        for prompt_type in corr_datum['type'].unique():
            
            corr_type = corr_datum[corr_datum['type']==prompt_type]

            print(f'[{corr_conf}, {prompt_type}] P value:')
            print(corr_type[corr_type['label']=='all'])
            
            if prompt_type == 'all':
                fig = px.bar(corr_type, x='name', y="pearson corr.", color='control',
                       hover_data=['label', 'pval'], title=f'{corr_conf} {prompt_type}')
                fig.show()
                fig.write_html(os.path.join('..',save_dir,f'corr_{corr_conf}_{prompt_type}.html'))
            else:
                fig = px.box(corr_type, x='name', y="pearson corr.", color='control',
                            points="all", hover_data=['label', 'pval'], title=f'{corr_conf} {prompt_type}')
                fig.show()
                fig.write_html(os.path.join('..',save_dir,f'corr_{corr_conf}_{prompt_type}.html'))

    


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
    fig.write_html(os.path.join('..',save_dir,f'fc1_reduce_proj_all_rel.html'))

    """
    Display distance in the OPT input embedding space vs. neural overlap
    """

    fig = scatter_slider(
        df_overlap,
        x='overlap',
        y='d_in',
        s='layer',
        c='relation',
        t='name',
        sb='type',
        title=f"Overlap vs. input cosine (all relations)",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'fc1_overlap_vs_in.html'))

    # for r in unique_relations:
    #     df_rel = df_overlap[df_overlap['relation']==r]
        
    #     fig = scatter_slider(
    #     df_rel,
    #     x='overlap',
    #     y='d_in',
    #     s='layer',
    #     c='relation',
    #     t='name',
    #     title=f"[{r}] Overlap vs. input cosine",)
    #     fig.show()
    #     fig.write_html(os.path.join('..',save_dir,f'fc1_overlap_vs_in_{r}.html'))


    """
    Display distance in prediction vs. neural overlap
    """

    fig = scatter_slider(
        df_overlap,
        x='overlap',
        y='d_out',
        s='layer',
        c='relation',
        t='name',
        sb='type',
        title=f"Overlap vs. agreement (all relations)",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'fc1_overlap_vs_out.html'))

    """
    Display distance in prediction vs. distance in the OPT input embedding space
    """

    fig = scatter_slider(
        df_overlap,
        x='d_in',
        y='d_out',
        s='relation',
        c='type',
        t='name',
        sb='type',
        title=f"Input cosine vs. agreement (all relations)",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'fc1_out_vs_in.html'))


    """
    Display micro vs. ppl
    """

    fig = scatter_slider(
        df_sensibility,
        x='micro',
        y='log_ppl',
        s='relation',
        c='type',
        t='template',
        sb='type',
        title=f"Micro vs. PPL",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'micro_vs_ppl.html'))

