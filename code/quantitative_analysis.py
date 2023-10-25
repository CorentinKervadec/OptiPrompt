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
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from typing import Callable

from fc1_utils import import_fc1, filter_templates
# # initialize the model and the tokenizer
# model = AutoModelForCausalLM.from_pretrained('facebook/opt-350m')
# tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')

# # initialize the hook
# temp_hook = [] # temporary variable used to store the result of the hook
# def save_input_hook() -> Callable:
#     def fn(_, input, output):
#         # input is what is fed to the layer
#         temp_hook.append(input[0].detach().mean(1).cpu().squeeze().numpy())
#     return fn
# # register the hook the the layers 0.
# # So we'll extract the representation at the input of layer 0
# model.model.decoder.layers[0].register_forward_hook(save_input_hook())

# # iterate on the data
# candidates = ["Let", "blue", "."]
# candidates_tkn = [tokenizer.encode(c,return_tensors="pt") for c in candidates]
# candidates_tkn = [c[:, 1:] for c in candidates_tkn] # remove the <eos> token
# [model.model(c_in) for c_in in candidates_tkn]

# # get the embeddings
# embeddings = {c:temp_hook[i] for i, c in enumerate(candidates)}

# --------------------
#
#       PARAMS
#
# --------------------

MODEL='opt-350m'
EXP_NAME=f'{MODEL}'

SENSIBILITY_TRESHOLD=0
TRIGGER_TRESHOLD_FREQ_RATE=0.2
LOAD_FC1=[
        f"../data/fc1/fc1_att_data_{MODEL}_t0_optiprompt_fullvoc_fixetok.pickle",
        # f"../data/fc1/fc1_ppl_pred_data_{MODEL}_t0_autoprompt-filter.pickle",
        f"../data/fc1/fc1_att_data_{MODEL}_t0_autoprompt-no-filter_fullvoc.pickle",
        f"../data/fc1/fc1_att_data_{MODEL}_t0_rephrase_fullvoc.pickle"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=f'facebook/{MODEL}', help='the huggingface model name')
    parser.add_argument('--output_dir', type=str, default='/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/analyze', help='the output directory to store prediction results')
    parser.add_argument('--common_vocab_filename', type=str, default='none', help='common vocabulary of models (used to filter triples)')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
    parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
    parser.add_argument('--fp16', action='store_true', help='use half precision')
    parser.add_argument('--min_template_accuracy', type=float, default=10.0, help='Remove all template with an accuracy lower than this treshold. From 0 to 100')
    parser.add_argument('--min_relation_accuracy_for_best_subset', type=float, default=30.0, help='Use to select a subset of relation having at least an accuracy of min_relation_accuracy with each prompt type')
    parser.add_argument('--best_template', action='store_true', help='Only keep the best template of each type-relation pair')
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
    return device

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

    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    args = parse_args()
    
    save_dir = os.path.join(args.output_dir, EXP_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    """
    Initialize the model
    """
    device = init_device(args.device)
    set_seed(args.seed)
    ## model
    model = build_model_by_name(args)
    model.set_analyse_mode()
    model.model.to(device)
    ## vocab
    if args.common_vocab_filename!='none':
        vocab_subset = load_vocab(args.common_vocab_filename)   
    else:
        vocab_subset = list(model.inverse_vocab.keys())
    logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)

    """
    Load the data:
    - predictions
    - fc1 stats
    """
    # load preds
    all_preds = {}
    for filename in LOAD_FC1:
        print(f'Loading {filename}...')
        with open(os.path.join(args.output_dir, filename),"rb") as f:
            dataloaded = pickle.load(f)
            all_preds.update(dataloaded['preds'])
    # load fc1 stats
    data = import_fc1(args.output_dir, LOAD_FC1, mode=['sensibility', 'overlap'])
    data = filter_templates(data, args.min_template_accuracy, only_best_template=args.best_template)
   
    df_sensibility = data['sensibility']
    df_overlap = data['overlap']

    """
    Compute accuracy + PPL + Consistency
    """
    df_scores = df_sensibility[['prompt','relation','micro', 'type', 'ppl', 'ent', 'l_nrg_att']].groupby(['prompt', 'relation', 'type']).mean().reset_index()
    df_scores = df_scores.set_index('prompt').drop_duplicates()
    avg_micro = df_scores.groupby(['type','relation']).mean().groupby(['type']).mean()
    max_micro = df_scores.groupby(['type','relation']).max().groupby(['type']).mean() # best template for each relation
    consistency = df_scores.groupby(['type', 'relation']).std().groupby(['type']).mean()
    print("Average: ", avg_micro)
    print("Consistency: ", consistency)
    print("Max: ", max_micro)

    #df_sensibility[['prompt','relation','micro', 'type', 'ppl', 'ent', 'l_nrg_att']].groupby(['type']).quantile(0.025).reset_index()
    #df_sensibility[['prompt','relation','micro', 'type', 'ppl', 'ent', 'l_nrg_att']].groupby(['type']).quantile(0.975).reset_index()


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
            input = model.tokenizer.encode(t.split('_')[-1].replace('[X]','').replace('[Y]',''), return_tensors="pt").to(device)
            model.model(input)
            input_rep_dic[t]=input_rep[-1]
            free_optiprompt(model, original_vocab_size)
        else:
            input = model.tokenizer.encode(t.replace('[X]','').replace('[Y]',''), return_tensors="pt").to(device)
            model.model(input)
            input_rep_dic[t]=input_rep[-1]
    del input_rep
    df_overlap['d_in'] = [cos_sim(input_rep_dic[tA], input_rep_dic[tB]) for tA, tB in zip(df_overlap['template_A'], df_overlap['template_B'])]

    # measure difference in prediction 
    df_overlap['d_out'] = [compute_agreement(all_preds[pA], all_preds[pB]) for pA, pB in zip(df_overlap['template_A'], df_overlap['template_B'])]

    print('[type] Metrics avg: ', df_overlap.groupby('type').mean())
    print('[type] Metrics std: ', df_overlap.groupby('type').std())
    for layer in ['l03', 'l12', 'l20']:
        print(f'[{layer}] Metrics avg: ', df_overlap[df_overlap['layer']==layer].groupby('type').mean())
        print(f'[{layer}] All Metrics avg: ', df_overlap[df_overlap['layer']==layer].mean())

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

    metrics_pair = ['overlap', 'd_in', 'd_out', 'sim_nrg_att']
    metrics_solo = ['micro', 'ppl', 'ent', 'l_nrg_att']
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

    for corr_conf, corr_datum in corr_data.items():
        for prompt_type in corr_datum['type'].unique():
            
            corr_type = corr_datum[corr_datum['type']==prompt_type]

            print(f'[{corr_conf}, {prompt_type}] P value:')
            print(corr_type[corr_type['label']=='all'])

    # bootstrap overlap mean (with replacement):
    S_sample=len(df_sensibility['template'].unique().tolist()) # size of the sample = nb of templates
    N_sample=100 # number of sampling (initialised with the sample size)
    lowerbound_cache = None
    ee = 1e-3 # convergence threshold
    e_conv = 1
    while e_conv > ee:
        bootstrap_list = []
        for n in tqdm(range(N_sample), desc=f'Number of samples: {N_sample} | Convergence error: {e_conv}'):
            # sample S_sample templates
            sampled_templates = random.choices(df_sensibility['template'].unique().tolist(), k=S_sample)
            df_sample = df_overlap[df_overlap['template_A'].isin(sampled_templates)&df_overlap['template_B'].isin(sampled_templates)]
            bootstrap_list.append(df_sample.groupby('type').mean(numeric_only=True))

        dcat = pd.concat(bootstrap_list)
    
        if lowerbound_cache is not None:
            e_conv = abs(lowerbound_cache - dcat.groupby('type').quantile(0.025)).max().max()
        lowerbound_cache = dcat.groupby('type').quantile(0.025)

        # Increase number of samples
        N_sample = 2*N_sample

    dcat.groupby('type').mean()
    dcat.groupby('type').quantile(0.025)
    dcat.groupby('type').quantile(0.975)

    # coorelation : bootstrap overlap mean (with replacement):
    S_sample=len(df_sensibility['template'].unique().tolist()) # size of the sample = nb of templates
    N_sample=100 # number of sampling (initialised with the sample size)
    lowerbound_cache = None
    ee = 5e-3 # convergence threshold
    e_conv = 1
    while e_conv > ee:
        bootstrap_corr_list = []
        for n in tqdm(range(N_sample), desc=f'Number of samples: {N_sample} | Convergence error: {e_conv}'):
            # sample S_sample templates
            sampled_templates = random.choices(df_sensibility['template'].unique().tolist(), k=S_sample)
            df_sample = df_overlap[df_overlap['template_A'].isin(sampled_templates)&df_overlap['template_B'].isin(sampled_templates)]
            # corr
            for t in df_sample['type'].unique():
                df_type = df_sample[df_sample['type']==t]
                corr = get_corr(df_type, ['overlap', 'd_in', 'd_out'], t, 'all', 'all')
                corr['name'] = ['/'.join([l0, l1]) for (l0,l1) in zip(corr['level_0'], corr['level_1'])]
                bootstrap_corr_list.append(corr)

        dcat = pd.concat(bootstrap_corr_list)
    
        if lowerbound_cache is not None:
            e_conv = abs(lowerbound_cache - dcat.groupby('type').quantile(0.025)).max().max()
        lowerbound_cache = dcat.groupby('type').quantile(0.025)

        # Increase number of samples
        N_sample = 2*N_sample





    # ---------------------
    #   PLOTS
    # ---------------------

    import seaborn as sns

    """
    Compare prompts while controling the agreement
    """
    df_overlap_agreement = []
    for t_agreement in [0.1, 0.3, 0.5, 0.7, 0.9]:
        filtered = df_overlap[df_overlap['d_out']>(t_agreement-0.1)][df_overlap['d_out']<t_agreement+0.1]
        filtered['agreement'] = [t_agreement,]*len(filtered)
        df_overlap_agreement.append(filtered)
        print(f'[{t_agreement}] Size: ', filtered.groupby('type').size())
        print(f'[{t_agreement}] Metrics avg: ', filtered.groupby('type').mean())
        # print(f'[{t_agreement}] Metrics std: ', df_overlap[df_overlap['d_out']>t_agreement].groupby('type').std())
    df_overlap_agreement = pd.concat(df_overlap_agreement)

    sns.set(font_scale=0.6)
    g = sns.relplot(data=pd.melt(df_overlap_agreement, ['relation', 'template_A', 'template_B', 'prompt_A', 'prompt_B', 'layer', 'type', 'name', 'agreement']),
                 x='agreement', y='value', hue='variable', kind='line', col='type')
    plt.tight_layout(h_pad=2, w_pad=2)
    # plt.show()
    plt.savefig(os.path.join('..',save_dir,f'agreement.pdf'))

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
    Cluster templates given their agreements
    """    
    fig = reduce_proj(
        df_sensibility,
        x='template',
        z='cpmr_d_out',
        sl='relation',
        c='d_out_cluster',
        sb='type',
        title=f"Reduce proj",
        algo='tsne',
        n=2,
        discrete_colors=True,
        n_neighbors=20,
        size=10,
    )
    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'agreement_cluster.html'))


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

    """
    Display entropy vs. ppl
    """

    fig = scatter_slider(
        df_sensibility,
        x='ent',
        y='log_ppl',
        s='layer',
        c='type',
        t='template',
        sb='type',
        title=f"Entropy vs. PPL",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'ent_vs_ppl.html'))

    """
    Display micro vs. entropy
    """

    fig = scatter_slider(
        df_sensibility,
        x='micro',
        y='ent',
        s='layer',
        c='type',
        t='template',
        sb='type',
        title=f"Micro vs. entropy",)

    fig.show()
    fig.write_html(os.path.join('..',save_dir,f'ent_vs_micro.html'))