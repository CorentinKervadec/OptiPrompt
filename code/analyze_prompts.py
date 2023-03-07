import torch
import os
import argparse
import pickle
import logging
import random
import numpy as np

from utils import get_relation_meta
from utils import load_vocab, load_data, batchify, analyze, get_relation_meta

import seaborn as sns
# import matplotlib.pyplot as plt

# import plotly.figure_factory as ff
# import plotly.graph_objects as go

from models import build_model_by_name

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, TruncatedSVD
# from plotly.validators.scatter.marker import SymbolValidator


"""
Template utils
"""

def init_template(prompt_file, relation):
    relation = get_relation_meta(prompt_file, relation)
    return relation['template']

def read_paraphrase(filename):
    # Open file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # template dictionary
    rephrase_dic = {}

    # iterate over lines
    current_relation = ''
    current_type = ''
    cpt_id = 0
    for l in lines:
        l = l.replace('\n', '')
        if l[0]=='*': # Relation name
            current_relation, current_type = l[1:].split()
            rephrase_dic[current_relation] = {'type': current_type}
            rephrase_dic[current_relation]['templates'] = []
            rephrase_dic[current_relation]['format'] = []
            rephrase_dic[current_relation]['micro'] = [] # score / will be filled during evaluation
        elif l[0]=='#': # Description
            rephrase_dic[current_relation]['description'] = l[2:]
        elif l[0] in ['S', 'Q']:
            rephrase_dic[current_relation]['templates'].append(l[3:])
            rephrase_dic[current_relation]['format'].append(l[0])
            cpt_id += 1 # increase id count

    return rephrase_dic

"""
Extract fc1 activations with a forward pass on the model
"""
def run_fc1_extract(model, all_prompt_files, relation_list, logger, test_data_dir, filter_indices, index_list,
                    vocab_subset, batch_size, sensibility_treshold,):
    all_fc1_act = []
    rl_2_nbfacts = {} 
    prompt2template = {prompt.split('/')[-1]:{rel:"" for rel in relation_list} for prompt in all_prompt_files}

    for prompt_file in all_prompt_files:

        if 'paraphrase' in prompt_file:
            rephrase_dic = read_paraphrase(prompt_file)

        for relation in relation_list:
            relation = relation.split(".")[0]
            print("RELATION {}".format(relation))

            if 'paraphrase' in prompt_file:
                template_list = rephrase_dic[relation]['templates']
            else:
                template_list = [init_template(prompt_file, relation)]

            for template in template_list:

                logger.info('Template: %s'%template)
                prompt2template[prompt_file.split('/')[-1]][relation] = template

                test_data = os.path.join(test_data_dir, relation, "test.jsonl")
                eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
                eval_samples_batches, eval_sentences_batches = batchify(eval_samples, batch_size)
                micro, result, fc1_act = analyze(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=None)
                
                # rl_2_nbfacts[relation]=torch.tensor(sum([len(batch) for batch in eval_samples_batches]))
                # directly store the binary tensor of activated neurons to save memory

                all_fc1_act += [dict(
                    relation = relation,
                    prompt = prompt_file.split('/')[-1],
                    template = template,
                    layer = l,
                    micro = micro,
                    nb_facts = torch.tensor(sum([len(batch) for batch in eval_samples_batches])),
                    sensibility_treshold = sensibility_treshold,
                    sensibility = compute_freq_sensibility(masked_act_l,sensibility_treshold))
                        for l, masked_act_l in select_pred_masked_act(fc1_act).items()]

    return all_fc1_act#, rl_2_nbfacts, prompt2template
        
"""
Handling fc1 data
"""
def select_pred_masked_act(res):
    # 1) Select the activation associated to the predict token
    masked_act = [
        {'l{:02d}'.format(i):torch.masked_select(act, batchum['predict_mask'].unsqueeze(-1)).view(batchum['predict_mask'].size(0),act.size(-1))
            for i, (layer, act) in enumerate(batchum['activations'].items())}
                for batchum in res
            ]
    layers_name = list(masked_act[0].keys())
    # 2) concat batch dim: [layer x (batch_size*nb_batch)] /!\ the last batch is not full
    cat_masked_act = {layer:torch.cat([act[layer] for act in masked_act]) for layer in layers_name}
    return cat_masked_act

def compute_freq_sensibility(act, treshold=0): # replace treshold by an interval
    # 1) Compute sensibilty: the neuron is activated if fc1 > treshold
    sensibility = torch.gt(act, treshold)
    # 2) Measure how frequently the neuron is activated
    freq_sensibility = sensibility.count_nonzero(dim=0)
    # torch.gt(h.view(batch_size, n_tokens, fc1_dim),0)
    return freq_sensibility

def count_activated_neurons(act, interval):
    count = [(torch.gt(act,t).count_nonzero(dim=0)/act.size(-1)*100).item() for t in interval]
    return count

def find_triggered_neurons(count, treshold):
    triggered = torch.gt(count,treshold)
    return triggered
"""
Plot utils
"""
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)#, row_cluster=False)

def heatmap_slider(df, x, y, z, s, z_scale, title):
    # Create figure
    fig = go.Figure()

    if z_scale is not None:
        zmin, zmax = z_scale
        zauto = False
    else:
        zmin, zmax = (None, None)
        zauto = True

    steps=[]

    # Add traces, one for each slider step
    for i,l in enumerate(df[s].unique()):
        df_layer=df[df[s]==l]
        fig.add_trace(
            go.Heatmap(z=df_layer[z],
                x=df_layer[x],
                y=df_layer[y],
                zauto = zauto, zmin = zmin, zmax = zmax, colorscale='edge',
                visible=False))
        step = dict(
            method="update",
            label=l,
            args=[{"visible": [False] * len(df[s].unique())},
                {"title": title + f' | {s}: {l}'}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make 1st trace visible
    fig.data[0].visible = True
      
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{s}: "},
        pad={"t": 50},
        steps=steps,
        transition= {'duration': 300, 'easing': 'cubic-in-out'},
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(paper_bgcolor="LightSteelBlue")

    return fig

def heatmap_pairs_button(df, x, y, z, pA, pB, z_scale, title):

    # Create figure
    fig = go.Figure()

    if z_scale is not None:
        zmin, zmax = z_scale
        zauto = False
    else:
        zmin, zmax = (None, None)
        zauto = True

    buttons = []
    nb_pairs = (len(df[pA].unique())*len(df[pB].unique()))
    visible_tmplt = [False,]*nb_pairs
    df_heatmap = [None,]*nb_pairs
    i = 0

    for p_A in df[pA].unique():
        df_A = df[df[pA]==p_A]

        # Add traces, one for each slider step
        for p_B in df_A[pB].unique():

            df_heatmap[i]=df_A[df_A[pB]==p_B].copy()
            
            name=f"{p_A}/{p_B}"
        
            fig.add_trace(
                go.Heatmap(z=df_heatmap[i][z], x=df_heatmap[i][x], y=df_heatmap[i][y],
                zauto = zauto, zmin = zmin, zmax = zmax, colorscale='edge',
                name=name, visible=False))
            
            visible = visible_tmplt.copy()
            visible[i] = True

            buttons.append(
                dict(label=name,
                        method="update",
                        args=[{"visible": visible},
                            {"title": title,}])
            )
            i += 1
            
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons
            )
        ])

    return fig

def dendrogram_average(df, x, y, z, avg):

    df = df[[x,y,z,avg]]
    df_avg = df.groupby(by=[x, y]).mean(['z']).reset_index() 

    pivot = df_avg.pivot(index=x, columns=y, values=z)

    fig = ff.create_dendrogram(pivot, labels=pivot.index, 
                               color_threshold=1.5, orientation='left')

    return fig

def clustermap_average(df, x, y, z, avg):

    fig = plt.figure()

    df = df[[x,y,z,avg]]
    df_avg = df.groupby(by=[x, y]).mean().reset_index() 

    pivot = df_avg.pivot(index=x, columns=y, values=z)

    sns.clustermap(
        pivot,
        col_cluster=False)
    
    return fig

def reduce_proj(df, x, z, sl, c, title, algo, use_symbols=True):

    # Create figure
    fig = go.Figure()

    n = 2

    if algo == 'tsne':
        proj = TSNE(n_components=n, random_state=0, perplexity=3.0)
    elif algo == 'pca':
        proj = PCA(n_components=n,)
    elif algo == 'tsvd':
        proj = TruncatedSVD(n_components=n,)

    df = df[[x,z,sl,c]]
    # df = df.set_index(x)
    # print(df[z].to_numpy())

    steps=[]

    # Add traces, one for each slider step
    for i,l in enumerate(df[sl].unique()):
        df_layer = df[df[sl]==l]
        # pivot = df_layer.pivot(index=x, columns=y, values=z)
        
        features = np.stack(df_layer[z].to_numpy())
        projections = proj.fit_transform(features)
        explained = proj.explained_variance_ratio_ if algo != 'tsne' else [0]
        labels = df_layer[x]
        if use_symbols:
            symbols = [SymbolValidator().values[4*sb] for sb in range(len(labels))]
        else:
            symbols = SymbolValidator().values[0]

        fig.add_trace(
            go.Scatter(
            x=projections[:,0],
            y=projections[:,1],
            # z=projections[:,2],
            mode='markers',
            marker=dict(
                color=df_layer[c],
                size=16,#df_layer[sz],
                symbol=symbols,
                cmax=100,
                cmin=0,
                colorbar=dict(
                    title="Colorbar"
                ),
                colorscale="Viridis"
            ),
            
            # marker_color=list(range(len(labels))),
            text=labels,
            visible=False)
        )
        fig.update_traces(textposition='top center')

        step = dict(
            method="update",
            label=l,
            args=[{"visible": [False] * len(df[sl].unique())},
                {"title": title + f' | {sl}: {l} | e: ' + ','.join(['d%d: %.2f '%(k,x) for k,x in enumerate(explained)])}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    # Make 1st trace visible
    fig.data[0].visible = True
      
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{sl}: "},
        pad={"t": 50},
        steps=steps,
        transition= {'duration': 300, 'easing': 'cubic-in-out'},
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(paper_bgcolor="LightSteelBlue")

    return fig


if __name__ == "__main__":

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

    RELATIONS_TEST = [
        "P1001",
        "P101",
        "P103",
        "P106",
        "P108",
        "P127",
        "P1303",
        "P131",
        "P136",
        "P1376",
        "P138",
        "P140",
        "P1412",
        "P159",
        "P17",
        "P176",
        "P178",
        "P19",
        "P190",
        "P20",
        "P264",
        "P27",
        "P276",
        "P279",
        "P30",
        "P36",
        "P361",
        "P364",
        "P37",
        "P39",
        "P407",
        "P413",
        "P449",
        "P463",
        "P47",
        "P495",
        "P530",
        "P740",
        "P937",
    ]

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
    all_prompt_files = args.prompt_files.split(',')

    filename = f"fc1_data_{args.model_name.split('/')[-1]}_t{SENSIBILITY_TRESHOLD}_rephrase.pickle"
    
    all_fc1_act = run_fc1_extract(
        model, all_prompt_files, relation_list, logger, args.test_data_dir, filter_indices,
        index_list, vocab_subset, args.eval_batch_size * n_gpu, SENSIBILITY_TRESHOLD)

    print("Saving fc1 activitaion into ", filename)
    with open(os.path.join(args.output_dir, filename),"wb") as f:
        pickle.dump(all_fc1_act,f)