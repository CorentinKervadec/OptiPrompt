import torch
import os
import numpy as np

from utils import get_relation_meta
from utils import load_vocab, load_data, batchify, analyze, get_relation_meta

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from plotly.validators.scatter.marker import SymbolValidator


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
