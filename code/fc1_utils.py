import os
import argparse
import pickle
import torch
import pandas as pd
import numpy as np
import random

TRIGGER_TRESHOLD_FREQ_RATE=0.2

# utils

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist] 
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

def get_type(prompt):
    """
    Return the prompt type given the prompt name
    """
    if 'paraphrase' in prompt:
        return 'Human'
    elif 'autoprompt-filter' in prompt:
        return 'autoprompt-filter'
    elif 'autoprompt-no-filter' in prompt:
        return 'M-disc'
    elif 'optiprompt' in prompt:
        return 'M-cont'

def get_type_2(type_A, type_B):
    if type_A==type_B:
        type_str = type_A
    else:
        if (type_A=='Human' and type_B=='M-cont') or (type_B=='Human' and type_A=='M-cont'):
            type_str = f'mixte_Human_M-cont'
        elif (type_A=='Human' and type_B=='M-disc') or (type_B=='Human' and type_A=='M-disc'):
            type_str = f'mixte_Human_M-disc'
        elif (type_A=='M-cont' and type_B=='M-disc') or (type_B=='M-cont' and type_A=='M-disc'):
            type_str = f'mixte_M-cont_M-disc'
    return type_str

def cos_sim(A,B):
    A_dot_B = np.dot(A,B)
    A_mag = np.sqrt(np.sum(np.square(A)))
    B_mag = np.sqrt(np.sum(np.square(B)))
    dist = (A_dot_B / (A_mag * B_mag))
    return dist


# fc1 functions

def load_fc1(datapath, filenames):
    """
    Load the pickle files in the filenames list containing the fc1 activations.
    """
    # ----- load fc1
    all_fc1_act = []
    all_preds = {}
    for filename in filenames:
        print(f'[FC1] Loading {filename}...')
        with open(os.path.join(datapath, filename),"rb") as f:
            dataloaded = pickle.load(f)
            all_fc1_act += dataloaded['fc1']
            all_preds.update(dataloaded['preds'])
    return all_fc1_act, all_preds

def preprocess_fc1(all_fc1_act, count=False, triggered=False, nrg_att=False, overlap=False):
    """
    Preprocess the fc1 data according to the arguments of the function
    """
    from analyze_prompts import count_activated_neurons, find_triggered_neurons

    # ----- format  fc1
    print("[FC1] Formatting data...")    
    unique_relations = set(d['relation'] for d in all_fc1_act)
    unique_prompt = set(d['prompt'] for d in all_fc1_act)
    unique_layer = set(d['layer'] for d in all_fc1_act)
    # remove prompts where [Y] is not at the end
    all_fc1_act = [p for p in all_fc1_act \
                    if (p['template'][-3:]=='[Y]' 
                        or p['template'][-4:]=='[Y]?'
                        or p['template'][-5:]=='[Y] .')]
    # Get the prompt type
    [d.update({
        'type':get_type(d['prompt'])
    }) for d in all_fc1_act]
    # Add activated count
    if count:
        [d.update({
            'count':count_activated_neurons(
                d['sensibility'],
                torch.logspace(start=0,end=torch.log2(d['nb_facts']), steps=10, base=2)),
        }) for d in all_fc1_act]
    # Add triggered mask
    if triggered:
        [d.update({
            'triggered':find_triggered_neurons(
                d['sensibility'],
                d['nb_facts']*TRIGGER_TRESHOLD_FREQ_RATE)
        }) for d in all_fc1_act]
    # add layer_nrg_att
    if nrg_att:
        [d.update({
            'l_nrg_att':d['nrg_att'].mean().item(),
        }) for d in all_fc1_act]
    # Compute overlap
    if overlap:    
        all_fc1_act = flatten([[[[{
            'relation':rel,
            'template_A':d_A['template'],
            'template_B':d_B['template'],
            'prompt_A':d_A['prompt'],
            'prompt_B':d_B['prompt'],
            'layer': l,
            'type': get_type_2(d_A['type'],d_B['type']),
            'sim_nrg_att':cos_sim(d_A['nrg_att'].numpy(),d_B['nrg_att'].numpy()),
            'overlap':(torch.mul(d_B['triggered'],d_A['triggered']).sum()/(d_B['triggered']+d_A['triggered']).sum()).item()}
                for d_B in all_fc1_act if d_B['relation']==rel and d_B['layer']==l]
                    for d_A in all_fc1_act if d_A['relation']==rel and d_A['layer']==l]
                        for l in unique_layer]
                            for rel in unique_relations])
    return all_fc1_act

def dataframe_fc1(fc1_data, mode):
    """
    Embed the fc1 data into a pandas dataframe
    """
    print("[FC1] To dataframe...")    
    df = None
    if mode == 'count':         
        df = pd.DataFrame(
            data=flatten([[dict(datum, **{'treshold':t, 'count':c}) for t,c in enumerate(datum['count'])] for datum in fc1_data]),
            columns=['relation', 'prompt', 'template', 'layer', 'count', 'treshold'])
        # sort by layer
        df = df.sort_values(['layer'])
    elif mode == 'sensibility':
        df = pd.DataFrame(
            data=fc1_data,
            columns=['relation', 'prompt', 'template', 'layer', 'sensibility', 'micro', 'ppl', 'ent', 'type', 'l_nrg_att', 'nb_facts'])
        df['sensibility'] = df['sensibility'].apply(
                lambda l: [x.item() for x in l]
            )
        df['np_sensibility'] = df['sensibility'].apply(lambda a: np.array(a))
        df['normed_sensibility'] = df['np_sensibility'] / df['nb_facts'].apply(lambda a: np.array(a))
        df['micro'] = df['micro'].apply(
                lambda x: x*100
            )
        df['log_ppl'] = np.log(df['ppl'])
    elif mode == 'overlap':
        df = pd.DataFrame(
            data=fc1_data,
            columns=['relation', 'template_A','template_B','prompt_A','prompt_B','layer','overlap', 'type', 'sim_nrg_att'])
        df['name'] = [ tA+'/'+tB for (tA, tB) in zip(df['template_A'], df['template_B']) ]
        # sort by layer
        df = df.sort_values(['layer','template_A','template_B'])
    elif mode == 'minimal':
        df = pd.DataFrame(
            data=fc1_data,
            columns=['relation', 'prompt', 'template', 'layer', 'micro', 'ppl', 'ent', 'type'])
        df['micro'] = df['micro'].apply(
                lambda x: x*100
            )
    return df

def filter_templates(data, min_template_accuracy, only_best_template=False):
    if 'sensibility' in data:
        m = 'sensibility'
    elif 'minimal' in data:
        m = 'minimal'
    else:
        print('[FC1 Filtering] accuracy not available')
        exit(0)

    """
    Filter out relations where the best template for at least one prompt type is lower than min_template_accuracy
    This to avoid having relation with only one prompt type after the template filtering
    """
    max_type_min_relation_accuracy = data[m].groupby(['type', 'relation'])['micro'].max().groupby('relation').min().reset_index(name='min_acc')
    filtered_relations = max_type_min_relation_accuracy[max_type_min_relation_accuracy['min_acc'] > min_template_accuracy]['relation'].to_list()
    data[m] = data[m][data[m]['relation'].isin(filtered_relations)]
    """
    Filter templates to only keep those with an accuracy greater than min_template_accuracy
    """
    data[m] = data[m][data[m]['micro']>=min_template_accuracy]
    filtered_templates = data[m]['template'].drop_duplicates().to_list()
    
    if only_best_template:
        score_best_template = data[m].groupby(['type', 'relation'])['micro'].max().to_dict()
        best_template = {r:{t:data[m][data[m]['type']==t][data[m]['relation']==r][data[m]['micro']==score_best_template[(t,r)]]['template'].drop_duplicates().to_list() for t in data[m]['type'].unique()} for r in data[m]['relation'].unique()}
        best_template = [[random.sample(best_template[r][t], 1) for t in best_template[r]] for r in best_template]
        filtered_templates = flatten(best_template)
        data[m] = data[m][data[m]['template'].isin(filtered_templates)]

    # count the number of template per relation and prompt type
    data_size = data[m][['type', 'relation', 'template']].drop_duplicates().groupby(['type', 'relation']).size().reset_index(name='counts')

    print('[FC1 Filtering] All relations after filtering:', filtered_relations)
    print('[FC1 Filtering] All templates after filtering:', filtered_templates)
    print(f'[FC1 Filtering] Number of template per prompt type after filtering (>={min_template_accuracy}):')
    print(data_size.groupby(['type'])['counts'].apply(lambda x: {'mean':np.mean(x), 'max':np.max(x), 'min':np.min(x)}))

    if 'count' in data:
        data['count'] = data['count'][data['count']['relation'].isin(filtered_relations)]
        data['count'] = data['count'][data['count']['template'].isin(filtered_templates)]
    if 'overlap' in data:
        data['overlap'] = data['overlap'][data['overlap']['relation'].isin(filtered_relations)]
        data['overlap'] = data['overlap'][data['overlap']['template_A'].isin(filtered_templates)]
        data['overlap'] = data['overlap'][data['overlap']['template_B'].isin(filtered_templates)]
    return data

# main fc1 import function

def import_fc1(datapath, filenames, mode):

    all_fc1_act, all_preds = load_fc1(datapath, filenames)

    data = {}
    for m in mode:
        if m=='count':
            d = preprocess_fc1(all_fc1_act, count=True)
        elif m=='sensibility':
            d = preprocess_fc1(all_fc1_act, triggered=True, nrg_att=True)
        elif m=='overlap':
            d = preprocess_fc1(all_fc1_act, triggered=True, nrg_att=True, overlap=True)
        elif m=='minimal':
            d = preprocess_fc1(all_fc1_act)
        data[m] = dataframe_fc1(d, m)
    
    return data