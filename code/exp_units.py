# python lib
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import os
import logging
from datetime import datetime
# source code
from models import build_model_by_name
from fc1_utils import import_fc1, filter_templates


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--save_dir', type=str, default='/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/analyze/token_units')
    parser.add_argument('--fc1_datapath', type=str, default='data/fc1')

    # device, batch size
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')

    # language model
    parser.add_argument('--model_name', type=str, default=f'facebook/opt-350m', help='the huggingface model name')

    # parameters
    parser.add_argument('--n_units', type=int, default=500, help='How many units to be extracted, -1 to take all of them')
    parser.add_argument('--k_tokens', type=int, default=100, help='How many tokens per units')

    # prompt types
    parser.add_argument('--autoprompt', action='store_true', help='adding autoprompt data')
    parser.add_argument('--optiprompt', action='store_true', help='adding optiprompt data')
    parser.add_argument('--paraphrase', action='store_true', help='adding paraphrase data')
    
    # unit types
    parser.add_argument('--shared_units', action='store_true', help='Extract shared units')
    parser.add_argument('--typical_units', action='store_true', help='Extract typical units')
    parser.add_argument('--high_units', action='store_true', help='Extract high units')
    parser.add_argument('--low_units', action='store_true', help='Extract low units')

    # top/low unit filtering
    parser.add_argument('--percentile_high', type=int, default=90, help='Percentile for highest activated units')
    parser.add_argument('--percentile_low', type=int, default=10, help='Percentile for lowest activated units')
    parser.add_argument('--percentile_typical_max', type=int, default=80, help='Top percentile for shared and typical units')
    parser.add_argument('--percentile_typical_min', type=int, default=20, help='Bottom percentile for share and typical units')
    parser.add_argument('--global_unit_filtering_threshold', action='store_true', help='Use a threshold based on all types together when extracting top/low units')

    # filter by accuracy
    parser.add_argument('--min_template_accuracy', type=float, default=10.0, help='Remove all template with an accuracy lower than this treshold. From 0 to 100')
    parser.add_argument('--min_relation_accuracy_for_best_subset', type=float, default=30.0, help='Use to select a subset of relation having at least an accuracy of min_relation_accuracy with each prompt type')
    parser.add_argument('--best_template', action='store_true', help='Only keep the best template of each type-relation pair')

    # other filter
    parser.add_argument('--layers', type=str, default='all', help='Limit the study to one or more layer, e.g. l00,l01')


    # debug
    parser.add_argument('--fast_for_debug', action='store_true', help='toy version, for debugging')
    return parser.parse_args()

# utils ----

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist] 
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

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
# unit extraction ----

def get_typical(high_sensibility_units_per_type, low_sensibility_units_per_type, n_units_layer, n_layers, prompt_types):
    # keep units belonging to the top percentile for only one prompt types
    typical_units = {}
    for t in prompt_types:
        this_prompt_high_unit = high_sensibility_units_per_type[high_sensibility_units_per_type['type']==t]
        # othr_prompts_low_unit = low_sensibility_units_per_type[low_sensibility_units_per_type['type']!=t]
        # all units which have a low sensibility with the others prompts
        inter_othr_prompts_low_unit = [all([low_sensibility_units_per_type[low_sensibility_units_per_type['type']==t_bis]['np_sensibility'].item()[i]\
                                        for t_bis in prompt_types if t_bis != t])\
                                            for i in range(n_layers*n_units_layer)]
        typical_units[t] = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
                if all([this_prompt_high_unit['np_sensibility'].item()[i], inter_othr_prompts_low_unit[i]])]
        # typical_units[t] = random.sample(typical_units[t], min(n_units,len(typical_units[t])))
    return typical_units

def get_shared(high_sensibility_units_per_type, n_units_layer, n_layers, prompt_types):
    # keep units belonging to the top percentile for all prompt types
    shared_units = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
            if all([high_sensibility_units_per_type[high_sensibility_units_per_type['type']==t]['np_sensibility'].item()[i]\
                for t in prompt_types])]
    # shared_units = random.sample(shared_units, min(n_units, len(shared_units)))
    return shared_units

def get_high(high_sensibility_units_per_type, n_units_layer, n_layers, prompt_types):
    # keep units having the highest activation
    high_units = {}
    for t in prompt_types:
        this_prompt_high_unit = high_sensibility_units_per_type[high_sensibility_units_per_type['type']==t]
        high_units[t] = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
                if this_prompt_high_unit['np_sensibility'].item()[i]]
        # high_units[t] = random.sample(high_units[t], min(n_units,len(high_units[t])))
    return high_units

def get_low(low_sensibility_units_per_type, n_units_layer, n_layers, prompt_types):
    # keep unit having the lowest activations
    low_units = {}
    for t in prompt_types:
        this_prompt_high_unit = low_sensibility_units_per_type[low_sensibility_units_per_type['type']==t]
        low_units[t] = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
                if this_prompt_high_unit['np_sensibility'].item()[i]]
        # low_units[t] = random.sample(low_units[t], min(n_units,len(low_units[t])))
    return low_units


def plot_sensibility_dist(data, relations, args):
    import pandas as pd
    import seaborn as sns
    for r in relations:
        if r =='all':
            df = data['sensibility']
        else:
            df = data['sensibility'][data['sensibility']['relation']==r]
        # average over templates
        avg_sensibility_per_type = df[['layer', 'type', 'np_sensibility']].groupby(['layer', 'type']).mean()
        # trick to unfold the array
        dic = avg_sensibility_per_type.reset_index().to_dict()
        n = len(dic['layer'])
        data_dic = flatten([[{'layer':dic['layer'][i], 'type':dic['type'][i], 'val':dic['np_sensibility'][i][j]} for j in range(len(dic['np_sensibility'][i]))] for i in range(n)])
        # data = flatten([[{'type':t, 'unit': i, 'val':x[i]} for i in range(len(x))] for t,x in avg_sensibility_per_type_flat.to_dict().items()])
        df_plot = pd.DataFrame(data_dic)
        labels = list(df_plot['type'].unique())
        g = sns.FacetGrid(df_plot, col='layer', height=4, col_wrap=4)
        g.map_dataframe(sns.histplot, x='val', hue='type', hue_order=labels, log_scale=(False,True),  element="step", legend='full', common_bins=False)
        for ax in g.axes.ravel():
            ax.legend(labels=labels)
        path = os.path.join('..',args.save_dir,f'd_sensibility_{r}.pdf')
        g.savefig(path)




def unit_extraction(df, threshold_mode, percentile_high, percentile_low, shared=False, typical=False, high=False, low=False):
    units = {}

    n_layers = len(df['layer'].unique())
    n_units_layer = len(df.sample(1).sensibility.item())
    prompt_types = df['type'].unique()

    # Compute avg sensibility per prompt type
    avg_sensibility_per_type = df[['layer', 'type', 'np_sensibility']].groupby(['layer', 'type']).mean()
    avg_sensibility_per_type_flat = avg_sensibility_per_type.groupby('type')['np_sensibility'].apply(list).apply(lambda x: np.concatenate(x))
    
    # filter unit to only keep those with a sensibility in the top p percentile (per type)

    if threshold_mode == 'percentile_all':
        high_threshold = np.percentile(np.concatenate(avg_sensibility_per_type_flat.values), percentile_high)
        low_threshold = np.percentile(np.concatenate(avg_sensibility_per_type_flat.values), percentile_low)
        str_high_threshold = str(high_threshold)
        str_low_threshold = str(low_threshold)
        print(f'[UNITS] Top threshold: sensibility>{high_threshold}')
        print(f'[UNITS] Low threshold: sensibility<{low_threshold}')
        # print type-top/low-percentile for comparison
        str_high_threshold_type = ', '.join([f'{x} ({t})' for t,x in avg_sensibility_per_type_flat.apply(lambda x: np.percentile(x, percentile_high)).reset_index().values])
        str_low_threshold_type = ', '.join([f'{x} ({t})' for t,x in avg_sensibility_per_type_flat.apply(lambda x: np.percentile(x, percentile_low)).reset_index().values])
        print(f'[UNITS] (this threshold is not used) Type-Top threshold: sensibility> [{str_high_threshold_type}] percentile')
        print(f'[UNITS] (this threshold is not used) Type-Low threshold: sensibility< [{str_low_threshold_type}] percentile')
        high_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x>=high_threshold).reset_index()
        low_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x<=low_threshold).reset_index()
    elif threshold_mode == 'percentile_type':
        str_high_threshold = ', '.join([f'{x} ({t})' for t,x in avg_sensibility_per_type_flat.apply(lambda x: np.percentile(x, percentile_high)).reset_index().values])
        str_low_threshold = ', '.join([f'{x} ({t})' for t,x in avg_sensibility_per_type_flat.apply(lambda x: np.percentile(x, percentile_low)).reset_index().values])
        print(f'[UNITS] Top threshold: sensibility> [{str_high_threshold}] percentile')
        print(f'[UNITS] Top threshold: sensibility< [{str_low_threshold}] percentile')
        high_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x>=np.percentile(x, percentile_high)).reset_index()
        low_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x<=np.percentile(x, percentile_low)).reset_index()

    str_top = ','.join([f'{t}: '+str(x.sum()) for t,x in high_sensibility_units_per_type.values])
    str_low = ','.join([f'{t}: '+str(x.sum()) for t,x in low_sensibility_units_per_type.values])
    print(f'[UNITS] Number of top sensibility units: {str_top}')
    print(f'[UNITS] NUmber of low sensibility units: {str_low}')

    # extract shared, typical, high and low units
    if shared:
        units['shared'] = {
            'data': get_shared(high_sensibility_units_per_type, n_units_layer, n_layers, prompt_types),
            'high_threshold': str_high_threshold,
            'low_threshold': str_low_threshold,}
    if typical:
        units['typical'] = {
            'data': get_typical(high_sensibility_units_per_type, low_sensibility_units_per_type, n_units_layer, n_layers, prompt_types),
            'high_threshold': str_high_threshold,
            'low_threshold': str_low_threshold,}
    if high:
        units['high'] = {
            'data': get_high(high_sensibility_units_per_type, n_units_layer, n_layers, prompt_types),
            'high_threshold': str_high_threshold,
            'low_threshold': str_low_threshold,}
    if low:
        units['low'] = {'data': get_low(low_sensibility_units_per_type, n_units_layer, n_layers, prompt_types),
            'high_threshold': str_high_threshold,
            'low_threshold': str_low_threshold,}

    return units

# token extraction ----

def token_extraction(units, model, k_tokens, batch_size, modes, debug):
    """
    For each unit, give a list of the k tokens causing the highest activation.
    Each item of the list is a tuple (token_id, activation)
    """

    device = model.model.device

    # gather all the extracted units into a unified set (to avoid doublons)
    unit_indeces = []
    for m in modes:
        if m == 'shared':
            unit_indeces += flatten([units[r][m]['data'] for r in units])
        else:
            unit_indeces += flatten([flatten(units[r][m]['data'].values()) for r in units])
    unit_indeces = set(unit_indeces)

    # initialize a list to store the k_tokens for each unit
    topk_token_unit = {(l, nu):[None] * k_tokens for l, nu in unit_indeces}
    
    # a list of the token id from the model vocabulary
    tokens_id = [[t_id,] for t_id in range(model.tokenizer.vocab_size)]
    if debug:
        tokens_id = random.sample(tokens_id, 2*batch_size)
    # split the tokens_id list into batches
    batch_tokens_id = [tokens_id[i: i+batch_size] for i in range(0,len(tokens_id),batch_size)]
    
    # iterate on the tokens_id (i.e. the vocabulary)
    for bti in tqdm(batch_tokens_id):

        # forward pass
        input_ids = torch.tensor(bti)
        attention_mask = torch.ones_like(input_ids)
        model.model(input_ids.to(device), attention_mask.to(device))

        # get fc1
        fc1_act = model.get_fc1_act()
        fc1_act = [f.view(len(bti), 1, -1) for f in fc1_act.values()]
    
        # the slow and inefficient code starts here :'(
        for l, nu in unit_indeces: # iterate on the unit set
            """
            For each unit, we update a list of the k tokens having the highest activation.
            To do so, we compare the activation value of each token in the batch to the 
            best-token-list (called topk_token_unit) of the current unit (l,nu).

            The comparison is done in reverse order to save computation.
            """
            for b in range(len(bti)): # iterate on the batch of token ids
                i_top = len(topk_token_unit[(l, nu)]) -1 # reversed order
                insert_id = None # indicate where to insert the token candidate
                for top in reversed(topk_token_unit[(l, nu)]):
                    if (top is not None) and (fc1_act[l][b,0,nu] < top[1]):
                        if (i_top == len(topk_token_unit[(l, nu)]) -1):
                            # lower than the lowest: stop here, the candidate is discarded
                            break
                        else:
                            # higher than the lower but lower than i_top
                            insert_id = i_top + 1 # insert in the previous index
                            break
                    elif (top is None) or (fc1_act[l][b,0,nu] > top[1]): # test is the current token is among the top one
                        if (i_top == 0):
                            # higher than the very best one
                            insert_id = i_top # insert on the first index
                            break 
                        else: 
                            # higer than an item which is not the first one: continue and compare with the next best token
                            i_top -= 1 # reversed order
                if insert_id is not None:
                    # the token candidate will be inserted in the list of the best tokens for the current unit
                    for m in range(len(topk_token_unit[(l, nu)])-2, insert_id-1, -1): # shift
                        topk_token_unit[(l, nu)][m+1] = topk_token_unit[(l, nu)][m]  # the last is pop outed
                    topk_token_unit[(l, nu)][insert_id] = (bti[b][0], fc1_act[l][b,0,nu])

    return topk_token_unit

def unit2token(model, unit_indices, topk_tokens):
    """
    Given unit indices and top k units id, retrieve the token names and format the result into a string
    """
    l, nu = unit_indices
    top_tokens = [model.tokenizer.decoder[i] for i,_ in topk_tokens] # token names
    top_s = [str(s.item()) for _,s in topk_tokens] # token activations
    return '\t'.join([f'Layer {l}', f'Unit {nu}'] + top_tokens + top_s)

def sample_units(units, n_units):
    """
    Randomly sampling n_units for the set of extracted units
    """
    for r in units:
        for m in units[r]:
            if m == 'shared':
                units[r][m]['data'] = random.sample(units[r][m]['data'], min(n_units,len(units[r][m]['data'])))
            else:
                for t in units[r][m]['data']:
                    units[r][m]['data'][t] = random.sample(units[r][m]['data'][t], min(n_units,len(units[r][m]['data'][t])))
    return units

# log ----

def get_exp_setup(args, mode, prompt_type, relation, data, str_high_threshold, str_low_threshold):
    """
    Return a string containing the setup of the experiment
    """
    this_date = datetime.utcnow().strftime('%Y%m%d-%H:%M:%S')
    if relation == 'all':
        this_data = data[data['type']==prompt_type]
    else:
        this_data = data[data['type']==prompt_type][data['relation']==relation]
    n_templates = len(this_data['template'].unique())
    this_accuracy = this_data['micro'].mean()
    all_prompts = ','.join(data['type'].unique())
    exp_name = 'token_unit_'
    exp_setup = ['SETUP',
                f'n_units: {args.n_units}',
                f'k_tokens: {args.k_tokens}',
                f'Seed: {args.seed}',
                f'Model name: {args.model_name}',
                f'This prompt type: {prompt_type}',
                f'All prompt types: {all_prompts}',
                f'Min template accuracy: {args.min_template_accuracy}',
                f'Number of templates: {n_templates}',
                f'Accuracy (this prompt and relation): {this_accuracy}',
                f'High threshold: {str_high_threshold}',
                f'Low threshold: {str_low_threshold}']
    
    if mode == 'shared' or mode == 'typical':
        exp_setup += [
            f'Percentile typical max: {args.percentile_typical_max}',
            f'Percentile typical min: {args.percentile_typical_min}',]
        exp_name +=  '.vs.'.join(data['type'].unique()) + '_'
    if mode == 'low':
        exp_setup += [
            f'Percentile low: {args.percentile_low}',]
    if mode == 'high':
        exp_setup += [
            f'Percentile high: {args.percentile_high}',]
    
    if args.best_template:
        if relation == 'all':
            template = 'best'
        else:
            template = list(this_data['template'].unique())
        exp_setup += [
            f'Template: {template}'
        ]

    exp_setup += [
        f'Date: {this_date}',]

    setup_str = '\t'.join(exp_setup)
    exp_name += f'{mode}_{prompt_type}_{relation}_{this_date}.txt'

    if args.fast_for_debug:
        setup_str = 'DEBUG --------- '+setup_str
        exp_name = 'debug_'+exp_name

    return setup_str, exp_name

# main -----

def unit_experiment(model, data, relations, args, modes=['shared', 'typical', 'high', 'low'], debug=False):
    """
    Select the units (layer+index) being (a) the most type specific; and (b) shared among prompt types.

    Output:
    * top_unit_positive_difference: indeces of type-specific units (dataframe)
    * top_similar_units: indeces of shared units (list)
    """
    units = {} # a dictionnary containing the indices of the extracted units

    if args.global_unit_filtering_threshold:
        threshold_mode = 'percentile_all' # compute the threshold on all type together
    else: 
        threshold_mode = 'percentile_type' # compute a type-based threshold

    for rel in relations:

        units[rel] = {}

        # filter data to only keep templates related to the current relation
        if rel == 'all':
            df = data
        else:
            df = data[data['relation']==rel]

        # Extract shared and typical units
        if 'shared' in modes or 'typical' in modes:
            print(f"[UNITS] Extracting shared and typical units for relation {rel}")
            units[rel].update(unit_extraction(df, threshold_mode, args.percentile_typical_max, args.percentile_typical_min, shared='shared' in modes, typical='typical' in modes))
        
        # Extract low and high units (use a different percentile)
        if 'high' in modes or 'low' in modes:
            print(f"[UNITS] Extracting high and low activation units for relation {rel}")
            units[rel].update(unit_extraction(df, threshold_mode, args.percentile_high, args.percentile_low, high='high' in modes, low='low' in modes))

        # save the units into a file so I don't have to re-compute them again and again
        # TODO

    # write units in a files
    for r in units:
        for m in units[r]:
            str_high_threshold = units[r][m]['high_threshold']
            str_low_threshold = units[r][m]['low_threshold']
            if m == 'shared': # if writing shared unit-tokens, no need to create one file per prompt type
                data_units = {'':units[r][m]['data']}
            else:
                data_units = units[r][m]['data']
            for t in data_units:
                setup_str, exp_name = get_exp_setup(args, m, t, r, data, str_high_threshold, str_low_threshold)
                unit_ids_string = '\n'.join([f'Layer {l}\tUnit {u}' for l,u in data_units[t]]) 
                filepath = os.path.join('..',args.save_dir,'unit_ids_'+exp_name)
                print(f"[UNITS] Writing {filepath}...")
                with open(filepath, 'w') as f:
                    f.write(setup_str+'\n')
                    f.write(unit_ids_string + '\n')   

    if args.n_units != -1:
        units = sample_units(units, args.n_units)
    
    print(f"[UNITS] Extracting unit-token stats")
    topk_token_unit = token_extraction(units, model, args.k_tokens, args.batch_size, modes, debug)

    print(f"[UNITS] Saving stats")
    for rel in relations:

        for m in modes:
            str_high_threshold = units[rel][m]['high_threshold']
            str_low_threshold = units[rel][m]['low_threshold']
            if m == 'shared': # if writing shared unit-tokens, no need to create one file per prompt type
                data_units = {'':units[rel][m]['data']}
            else:
                data_units = units[rel][m]['data']

            for prompt_type, extracted_units in data_units.items(): 
                # Create a file with typical/shared/high/low unit-tokens given the prompt type
                setup_str, exp_name = get_exp_setup(args, m, prompt_type, rel, data, str_high_threshold, str_low_threshold)
                unit_tokens_string = '\n'.join([unit2token(model, unit_indices, topk_token_unit[unit_indices]) for unit_indices in extracted_units]) + '\n'
                filepath = os.path.join('..',args.save_dir,exp_name)
                print(f"[UNITS] Writing {filepath}...")
                with open(filepath, 'w') as f:
                    f.write(setup_str+'\n')
                    f.write(unit_tokens_string)
    print(f"[UNITS] Ended")

if __name__ == "__main__":

    # arguments
    args = parse_args()
    args.k = 1

    # set random seed
    set_seed(args.seed)

    # save directory
    model_str = args.model_name.split('/')[-1]
    args.save_dir = os.path.join(args.save_dir, model_str)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # init device
    device = init_device(args.device)

    # init model
    model = build_model_by_name(args)
    model.set_analyse_mode()
    model.model.to(device)

    # load fc1 data
    fc1_files = []
    if args.autoprompt:
        fc1_files += [f'fc1_att_data_{model_str}_t0_autoprompt-no-filter_fullvoc.pickle',]
    if args.optiprompt:
        fc1_files += [f'fc1_att_data_{model_str}_t0_optiprompt_fullvoc_fixetok.pickle',]
    if args.paraphrase:
        fc1_files += [f'fc1_att_data_{model_str}_t0_rephrase_fullvoc.pickle']
    data = import_fc1(args.fc1_datapath, fc1_files, mode=['sensibility',])

    if args.layers != 'all':
        data['sensibility'] = data['sensibility'][data['sensibility']['layer'].isin(args.layers.split(','))]

    if args.fast_for_debug:
        # only two relations
        data['sensibility'] = data['sensibility'][data['sensibility']['relation'].isin(['P1001','P176'])]
        # reduce the number of tokens and units to extract
        args.n_units = 3
        args.k_tokens = 2
    data = filter_templates(data, args.min_template_accuracy, only_best_template=args.best_template)

    # Select a subset of relations with "high" accuracy (greater than min_relation_accuracy_for_best_subset fo all prompt types considered in the experiment)
    min_type_relation_accuracy = data['sensibility'].groupby(['type', 'relation'])['micro'].mean().groupby('relation').min().reset_index()
    filtered_relations_2 = min_type_relation_accuracy[min_type_relation_accuracy['micro']>=args.min_relation_accuracy_for_best_subset]['relation'].tolist()
    selected_relations = ['all',] + filtered_relations_2 # all: because we also want to compute the stats on all relations
    print(f'[EXPERIMENT SETUP] Selection of best relations (>={args.min_relation_accuracy_for_best_subset}):', filtered_relations_2)

    if args.fast_for_debug:
        selected_relations = ['P1001','P176',]

    # which units to extract:
    modes = []
    if args.shared_units:
        modes.append('shared')
    if args.typical_units:
        modes.append('typical')
    if args.high_units:
        modes.append('high')
    if args.low_units:
        modes.append('low')

    # plot_sensibility_dist(data, selected_relations, args)

    # launch unit experiment
    unit_experiment(model, data['sensibility'], selected_relations, args, modes, debug=args.fast_for_debug)