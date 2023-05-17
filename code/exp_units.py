# python lib
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import os
import logging
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
    parser.add_argument('--n_units', type=int, default=500, help='How many units to be extracted')
    parser.add_argument('--k_tokens', type=int, default=100, help='How many tokens per units')
    parser.add_argument('--percentile_high', type=int, default=90, help='Percentile for highest activated units')
    parser.add_argument('--percentile_low', type=int, default=10, help='Percentile for lowest activated units')
    parser.add_argument('--percentile_typical_max', type=int, default=80, help='Top percentile for shared and typical units')
    parser.add_argument('--percentile_typical_min', type=int, default=20, help='Bottom percentile for share and typical units')

    # prompt types
    parser.add_argument('--autoprompt', action='store_true', help='adding autoprompt data')
    parser.add_argument('--optiprompt', action='store_true', help='adding optiprompt data')
    parser.add_argument('--paraphrase', action='store_true', help='adding paraphrase data')
    
    # unit types
    parser.add_argument('--shared_units', action='store_true', help='Extract shared units')
    parser.add_argument('--typical_units', action='store_true', help='Extract typical units')
    parser.add_argument('--high_units', action='store_true', help='Extract high units')
    parser.add_argument('--low_units', action='store_true', help='Extract low units')

    # filter by accuracy
    parser.add_argument('--min_template_accuracy', type=float, default=10.0, help='Remove all template with an accuracy lower than this treshold. From 0 to 100')
    parser.add_argument('--min_relation_accuracy_for_best_subset', type=float, default=30.0, help='Use to select a subset of relation having at least an accuracy of min_relation_accuracy with each prompt type')
    
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

def get_typical(high_sensibility_units_per_type, low_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types):
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
        typical_units[t] = random.sample(typical_units[t], min(n_units,len(typical_units[t])))
    return typical_units

def get_shared(high_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types):
    # keep units belonging to the top percentile for all prompt types
    shared_units = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
            if all([high_sensibility_units_per_type[high_sensibility_units_per_type['type']==t]['np_sensibility'].item()[i]\
                for t in prompt_types])]
    shared_units = random.sample(shared_units, min(n_units, len(shared_units)))
    return shared_units

def get_high(high_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types):
    # keep units having the highest activation
    high_units = {}
    for t in prompt_types:
        this_prompt_high_unit = high_sensibility_units_per_type[high_sensibility_units_per_type['type']==t]
        high_units[t] = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
                if this_prompt_high_unit['np_sensibility'].item()[i]]
        high_units[t] = random.sample(high_units[t], min(n_units,len(high_units[t])))
    return high_units

def get_low(low_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types):
    # keep unit having the lowest activations
    low_units = {}
    for t in prompt_types:
        this_prompt_high_unit = low_sensibility_units_per_type[low_sensibility_units_per_type['type']==t]
        low_units[t] = [(int(i/n_units_layer), i%n_units_layer) for i in range(n_layers*n_units_layer)\
                if this_prompt_high_unit['np_sensibility'].item()[i]]
        low_units[t] = random.sample(low_units[t], min(n_units,len(low_units[t])))
    return low_units

def unit_extraction(df, percentile_high, percentile_low, n_units, shared=False, typical=False, high=False, low=False):
    units = {}

    n_layers = len(df['layer'].unique())
    n_units_layer = len(df.sample(1).sensibility.item())
    prompt_types = df['type'].unique()

    # Compute avg sensibility per prompt type
    avg_sensibility_per_type = df[['layer', 'type', 'np_sensibility']].groupby(['layer', 'type']).mean()
    avg_sensibility_per_type_flat = avg_sensibility_per_type.groupby('type')['np_sensibility'].apply(list).apply(lambda x: np.concatenate(x))
    
    # filter unit to only keep those with a sensibility in the top p percentile (per type)
    high_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x>=np.percentile(x, percentile_high)).reset_index()
    low_sensibility_units_per_type = avg_sensibility_per_type_flat.apply(lambda x: x<=np.percentile(x, percentile_low)).reset_index()

    # extract shared, typical, high and low units
    if shared:
        units['shared'] = get_shared(high_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types)
    if typical:
        units['typical'] = get_typical(high_sensibility_units_per_type, low_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types)
    if high:
        units['high'] = get_high(high_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types)
    if low:
        units['low'] = get_low(low_sensibility_units_per_type, n_units, n_units_layer, n_layers, prompt_types)

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
            unit_indeces += flatten([units[r][m] for r in units])
        else:
            unit_indeces += flatten([flatten(units[r][m].values()) for r in units])
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

# log ----

def get_exp_setup(args, mode, prompt_type, relation, data):
    """
    Return a string containing the setup of the experiment
    """
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
                f'Accuracy (this prompt and relation): {this_accuracy}',]
    
    if mode == 'shared' or mode == 'typical':
        exp_setup += [
            f'Percentile typical max: {args.percentile_typical_max}',
            f'Percentile typical min: {args.percentile_typical_min}',]
        exp_name +=  '.vs.'.join(data['type'].unique())
    if mode == 'low':
        exp_setup += [
            f'Percentile low: {args.percentile_low}',]
    if mode == 'high':
        exp_setup += [
            f'Percentile high: {args.percentile_high}',]
        
    setup_str = '\t'.join(exp_setup)
    exp_name += f'{mode}_{prompt_type}_{relation}.txt'

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

    for rel in relations:

        units[rel] = {}

        # filter data to only keep templates related to the current relation
        if rel == 'all':
            df = data
        else:
            df = data[data['relation']==rel]

        # Extract shared and typical units
        print(f"[UNITS] Extracting shared and typical units for relation {rel}")
        units[rel].update(unit_extraction(df, args.percentile_typical_max, args.percentile_typical_min, args.n_units, shared='shared' in modes, typical='typical' in modes))
        
        # Extract low and high units (use a different percentile)
        print(f"[UNITS] Extracting high and low activation units for relation {rel}")
        units[rel].update(unit_extraction(df, args.percentile_high, args.percentile_low, args.n_units, high='high' in modes, low='low' in modes))

        # save the units into a file so I don't have to re-compute them again and again
        # TODO

    print(f"[UNITS] Extracting unit-token stats")
    topk_token_unit = token_extraction(units, model, args.k_tokens, args.batch_size, modes, debug)

    print(f"[UNITS] Saving stats")
    for rel in relations:
        for m in modes:

            if m == 'shared': # if writing shared unit-tokens, no need to create one file per prompt type
                data_units = {'':units[rel][m]}
            else:
                data_units = units[rel][m]

            for prompt_type, extracted_units in data_units.items(): 
                # Create a file with typical/shared/high/low unit-tokens given the prompt type
                setup_str, exp_name = get_exp_setup(args, m, prompt_type, rel, data)
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
    if args.fast_for_debug:
        # only two relations
        data['sensibility'] = data['sensibility'][data['sensibility']['relation'].isin(['P1001',])]
        # reduce the number of tokens and units to extract
        args.n_units = 3
        args.k_tokens = 2
    data = filter_templates(data, args.min_template_accuracy)

    # Select a subset of relations with "high" accuracy (greater than min_relation_accuracy_for_best_subset fo all prompt types considered in the experiment)
    min_type_relation_accuracy = data['sensibility'].groupby(['type', 'relation'])['micro'].mean().groupby('relation').min().reset_index()
    filtered_relations_2 = min_type_relation_accuracy[min_type_relation_accuracy['micro']>=args.min_relation_accuracy_for_best_subset]['relation'].tolist()
    selected_relations = ['all',] + filtered_relations_2 # all: because we also want to compute the stats on all relations
    print(f'[EXPERIMENT SETUP] Selection of best relations (>={args.min_relation_accuracy_for_best_subset}):', filtered_relations_2)

    if args.fast_for_debug:
        selected_relations = ['P1001',]

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

    # launch unit experiment
    unit_experiment(model, data['sensibility'], selected_relations, args, modes, debug=args.fast_for_debug)