from utils import load_data, batchify
import os
from models.causallm_connector import CausalLM
from argparse import Namespace
from fc1_utils import import_fc1, filter_templates
from tqdm import tqdm
import torch
from typing import Callable
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--extract_gradient', action='store_true')
parser.add_argument('--analyse_units', action='store_true')
parser.add_argument('--softmax', action='store_true')
parser.add_argument('--typical_units_file', type=str, default='/homedtcl/ckervadec/analyze/token_units/my_selection/unit_ids_token_unit_autoprompt-no-filter.vs.paraphrase_typical_autoprompt-no-filter_all_20230530-13_29_04.txt')
parser.add_argument('--shared_units_file', type=str, default='/homedtcl/ckervadec/analyze/token_units/my_selection/unit_ids_token_unit_autoprompt-no-filter.vs.paraphrase_typical_autoprompt-no-filter_all_20230530-13_29_04.txt')
parser.add_argument('--grad_data_path', type=str, default='/homedtcl/ckervadec/analyze/opt-350m_autoprompt-no-filter_grad_analysis.pickle')
parser.add_argument('--prompt_types', type=str, default='autoprompt,paraphrase')
parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
args = parser.parse_args()


OUTPUT_PATH = '../../analyze'

model_string = args.model_name.split('/')[-1]
if args.softmax:
    model_string += '-softmax'

model_args = Namespace(
    model_name=args.model_name,
    k=5,
    fp16=False)
model = CausalLM(model_args)
model.model.train()
tokenizer = model.tokenizer
vocab = model.tokenizer.get_vocab()
hidden_dim = model.model.model.decoder.layers[0].fc1.out_features


if args.extract_gradient:
    data_dir = "../../data/filtered_LAMA_opt"
    batch_size=8

    """ Initialise the model using the optiprompt wrapper.

    Note that now we have to use the tokenizer associated to this wrapper.
    (in practice it is almost equivalent to the one we used before)
    """

    """
    Load templates
    """
    model_split = args.model_name.split('/')[-1]
    files = [f'fc1_att_data_{model_split}_t0_autoprompt-no-filter_fullvoc.pickle', 
            f'fc1_att_data_{model_split}_t0_optiprompt_fullvoc_fixetok.pickle',
            f'fc1_att_data_{model_split}_t0_rephrase_fullvoc.pickle']
    datapath = '../../data/fc1_data'

    # import the data from the pickle files
    mode = 'minimal'
    fc1_data = import_fc1(datapath, files, mode=[mode])

    # if you want to filter the templates
    fc1_data = filter_templates(fc1_data, min_template_accuracy=10, only_best_template=False)[mode]
    df_templates = fc1_data[['type', 'template', 'relation']].drop_duplicates()

    """
    Define a hook that save the fc1 with its gradient
    """

    layers = model.model.model.decoder.layers
    temp_fc1 = {layer: torch.empty(0) for layer in layers} # temporary variable used to store the result of the hook
    def save_fc1_hook(layer) -> Callable:
        def fn(_, input, output):
            # input is what is fed to the layer
            output.retain_grad() # force pytorch to store the gradient of this non-parameters tensor
            temp_fc1[layer] = output
        return fn

    # Setting a hook for saving FFN intermediate output
    for layer in layers:
        for name, sub_layer in layer.named_modules():
            if name == 'fc1':
                sub_layer.register_forward_hook(save_fc1_hook(layer))


    """
    Iterate
    """
    for prompt_type in df_templates['type'].unique():

        if not any([p in prompt_type for p in args.prompt_types.split(',')]):
            continue

        # define a variable to accumulate the gradient
        # gradient for all templates will be accumulated
        fc1_grad_store = {}
        
        for relation in df_templates['relation'].unique():

            print(f'[{prompt_type}|{relation}]')

            # lama datapath
            lama_data = os.path.join(data_dir, relation, "test.jsonl")

            # get templates
            templates_list = df_templates[df_templates['type']==prompt_type][df_templates['relation']==relation]['template'].to_list()

            # test relation sample size
            samples = load_data(lama_data, templates_list[0], vocab_subset=vocab, mask_token='[MASK]')
            if len(samples) == 0:
                continue
            
            # store the gradient distribution fot this relation
            fc1_grad_store[relation] = torch.zeros((len(layers),hidden_dim))

            for template in templates_list:

                # load the LAMA data for the current relation
                # and combine them with the given template
                samples = load_data(lama_data, template, vocab_subset=vocab, mask_token='[MASK]')
                samples_batches, sentences_batches = batchify(samples, batch_size)

                for i in tqdm(range(len(samples_batches)), desc=f'{template}'):
                    input, masked_indices_list, labels_tensor, mlm_label_ids, predict_mask = model.get_input_tensors_batch_train(sentences_batches[i], samples_batches[i])
                    with torch.enable_grad():
                        output = model.model(**input.to(model.model.device))
                        predict_logits = output.logits
                        if args.softmax:
                            predict_logits = torch.softmax(predict_logits, dim=-1)
                        last_token_logits = predict_logits[predict_mask]
                        predict_argmax = torch.argmax(last_token_logits, dim=-1)
                        argmax_logits = torch.gather(input=last_token_logits, axis=-1, index=predict_argmax.unsqueeze(-1))
                        # compute the gradient of the argmax logit, at the batch level
                        argmax_logits.sum().backward(retain_graph=True)
                        
                    for i_l, l in enumerate(layers):
                        fc1_grad = temp_fc1[l].grad
                        # reshape
                        fc1_grad = fc1_grad.view(input.input_ids.size(0), input.input_ids.size(1), -1)
                        # only activation associated to the tokens used for prediction
                        fc1_grad = fc1_grad[predict_mask]
                        # sum over the batch dimension
                        fc1_grad = fc1_grad.sum(dim=0)
                        # add to the grad_store
                        fc1_grad_store[relation][i_l] += fc1_grad.cpu()
                    # zero the gradient to avoid accumulation between batches
                    model.model.zero_grad()

            # Normalise the gradient accumulation to get the distribution (over all layers)
            fc1_grad_store[relation] = fc1_grad_store[relation] / (len(templates_list)*len(samples))
            if fc1_grad_store[relation].isnan().any():
                print(relation)
                print(fc1_grad_store[relation])
                exit(0)
        # save the average grad dist over relation 
        stack_rel_grad = torch.stack(list(fc1_grad_store.values()), dim=0)
        fc1_grad_dist = {
            'mean': stack_rel_grad.mean(dim=0),
            'std': stack_rel_grad.std(dim=0),
        }
        save_path = os.path.join(OUTPUT_PATH, f'{model_string}_{prompt_type}_grad_analysis.pickle')
        with open(save_path, 'wb') as handle:
            pickle.dump(fc1_grad_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # free mem
        # cpu
        del fc1_grad_store
        del stack_rel_grad
        del fc1_grad_dist
        # device

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist]
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

def load_pickles(f):
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    return data

if args.analyse_units:
    # load grad data
    grad_data = load_pickles(args.grad_data_path)
    
    # load unit file
    with open(args.shared_units_file, 'r') as f:
        shared_units = f.read()
    shared_units = [l.split('\t') for l in shared_units.split('\n')]
    header = shared_units[0]
    shared_units = shared_units[1:]
    shared_units = [{'Layer':int(l[0].split()[1]), 'Unit':int(l[1].split()[1])} for l in shared_units[:-2]]

    df_shared_grads =  pd.DataFrame([{'Layer':u['Layer'], 'Unit':u['Unit'], 'Grad':grad_data['mean'][u['Layer'], u['Unit']].item()} for u in shared_units])
    df_shared_grads['mode'] = 'shared'

                           
    with open(args.typical_units_file, 'r') as f:
        units = f.read()    
    typical_units = [l.split('\t') for l in units.split('\n')]
    header = typical_units[0]
    typical_units = typical_units[1:]
    typical_units = [{'Layer':int(l[0].split()[1]), 'Unit':int(l[1].split()[1])} for l in typical_units[:-2]]
    
    """
    with open(args.typical_units_file, 'r') as f:
        units = f.read()
    typical_units = [l.split('\t') for l in units.split('\n')]
    typical_units = [{'Layer':int(l[0]), 'Unit':int(l[1])} for l in typical_units[:-2] if (int(l[2])>=3 and l[3][0]=='n')]
    """
    all_units = flatten([[{'Layer':l, 'Unit':u} for u in range(hidden_dim)] for l in range(24)])

    df_typical_grads =  pd.DataFrame([{'Layer':u['Layer'], 'Unit':u['Unit'], 'Grad':grad_data['mean'][u['Layer'], u['Unit']].item()} for u in typical_units])
    print(df_typical_grads)
    df_typical_grads['mode'] = 'typical'
    df_random_grads = pd.DataFrame([{'Layer':u['Layer'], 'Unit':u['Unit'], 'Grad':grad_data['mean'][u['Layer'], u['Unit']].item()} for u in all_units])
    df_random_grads['mode'] = 'random'
    df_grads = pd.concat([df_random_grads, df_shared_grads, df_typical_grads])

    df_grouped = df_grads.groupby('mode')
    print(df_grouped.mean())
    
    baseline_q75 = df_grads[df_grads['mode']=='random']['Grad'].quantile(q=0.75)
    print(baseline_q75)
    typical_in_q75 = df_typical_grads[df_typical_grads['Grad']>=baseline_q75]
    print('Typical units in baseline q75: ', len(typical_in_q75))
    print('Typical units: ', len(df_typical_grads))

    print(typical_in_q75[['Layer', 'Unit']].to_csv(sep='\t', index=False))

    sns_plot = sns.scatterplot(data=df_grads, x='Layer', y='Grad', hue='mode', alpha=0.7)
    #plt.yscale('log')
    sns_plot.get_figure().savefig('grad.png')

    """
    for l in range(24):
        units_layer = [u for u in units if u['Layer']==l]        
        # gather grad for select units
        grads_mean = [grad_data['mean'][u['Layer'], u['Unit']] for u in units_layer]
        grads_mean = torch.tensor(grads.mean()).item()
        # grads_std = torch.tensor([grad_data['std'][u['Layer'], u['Unit']] for u in units_layer]).mean().item()
        # baseline
        all_grads_mean = grad_data['mean'][l].mean().item()
        # all_grads_std = grad_data['mean'].std().item()
        print(f'[Selected units-{l}] Mean grad {grads_mean}')
        print(f'[All units-{l}] Mean grad {all_grads_mean}')
    """
