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


parser = argparse.ArgumentParser()
parser.add_argument('--extract_gradient', action='store_true')
parser.add_argument('--analyse_units', action='store_true')
parser.add_argument('--units_file', type=str, default='/homedtcl/ckervadec/analyze/token_units/my_selection/unit_ids_token_unit_autoprompt-no-filter.vs.paraphrase_typical_autoprompt-no-filter_all_20230530-13_29_04.txt')
parser.add_argument('--grad_data_path', type=str, default='/homedtcl/ckervadec/analyze/opt-350m_autoprompt-no-filter_grad_analysis.pickle')
args = parser.parse_args()

OUTPUT_PATH = '../../analyze'
model_name = 'facebook/opt-350m'
model_string = model_name.split('/')[-1]

if args.extract_gradient:
    data_dir = "../../data/filtered_LAMA_opt"
    batch_size=128

    """ Initialise the model using the optiprompt wrapper.

    Note that now we have to use the tokenizer associated to this wrapper.
    (in practice it is almost equivalent to the one we used before)
    """
    model_args = Namespace(
            model_name=model_name,
            k=5,
            fp16=False)
    model = CausalLM(model_args)
    model.model.train()
    # model.enable_output_hidden_states()
    # model.model.to('cpu') # force cpu (if needed)
    tokenizer = model.tokenizer
    vocab = model.tokenizer.get_vocab()

    """
    Load templates
    """
    model_name = 'opt-350m'
    files = [f'fc1_att_data_{model_name}_t0_autoprompt-no-filter_fullvoc.pickle', 
            f'fc1_att_data_{model_name}_t0_optiprompt_fullvoc_fixetok.pickle',
            f'fc1_att_data_{model_name}_t0_rephrase_fullvoc.pickle']
    datapath = '../../data/fc1_data'

    # import the data from the pickle files
    mode = 'minimal'
    fc1_data = import_fc1(datapath, files, mode=[mode])

<<<<<<< HEAD
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
=======
    # if you want to filter the templates
    fc1_data = filter_templates(fc1_data, min_template_accuracy=10, only_best_template=False)[mode]
    df_templates = fc1_data[['type', 'template', 'relation']].drop_duplicates()
>>>>>>> bd8f529b6cbf74713aedbc431bdc363b3f30b48e

    """
    Define a hook that save the fc1 with its gradient
    """
    # function to extract grad
    def set_grad(var):
        def hook(grad):
            var.grad = grad
        return hook
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

        if not 'paraphrase' in prompt_type:
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

            # store the gradient distribution fot this relation
            fc1_grad_store[relation] = torch.zeros((len(layers),4096))

            for template in templates_list:

                # load the LAMA data for the current relation
                # and combine them with the given template
                samples = load_data(lama_data, template, vocab_subset=vocab, mask_token='[MASK]')
                samples_batches, sentences_batches = batchify(samples, batch_size)

                for i in tqdm(range(len(samples_batches)), desc=f'{template}'):
                    input, masked_indices_list, labels_tensor, mlm_label_ids, predict_mask = model.get_input_tensors_batch_train(sentences_batches[i], samples_batches[i])
                    with torch.enable_grad():
                        output = model.model(**input.to(model.model.device))
                        predict_logits = output.logits[predict_mask]
                        predict_argmax = torch.argmax(predict_logits, dim=-1)
                        argmax_logits = torch.gather(input=predict_logits, axis=-1, index=predict_argmax.unsqueeze(-1))
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

def load_pickles(f):
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    return data

if args.analyse_units:
    # load unit file
    with open(args.units_file, 'r') as f:
        units = f.read()
    units = [l.split('\t') for l in units.split('\n')]
    header = units[0]
    units = units[1:]
    units = [{'Layer':int(l[0].split()[1]), 'Unit':int(l[1].split()[1])} for l in units[:-2]]
    # load grad data
    grad_data = load_pickles(args.grad_data_path)
    # gather grad for select units
    grads_mean = torch.tensor([grad_data['mean'][u['Layer'], u['Unit']] for u in units]).mean().item()
    grads_std = torch.tensor([grad_data['std'][u['Layer'], u['Unit']] for u in units]).mean().item()
    # baseline
    all_grads_mean = grad_data['mean'].mean().item()
    all_grads_std = grad_data['mean'].std().item()
    print(f'[Selected units] Mean grad {grads_mean} ({grads_std})')
    print(f'[All units] Mean grad {all_grads_mean} ({all_grads_std})')