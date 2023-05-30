import argparse
from datasets import load_dataset
from models import build_model_by_name
import torch
from tqdm import tqdm
import numpy as np
import random
import itertools
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--window_size', type=int, default=15)
parser.add_argument('--window_stride', type=int, default=15)
parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
parser.add_argument('--output_dir', type=str, default='./unit-token-analyze', help='the output directory to store prediction results')
parser.add_argument('--compute_global_units', default=False, help='whether to compute global units stats')
parser.add_argument('--compute_token_units_input', default=True, help='whether to compute tokens units stats')
parser.add_argument('--compute_token_units_output', default=True, help='whether to compute tokens units stats')
parser.add_argument('--fp16', action='store_true', help='use half precision')


def slice_tokenized_datum(tokenized_datum, window_size, window_stride):
    datum_length = len(tokenized_datum)
    start_offset = random.randint(0,args.window_size-1) # add a random start offset to increase the diversity (and avoid havint too much bos)
    slices = [tokenized_datum[t:t+window_size] for t in range(start_offset,datum_length-window_size,window_stride)]
    return slices

def batchify(datalist, batch_size):
    batches = [datalist[i:i+args.batch_size] for i in range(0,len(datalist),batch_size)]
    # to pytorch
    batches = [torch.tensor(batch) for batch in batches]
    attention_masks = [torch.ones_like(batch) for batch in batches]
    return zip(batches, attention_masks), len(batches)

def cumulative_average(new_item, new_count, old_count, old_average, device='cpu'):
    new_item = new_item.to(device)
    new_count = new_count.to(device)
    old_count = old_count.to(device)
    old_average = old_average.to(device)
    return (new_item + (old_count) * old_average) / (new_count)

def update_token_unit(unit_tokens, fc1_act, layer, unique_id, token_ids, tokens_count, old_token_count, device):
    save_device = unit_tokens[layer].device
    # create an index mask, to only process tokens in the batch
    expand_unique_id = unique_id.unsqueeze(0).expand(token_ids.size(0), -1)
    index_mask = (expand_unique_id == token_ids.unsqueeze(-1)).t().half()
    # compute the unit-token activations for the batch, on the device
    batch_unit_token_cum = torch.matmul(index_mask.to(device), fc1_act[layer].to(device))
    # update the cumuluative average
    unit_tokens[layer][unique_id] = batch_unit_token_cum.to(save_device)
    
    # cumulative_average(
    #     new_item    = batch_unit_token_cum,
    #     new_count   = tokens_count[unique_id].unsqueeze(-1),
    #     old_count   = old_token_count[unique_id].unsqueeze(-1),
    #     old_average = unit_tokens[l][unique_id.to(save_device)],
    #     device      = device,
    # ).to(save_device)
    return unit_tokens

if __name__ == "__main__":
    args = parser.parse_args()

    if args.seed == -1:
        random_seed = random.randint(0, 99999)
    else:
        random_seed = args.seed
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # init device
    device=torch.device(args.device)
    if args.device == 'cuda':
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            print('No GPU found! exit!')
        print('# GPUs: %d'%n_gpu)

    elif args.device == 'mps':
        n_gpu = 1
    else:
        print('# Running on CPU')
        n_gpu = 0

    # init data
    wikidata = load_dataset("wikipedia", "20220301.en", split='train')
    # random sample
    if args.n_samples > 0:
        random_idx = random.sample(range(len(wikidata)), args.n_samples)
        wikidata = wikidata.select(random_idx)


    # init model
    model = build_model_by_name(args)
    model.set_analyse_mode(attention=False)
    model.model.to(device)
    n_layers = len(model.model.model.decoder.layers)
    n_units = model.model.model.decoder.layers[0].fc1.out_features

    
    # init token/unit matrix
    vocab_list = model.vocab
    n_vocab = len(vocab_list)
    unit_list = list(range(n_units))
    tokens_count = {'input':torch.zeros(size=(n_vocab,)).int().to(device), 'output':torch.zeros(size=(n_vocab,)).int().to(device)}
    """
    * unit_tokens_accum_avg: For each token, provides the average unit activation
    * unit_global_accum_avg: Provides the average unit activation
    """
    unit_tokens_accum = {}
    if args.compute_token_units_input:
        unit_tokens_accum['input'] = [torch.full(size=(n_vocab, n_units), fill_value=0.0, dtype=torch.float16) for l in range(n_layers)]
    if args.compute_token_units_output:
        unit_tokens_accum['output'] = [torch.full(size=(n_vocab, n_units), fill_value=0.0, dtype=torch.float16) for l in range(n_layers)]
    if args.compute_global_units:
        unit_global_accum = [torch.full(size=(n_units,), fill_value=0.0) for l in range(n_layers)]

    # process data: tokenize/slice/batch
    wikidata = wikidata.map(lambda s: model.tokenizer(s['text']), num_proc=4) # tokenize
    wikidata_sliced = [slice_tokenized_datum(datum['input_ids'], args.window_size, args.window_stride) for datum in wikidata] # slice
    wikidata_sliced = list(itertools.chain.from_iterable(wikidata_sliced)) # flatten
    wikidata_sliced_batched, n_batch = batchify(wikidata_sliced, args.batch_size) # batch. this is an iterator

    # iterate on the dataset
    for input_ids, attention_mask in tqdm(wikidata_sliced_batched, total=n_batch):
        d_batch, d_sent = input_ids.shape
        # forward pass
        with torch.no_grad():
            output = model.model(input_ids.to(device), attention_mask.to(device))
        # exctract fc1 activations
        fc1_act = model.get_fc1_act()
        fc1_act = list(fc1_act.values())
        # relu
        fc1_act = [torch.relu(f.to(device)) for f in fc1_act]
        # accumulate input and output token ids
        tokens_ids = {
            'input': input_ids.flatten().to(device),
            'output': torch.argmax(output.logits.detach(), dim=-1).flatten().to(device)
        }
        # count unique tokens
        unique_id = {}
        old_token_count = {}

        for mode in unit_tokens_accum.keys():
            uni_res = torch.unique(tokens_ids[mode], return_counts=True)
            unique_id[mode] = uni_res[0]
            count_id = uni_res[1]
            old_token_count[mode] = tokens_count[mode].clone().detach() #tokens_count.sum()
            tokens_count[mode][unique_id[mode]] += count_id

        # for each layer accumulate the unit-token association
        for l in range(n_layers):
            # per token stats
            for mode in unit_tokens_accum.keys():
                unit_tokens_accum[mode] = update_token_unit(
                    unit_tokens=unit_tokens_accum[mode],
                    fc1_act=fc1_act,
                    layer=l,
                    unique_id=unique_id[mode],
                    token_ids=tokens_ids[mode],
                    tokens_count=tokens_count[mode],
                    old_token_count=old_token_count[mode],
                    device=device,)
            # global stats
            if args.compute_global_units: # deprecated
                unit_global_accum[l] = cumulative_average(
                    new_item    = fc1_act[l].sum(0),
                    new_count   = tokens_count.sum(),
                    old_count   = old_token_count.sum(),
                    old_average = unit_global_accum[l],
                    device      = device,
                ).cpu()

# to cpu
tokens_count = {k:v.cpu() for k,v  in tokens_count.items()}

# Save with pickle
print('Saving stats...')
exp_name = f'{args.model_name.split("/")[-1]}-N{args.n_samples}-{random_seed}'
save_dir = os.path.join(args.output_dir,f'unit-token-wiki.{random_seed}')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.compute_global_units:
    with open(os.path.join(save_dir,f'unit_global_accum-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(unit_global_accum, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_dir, f'tokens-count-{exp_name}.pickle'), 'wb') as handle:
    pickle.dump(tokens_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

if args.compute_token_units_input or args.compute_token_units_output:
    with open(os.path.join(save_dir,f'unit_tokens_accum-{exp_name}.pickle'), 'wb') as handle:
        pickle.dump(unit_tokens_accum, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Done!')