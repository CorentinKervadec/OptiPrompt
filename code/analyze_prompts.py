import sys
sys.path.insert(0, '/homedtcl/ckervadec/OptiPrompt')
import argparse
import os
import random
import logging
import torch

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(font_scale=0.4)

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, analyze, get_relation_meta

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

def init_template(prompt_file, relation):
    relation = get_relation_meta(prompt_file, relation)
    return relation['template']

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

def compute_freq_sensibility(res, treshold=0): # replace treshold by an interval
    layers_name = list(res.keys())
    # 1) Compute sensibilty: the neuron is activated if fc1 > treshold
    sensibility = {layer:torch.gt(res[layer],treshold) for layer in layers_name}
    # 2) Measure how frequently the neuron is activated
    freq_sensibility = {layer:sensibility[layer].count_nonzero(dim=0) for layer in layers_name}
    # torch.gt(h.view(batch_size, n_tokens, fc1_dim),0)
    return freq_sensibility

def count_activated_neurons(res, interval):
    layers_name = list(res.keys())
    count = {layer:[(torch.gt(res[layer],t).count_nonzero(dim=0)/res[layer].size(-1)*100).item() for t in interval] for layer in layers_name}
    return count

def find_triggered_neurons(res, treshold):
    layers_name = list(res.keys())
    triggered = {layer:torch.gt(res[layer],treshold) for layer in layers_name}
    return triggered


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')
parser.add_argument('--output_dir', type=str, default='output', help='the output directory to store prediction results')
parser.add_argument('--common_vocab_filename', type=str, default='data/vocab/common_vocab_opt_probing_prompts.txt', help='common vocabulary of models (used to filter triples)')

parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA_opt")
parser.add_argument('--eval_batch_size', type=int, default=32)

parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--output_predictions', default=True, help='whether to output top-k predictions')
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
parser.add_argument('--output_all_log_probs', action="store_true", help='whether to output all the log probabilities')

parser.add_argument('--prompt_files', type=str, default='prompts/LAMA_relations.jsonl,data/prompts/my-autoprompt-filter-causal-facebook-opt-350m_seed0.jsonl', help='prompt file separated by coma')
parser.add_argument('--relation', type=str, default='all', help='which relation to evaluate.')

if __name__ == "__main__":
    args = parser.parse_args()

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

    logger.info('Model: %s'%args.model_name)

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

    # store result
    all_fc1_act = {r:{p.split('/')[-1]:None for p in all_prompt_files} for r in relation_list}
    rl_2_nbfacts = {} 
    for relation in relation_list:
        relation = relation.split(".")[0]
        print("RELATION {}".format(relation))

        for prompt_file in all_prompt_files:

            # output_dir = os.path.join(args.output_dir, os.path.basename(args.prompt_file).split(".")[0],args.model_name.replace("/","_"))
            # os.makedirs(output_dir , exist_ok=True)
            output_dir = None

            template = init_template(prompt_file, relation)
            logger.info('Template: %s'%template)

            test_data = os.path.join(args.test_data_dir, relation, "test.jsonl")
            eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
            eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
            micro, result, fc1_act = analyze(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=output_dir if args.output_predictions else None)
            
            all_fc1_act[relation][prompt_file.split('/')[-1]]=fc1_act

            rl_2_nbfacts[relation]=torch.tensor(sum([len(batch) for batch in eval_samples_batches]))

    # Now compare prompt (we reduce on the tkn dimension, taking into account the padding mask)
    # Predict token only
    # take fc1_act for the predicted token accross all batch
    predict_fc1_act = {rl:{prmpt:select_pred_masked_act(res) for prmpt,res in values.items()} for rl,values in all_fc1_act.items()}
    # Measure fc1_1 sensibility accross fact (we obtain one vector per relation)
    SENSIBILITY_TRESHOLD=0
    predict_fc1_act_sens_freq = {rl:{prmpt:compute_freq_sensibility(res, SENSIBILITY_TRESHOLD) for prmpt,res in values.items()} for rl,values in predict_fc1_act.items()}
    
    # -----------------------------
    # Control check:
    #  plot the number of activated neurons against the frequency treshold
    predict_fc1_act_count = {rl:{prmpt:count_activated_neurons(res,torch.logspace(start=0,end=torch.log2(rl_2_nbfacts[rl]), steps=10, base=2)) for prmpt,res in values.items()} for rl,values in predict_fc1_act_sens_freq.items()}
    # To dataframe
    df = pd.concat([pd.concat([pd.concat([pd.DataFrame(count, columns=['Ratio']).assign(Layer = layer) for layer,count in res.items()]).assign(Prompt = prmpt[:20]) for prmpt,res in values.items()]).assign(Relation = rl) for rl,values in predict_fc1_act_count.items()])
    df = df.reset_index().rename(columns = {'index':'Treshold'})
    # Seaborn plot
    fig = plt.figure()
    ax = sns.relplot(kind="line", x="Treshold", y="Ratio",
             hue="Layer", col="Prompt", row="Relation", data=df,facet_kws=dict(sharex=False),)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.savefig('count_activate.png', dpi=fig.dpi)

    # Correlation of activation accross prompts
    # fig = plt.figure()
    # fg = sns.FacetGrid(df.sort_values('Layer'), col='Treshold', row='Relation')
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)#, row_cluster=False)
    # fg.map_dataframe(draw_heatmap, 'Prompt', 'Layer', 'Ratio')
    # # sns.clustermap(df[df['Treshold']==5][df['Relation']=='P101'].pivot("Layer", "Prompt", "Ratio"), row_cluster=False)
    # plt.savefig('act_cluster_map.png', dpi=fig.dpi)
    # -----------------------------

    # Measure activation overlap between prompts
    TRIGGER_TRESHOLD_FREQ_RATE=0.2
    TRIGGER_TRESHOLD_FREQ=lambda x: rl_2_nbfacts[x]*TRIGGER_TRESHOLD_FREQ_RATE
    predict_fc1_triggered = {rl:{prmpt:find_triggered_neurons(res,TRIGGER_TRESHOLD_FREQ(rl)) for prmpt,res in values.items()} for rl,values in predict_fc1_act_sens_freq.items()}
    predict_triggered_overlap = {rl:{prmpt_A:{prmpt_B:{l:((res_B[l]==res_A[l]).sum()/res_B[l].size(-1)).item() for l in res_B} for prmpt_B, res_B in values.items()} for prmpt_A,res_A in values.items()} for rl,values in predict_fc1_triggered.items()}
    df = pd.concat([pd.concat([pd.concat([pd.concat([pd.DataFrame([overlap], columns=['Overlap']).assign(Layer = layer) for layer,overlap in res_B.items()]).assign(Prompt_B = prmpt_B[:20]) for prmpt_B,res_B in res_A.items()]).assign(Prompt_A = prmpt_A[:20]) for prmpt_A,res_A in values.items()]).assign(Relation = rl) for rl,values in predict_triggered_overlap.items()])
    # Adapted when there is many different prompts. But not convenient for dealing with layers
    fig = plt.figure()
    fg = sns.FacetGrid(df, col='Layer', row='Relation')
    fg.map_dataframe(draw_heatmap, 'Prompt_A', 'Prompt_B', 'Overlap')
    plt.suptitle(f"Sensibility: {SENSIBILITY_TRESHOLD} / Trigger treshold rate: {TRIGGER_TRESHOLD_FREQ_RATE}")
    plt.savefig('fc1_overlap.png', dpi=fig.dpi)
    # Adapted to compare few prompts accross relation and layers
    fg = sns.FacetGrid(df, col='Prompt_A', row='Prompt_B')
    fg.map_dataframe(draw_heatmap, 'Layer', 'Relation', 'Overlap', vmin=0.7, vmax=1.0)
    plt.suptitle(f"Sensibility: {SENSIBILITY_TRESHOLD} / Trigger treshold rate: {TRIGGER_TRESHOLD_FREQ_RATE}")
    plt.tight_layout()
    plt.savefig('fc1_overlap_2.png', dpi=fig.dpi)
