import argparse
import os
import random
import logging
import torch

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, evaluate, get_relation_meta

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
            rephrase_dic[current_relation]['tmplt_id'] = [] # unique id to differentiate templates
            rephrase_dic[current_relation]['micro'] = [] # score / will be filled during evaluation
        elif l[0]=='#': # Description
            rephrase_dic[current_relation]['description'] = l[2:]
        elif l[0] in ['S', 'Q']:
            rephrase_dic[current_relation]['templates'].append(l[3:])
            rephrase_dic[current_relation]['format'].append(l[0])
            rephrase_dic[current_relation]['tmplt_id'].append(f'id{cpt_id}')
            rephrase_dic[current_relation]['micro'].append(-1) # will be filled during evaluation
            cpt_id += 1 # increase id count

    return rephrase_dic
        
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')
parser.add_argument('--output_dir', type=str, default='output', help='the output directory to store prediction results')
parser.add_argument('--common_vocab_filename', type=str, default='common_vocab_cased.txt', help='common vocabulary of models (used to filter triples)')
parser.add_argument('--prompt_file', type=str, default='prompts/marco_rephrasing/relation-paraphrases_v2.txt', help='file containing rephrased prompts')

parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA")
parser.add_argument('--eval_batch_size', type=int, default=32)

parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--output_predictions', default=True, help='whether to output top-k predictions')
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
parser.add_argument('--output_all_log_probs', action="store_true", help='whether to output all the log probabilities')

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


    # load model
    model = build_model_by_name(args)

    # load vocab
    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
    else:
        filter_indices = None
        index_list = None

    # Load rephrasing
    rephrase_dic = read_paraphrase(args.prompt_file)

    # iterate on relations
    for relation in os.listdir(args.test_data_dir):

        relation = relation.split(".")[0]
        print("-----RELATION {}".format(relation))

        # get rephrased templates for the given relation
        try:
            rephrase_list = rephrase_dic[relation]['templates']
        except KeyError:
            logger.info(f"Relation {relation} has no rephrase. Next")
            continue

        # iterate on rephrased templates
        for idx, template in enumerate(rephrase_list):
            
            logger.info(f'[{relation}] Template: {template}')

            # evaluate the rephrased template on the data test
            test_data = os.path.join(args.test_data_dir, relation + ".jsonl")
            eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
            eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
            micro, _ = evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=None)

            rephrase_dic[relation]['micro'][idx] = micro

        # write result into a tsv file
        output_dir = os.path.join(args.output_dir, os.path.basename(args.prompt_file).split(".")[0],args.model_name.replace("/","_"))
        os.makedirs(output_dir , exist_ok=True)

        with open(os.path.join(output_dir, relation+'.tsv'), 'w') as f:
            f.write('id\tformat\ttemplate\tmicro\n')
            for idx, template in enumerate(rephrase_dic[relation]['templates']):
                micro = rephrase_dic[relation]['micro'][idx]
                format = rephrase_dic[relation]['format'][idx]
                tempid = rephrase_dic[relation]['tmplt_id'][idx]
                line = f'{tempid}\t{format}\t{template}\t{micro}\n'
                f.write(line)