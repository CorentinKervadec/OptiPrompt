import argparse
import os
import random
import logging
import torch

from transformers import AutoTokenizer

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, evaluate, get_relation_meta

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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='facebook/opt-350m', help='the huggingface model name')
parser.add_argument('--output_dir', type=str, default='output', help='the output directory to store prediction results')
parser.add_argument('--common_vocab_filename', type=str, default='data/vocab/common_vocab_opt_probing_prompts.txt', help='common vocabulary of models (used to filter triples)')
parser.add_argument('--prompt_file', type=str, default='prompts/LAMA_relations.jsonl', help='prompt file containing 41 relations')

parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA_opt")
parser.add_argument('--eval_batch_size', type=int, default=32)

parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--output_predictions', default=True, help='whether to output top-k predictions')
parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')
parser.add_argument('--device', type=str, default='mps', help='Which computation device: cuda or mps')
parser.add_argument('--output_all_log_probs', action="store_true", help='whether to output all the log probabilities')
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

    if args.common_vocab_filename!='none':
        vocab_subset = load_vocab(args.common_vocab_filename)   
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    else:
        vocab_subset = list(model.inverse_vocab.keys())
        logger.info('Full %s vocab, size: %d'%(args.model_name, len(vocab_subset)))

    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)

    if args.output_all_log_probs:
        model.k = len(vocab_subset)

    if args.relation=='all':
        relation_list = RELATIONS_TEST
    else:
        relation_list=[r for r in args.relation.split(',')]

    for relation in relation_list:
        relation = relation.split(".")[0]
        print("RELATION {}".format(relation))

        output_dir = os.path.join(args.output_dir, os.path.basename(args.prompt_file).split(".")[0],args.model_name.replace("/","_"))
        os.makedirs(output_dir , exist_ok=True)

        template = init_template(args.prompt_file, relation)
        logger.info('Template: %s'%template)

        test_data = os.path.join(args.test_data_dir, relation, "test.jsonl")
        eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
        evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=output_dir if args.output_predictions else None)
