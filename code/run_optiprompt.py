import json
import argparse
import os
import random
import sys
import logging
from tqdm import tqdm

import torch
from torch import optim

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, evaluate, get_relation_meta

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

MAX_NUM_VECTORS = 10

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return ' [V%d]'%(vid)

def convert_random_to_dense(model):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        logger.info('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
        id_a = model.tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = model.tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.embeddings.weight[id_a] = model.embeddings.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    emb_len = model.embeddings.weight.shape[0]
    for i in range(5):
        token_id = random.randint(0, emb_len)
        token = model.tokenizer.decode(token_id)
        print(token_id, token)
        new_token_id += 1         
        template.append(get_new_token(new_token_id))
        assign_embedding(get_new_token(new_token_id), token)
    template.insert(0, "[X]")
    template.append("[Y]")
    return ' '.join(template)

def convert_manual_to_dense(manual_template, model):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        logger.info('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
        id_a = model.tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = model.tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.embeddings.weight[id_a] = model.embeddings.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    for word in manual_template.split():
        if word in ['[X]', '[Y]']:
            template.append(word)
        else:
            tokens = model.tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)

    return ' '.join(template)

def init_template(args, model, relation_name):
    if args.init_manual_template:
        relation = get_relation_meta(args, relation_name)
        template = convert_manual_to_dense(relation['template'], model)
    elif args.init_random:
        print("INIT RANDOM WORDS")
        template = convert_random_to_dense(model)
    else:
        print("INIT RANDOM VECS")
        template = '[X] ' + ' '.join(['[V%d]'%(i+1) for i in range(args.num_vectors)]) + ' [Y] .'
    return template

def prepare_for_dense_prompt(model):
    new_tokens = [get_new_token(i+1) for i in range(MAX_NUM_VECTORS)]
    model.tokenizer.add_tokens(new_tokens)
    ebd = model.model.resize_token_embeddings(len(model.tokenizer))
    logger.info('# vocab after adding new tokens: %d'%len(model.tokenizer))
    model.update_embeddings()

def save_optiprompt(output_dir, model, original_vocab_size):
    logger.info("Saving OptiPrompt's [V]s..")
    vs = model.embeddings.weight[original_vocab_size:].detach().cpu().numpy()
    with open(os.path.join(output_dir, 'prompt_vecs.npy'), 'wb') as f:
        np.save(f, vs)

def load_optiprompt(args, output_dir):
    model = build_model_by_name(args)
    original_vocab_size = len(list(model.tokenizer.get_vocab()))
    prepare_for_dense_prompt(model)
    
    logger.info("Loading OptiPrompt's [V]s..")
    with open(os.path.join(output_dir, 'prompt_vecs.npy'), 'rb') as f:
        vs = np.load(f)
    
    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        model.embeddings.weight[original_vocab_size:] = torch.Tensor(vs)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
    parser.add_argument('--model_dir', type=str, default=None, help='the model directory (if not using --model_name)')
    parser.add_argument('--output_dir', type=str, help='the output directory to store trained model and prediction results', default="output")
    parser.add_argument('--common_vocab_filename', type=str, default="", help='common vocabulary of models (used to filter triples)')
    parser.add_argument('--relation_profile', type=str, default="data/prompts/LAMA_relations.jsonl", help='meta infomation of 41 relations, containing the pre-defined templates')

    parser.add_argument('--train_data_dir', type=str, default="data/filtered_LAMA_opt")
    parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA_opt")

    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size per GPU')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_per_epoch', type=int, default=3)

    parser.add_argument('--do_shuffle', action='store_true')
    parser.add_argument('--do_eval', action='store_true', help="whether to run evaluation")
    parser.add_argument('--do_train', action='store_true', help="whether to run training process")
    parser.add_argument('--check_step', type=int, default=-1, help='how often to output training loss')

    parser.add_argument('--seed', type=int, default=6)

    parser.add_argument('--init_manual_template', action='store_true', help='whether to use manual template to initialize the dense vectors')
    parser.add_argument('--init_random', action='store_true', help='whether to use random words to initialize the dense vectors')
    parser.add_argument('--random_init', type=str, default='none', choices=['none', 'embedding', 'all'], help='none: use pre-trained model; embedding: random initialize the embedding layer of the model; all: random initialize the whole model')
    parser.add_argument('--num_vectors', type=int, default=5, help='how many dense vectors are used in OptiPrompt')

    parser.add_argument('--output_predictions', action='store_true', help='whether to output top-k predictions')
    parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')

    parser.add_argument('--device', type=str, default='cuda', help='Which computation device: cuda or mps')


    args = parser.parse_args()

    logger.info(args)

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
    original_vocab_size = len(list(model.tokenizer.get_vocab()))
    logger.info('Original vocab size: %d'%original_vocab_size)
    prepare_for_dense_prompt(model)

    if args.common_vocab_filename!='':
        vocab_subset = load_vocab(args.common_vocab_filename)   
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
    else:
        vocab_subset = list(model.inverse_vocab.keys())
        logger.info('Full %s vocab, size: %d'%(args.model_name, len(vocab_subset)))

    filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)

    if n_gpu > 1:
        model.model = torch.nn.DataParallel(model.model)


    for relation in os.listdir(args.train_data_dir):
        print("RELATION {}".format(relation))
        template = init_template(args, model, relation)
        logger.info('Template: %s'%template)
        output_dir = os.path.join(args.output_dir, args.model_name.replace("/","_"), relation)
        os.makedirs(output_dir, exist_ok=True)
        if args.do_train:
            logger.addHandler(logging.FileHandler(os.path.join(output_dir, "train.log"), 'w'))
        else:
            logger.addHandler(logging.FileHandler(os.path.join(output_dir, "eval.log"), 'w'))

        if args.do_train:
            # Prepare train/valid data
            train_data = os.path.join(args.train_data_dir, relation, "train.jsonl")
            dev_data = os.path.join(args.train_data_dir, relation, "dev.jsonl")
            train_samples = load_data(train_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
            train_samples_batches, train_sentences_batches = batchify(train_samples, args.train_batch_size * max(n_gpu, 1))
            logger.info('Train batches: %d'%len(train_samples_batches))
            valid_samples = load_data(dev_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
            valid_samples_batches, valid_sentences_batches = batchify(valid_samples, args.eval_batch_size * max(n_gpu, 1))
            logger.info('Valid batches: %d'%len(valid_samples_batches))

            # Valid set before train
            best_result, result_rel = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
            save_optiprompt(output_dir, model, original_vocab_size)

            # Add word embeddings to the optimizer
            optimizer = AdamW([{'params': model.embeddings.parameters()}], lr=args.learning_rate, correct_bias=False)
            t_total = len(train_samples_batches) * args.num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)

            # Train!!!
            global_step = 0
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            eval_step = len(train_samples_batches) // args.eval_per_epoch
            for _ in range(args.num_epoch):
                if args.do_shuffle:
                    logger.info('Shuffle train samples')
                    train_samples_batches, train_sentences_batches = random.shuffle(zip(train_samples_batches, train_sentences_batches))
                for i in tqdm(range(len(train_samples_batches))):
                    samples_b = train_samples_batches[i]
                    sentences_b = train_sentences_batches[i]

                    loss = model.run_batch(sentences_b, samples_b, training=True)
                    if n_gpu > 1:
                        loss = loss.mean()
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += len(samples_b)
                    nb_tr_steps += 1
                    global_step += 1

                    # set normal tokens' gradients to be zero
                    for p in model.embeddings.parameters():
                        # only update new tokens
                        p.grad[:original_vocab_size, :] = 0.0

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                    if args.check_step > 0 and ((nb_tr_steps + 1) % args.check_step == 0):
                        logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / nb_tr_examples))
                        sys.stdout.flush()
                        tr_loss = 0
                        nb_tr_examples = 0

                    if eval_step > 0 and (global_step + 1) % eval_step == 0:
                        # Eval during training
                        logger.info('Global step=%d, evaluating...'%(global_step))
                        precision, current_result = evaluate(model, valid_samples_batches, valid_sentences_batches, filter_indices, index_list)
                        if precision > best_result:
                            best_result = precision
                            result_per = current_result
                            logger.info('!!! Best valid (epoch=%d): %.2f' %
                                (_, best_result * 100))
                            save_optiprompt(output_dir, model, original_vocab_size)
            logger.info('Best Valid: %.2f'%(best_result*100))

        if args.do_eval:
            test_data = os.path.join(args.test_data_dir, relation + ".jsonl")
            model = load_optiprompt(args, output_dir)

            eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
            eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * max(n_gpu, 1))
            
            evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=output_dir if args.output_predictions else None)
