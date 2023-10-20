from utils import load_data, load_optiprompt, free_optiprompt
from fc1_utils import import_fc1, filter_templates
import os
from models.causallm_connector import CausalLM
from argparse import Namespace
import torch
import random

LAMA_DIR = "./data/filtered_LAMA_opt"
MODEL_NAME='facebook/opt-1.3b'
RELATION='P1001' # you can also load all relations using a for loop

""" Initialise the model using the optiprompt wrapper (even though we are not using optiprompt).

Note that now we have to use the tokenizer associated to this wrapper.
(in practice it is almost equivalent to the one we used before)
"""
args = Namespace(
        model_name=MODEL_NAME,
        k=5, # number of top-k output prob you want do output
        fp16=False) # half precision. True if you use a larger model
model = CausalLM(args)
model.model.to('cpu') # force cpu (if needed)
tokenizer = model.tokenizer
vocab = model.tokenizer.get_vocab()

"""
Load LAMA/Rephrased templates from the pickle files.
"""
split_model_name = MODEL_NAME.split('/')[-1]
files = [f'fc1_att_data_{split_model_name}_t0_autoprompt-no-filter_fullvoc.pickle', # autoprompt
         f'fc1_att_data_{split_model_name}_t0_rephrase_fullvoc.pickle']             # human prompts (rephrased)  
datapath = '/Users/corentk/ALiEN/Prompting_prompts/source_code/OptiPrompt/data/fc1'

# import the data from the pickle files
mode = 'minimal' # unless you need activation data
template_data = import_fc1(datapath, files, mode=[mode])

# if you want to filter out the templates with a low accuracy
template_data = filter_templates(template_data, min_template_accuracy=10, only_best_template=False)[mode]

# 'template_data' is a pandas dataframe object. 
# here are the keys (i.e. column names)
print('Template_data contains these keys:', template_data.keys())
print('All templates:', template_data['template'].unique()) # do not forget unique to avoid duplicates
print('Human templates:', template_data[template_data['type']=='Human']['template'].unique()) # do not forget unique to avoid duplicates
print('Human templates for relation P1001:', template_data[template_data['relation']=='P1001'][template_data['type']=='Human']['template'].unique())
print('Average accuracy for human templates for relation P1001:', template_data[template_data['relation']=='P1001'][template_data['type']=='Human']['micro'].mean())
# etc

"""
Load the LAMA dataset (obj, subj, relation) that will be used to fill in the templates
"""
# randomly sample a human template for the current relation.
template_list = list(template_data[template_data['relation']==RELATION][template_data['type']=='Human']['template'].unique())
my_template = random.sample(template_list, 1)[0]
print("My template:", my_template)
# load the LAMA data for the current relation
# and combine them with the given templates
lama_test = os.path.join(LAMA_DIR, RELATION, "test.jsonl")
samples = load_data(lama_test, my_template, vocab_subset=vocab, mask_token='[MASK]')

for sample in samples: # iterating on LAMA's obj,subj pairs for the current relation
    input_sentence = sample['input_sentences']
    label = sample['obj_label']
    input_tkn = tokenizer(input_sentence, return_tensors="pt")
    output = model.model(**input_tkn.to(model.model.device))
    argmax_logit = torch.argmax(output.logits.squeeze()[-1])
    pred = tokenizer.decode([argmax_logit.item()])

    print(f'Input: {input_sentence}, Pred: {pred}, Label: {label}')  
    # print(f'Tokenized: {input_tkn}')