# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
from .base_connector import *


class CausalLM(Base_Connector):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        use_fast=True
        if 'opt' in args.model_name:
            use_fast=False # because of a bug with the OPT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=use_fast)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.tokenizer.add_special_tokens({'mask_token': "[MASK]"})
        self.tokenization = TOKENIZATION[self.model_name]
        self.MASK = self.tokenizer.mask_token

        if self.model_name == "transfo-xl-wt103":
            self.vocab = list(self.tokenizer.idx2sym)
            self._init_inverse_vocab()
        else:
            self._init_vocab()

        if self.model_name in LARGE_MODEL_LIST:
            # requires to create the ./offload_folder beforehand (already done in the container)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", offload_folder='/offload_folder', torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.model.eval()

        self.EOS = self.tokenizer.eos_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.UNK = self.tokenizer.unk_token
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_index = self.tokenizer.unk_token_id

        self.k = args.k

    def _cuda(self):
        self.model.cuda()
    
    def get_id(self, string):
        if "bpe" in self.tokenization:
            string = " " + string
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        # Replace the added [MASK] token with EOS token to make embeddings work
        sentences_list = [self.tokenizer.eos_token + item for sublist in sentences_list for item in sublist]
        input = self.tokenizer(sentences_list, padding=True, return_tensors="pt").input_ids
        masked_indices_list = np.argwhere(input.numpy() == self.tokenizer.mask_token_id)[:,1] - 1 
        masked_indices_list = [[i] for i in masked_indices_list]
        input = torch.where(input == self.tokenizer.mask_token_id, self.tokenizer.eos_token_id, input)
        with torch.no_grad():
            log_probs = self.model(input.to(self._model_device))
            if self.model_name == "transfo-xl-wt103":
                log_probs = log_probs.prediction_scores.cpu()
            else:
                log_probs = log_probs.logits.cpu()
        return log_probs, list(input.cpu().numpy()), masked_indices_list
    
    def get_input_tensors_batch_train(self, sentences_list, samples_list):
        if not sentences_list:
            return None
            
        # Compatibility with existing code
        sentences_list = [self.tokenizer.eos_token + item for sublist in sentences_list for item in sublist]
        input = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        masked_indices_list = np.argwhere(input.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1] # not like LAMA evaluation
        masked_indices_list = [[i] for i in masked_indices_list]

        #Optiprompt specific
        sample = pd.DataFrame(samples_list)
        sample["obj_label"] = " " + sample["obj_label"]
        sample["token_id"] = sample["obj_label"].apply(lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))[0])
        labels = sample["token_id"].values
        labels_tensor = torch.full_like(input.attention_mask, -100)

        predict_mask = input.input_ids.eq(self.tokenizer.mask_token_id)
        last_trigger_mask = torch.zeros_like(predict_mask)
        last_trigger_id =  (np.argwhere(predict_mask)[1] - 1)
        last_trigger_mask[torch.arange(len(last_trigger_id)),last_trigger_id] = True
        labels_tensor[predict_mask] = torch.tensor(labels)

        input["input_ids"] = torch.where(input.input_ids == self.tokenizer.mask_token_id, self.tokenizer.eos_token_id, input.input_ids)
        return input ,masked_indices_list, labels_tensor, labels.tolist(), last_trigger_mask

