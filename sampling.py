import sampling_utils
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, StoppingCriteriaList
from SQAE_model import SQAEModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from sampling_utils import SentenceEndCriteria, device, gen_sent
from nltk.tokenize import sent_tokenize

import sys
# class FileWriter:
#     def __init__(self, filename):
#         self.file = open(filename, 'a',encoding="utf-8")
#
#     def write(self, message):
#         self.file.write(message)
#
#     def flush(self):
#         self.file.flush()
#
#     def close(self):
#         self.file.close()
#
# # ʵ���� FileWriter ���ض����׼���
# file_writer = FileWriter('output.txt')
# sys.stdout = file_writer


#def get_text_split(sentence,chunk_length=10):
#    words = sentence.split()
#    return [(" ".join(words[x: x + chunk_length])).strip() for x in range(0, len(words), chunk_length)]
#
#def my_sent_tokenize(text):
#
#    return get_text_split(text)

def my_sent_tokenize(text):
    return sent_tokenize(text)

# import logging
#
#
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG)
#
#
# file_handler = logging.FileHandler('app.log', mode='a')
# file_handler.setLevel(logging.DEBUG)
#
#
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
#
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)
#
#
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)





# rng = torch.Generator()
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)
MAX_TRIALS = sampling_utils.MAX_TRIALS


def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))

import random
def get_mask_from_seed(dim: int, accept_rate: float, seed: int):

    hash_key = 15485863

    test = [i for i in range(dim)]


    random.seed(hash_key*seed)


    num_to_select = int(len(test) * accept_rate)
    selected_elements = random.sample(test, num_to_select)
    
    while seed in selected_elements:
        selected_elements = random.sample(test, num_to_select)
    
    return selected_elements


def reject_close_generation(lsh_model, sents, margin, cutoff=None):
    embeds = lsh_model.get_embeddings(sents)
    embeds = torch.tensor(embeds, device='cuda')
    normals = torch.tensor(lsh_model.hasher.normals, device='cuda')
    if cutoff != None:
        normals = normals[:cutoff]

    # sims[i, j] is the cosine similarity between the ith generation and the jth normal vec
    sims = cosine_distance_matrix(embeds, normals)
    sims_abs = torch.abs(sims)
    # max_sim is the highest cosine similarity of each generation with any normal vec
    min_sims = sims_abs.min(dim=1).values
    select = []
    for i in range(len(min_sims)):
        # print(max_sims[i])
        min_sim = min_sims[i].item()
        if (abs(min_sim) >= margin):
            # print(min_sim)
            select.append(i)
    sents = np.array(sents)
    sents = sents[select]
    return list(sents), select

import re
from string import punctuation
from itertools import groupby
def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def deal_text(text):
    text = text.replace('  ', ' ')
    text = text.strip()
    text =  re.sub(r'\s+([,.\'?!])', r'\1', text)
    return clean_text(text)




def reject_completion(
        prompt: str,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        sqae_model: SQAEModel,
        dim: int,  # LSH args
        lmbd=1.0,
        device='cuda',
        min_width = 1,
        **kwargs):
    # print(f"prompt: {prompt}")
    stats = {}
    all_lsh_seed = []
    sent_end_criteria = SentenceEndCriteria(tokenizer)
    lsh_seed = sqae_model.get_hash([prompt.strip().replace('\n', '').replace('"', '')])[0]
    all_lsh_seed.append(lsh_seed)
    
    accept_mask = get_mask_from_seed(dim, lmbd, lsh_seed)

    
    
    
    prompt = deal_text(prompt)
    
    prompt = prompt.replace('\u201C', '"').replace('\u201D', '"').replace('\u2018', "'").replace('\u2019', "'")
    prompt = prompt.strip()
    
    
    text = prompt
    new_text = prompt
    old_text = text
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(prompt.split())
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0  # also include trials that maxed out MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]
    while True:
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        #if "Llama" in model.config._name_or_path:
        new_text, new_text_ids = gen_sent(model = model, 
            tokenizer = tokenizer, 
            text_ids = text_ids,
            gen_config = gen_config,
            stopping_criteria = stopping_criteria
        )


        
        flag=1

        
        new_text = new_text.replace('\u201C', '"').replace('\u201D', '"').replace('\u2018', "'").replace('\u2019', "'")
        new_text = new_text.strip()
        if len(new_text)>0 and new_text[-1] not in ".!?\"":
            new_text = new_text+"."  
            
        if len(my_sent_tokenize(new_text))>1:
            new_text = my_sent_tokenize(new_text)[0]



        total_trials += 1
        current_trials += 1

        
        
        new_text = deal_text(new_text)
        
        lsh_candidate = sqae_model.get_hash([new_text.strip().replace('\n', '').replace('"', '')])[0]
        # logger.info(str(lsh_candidate)+str(accept_mask)+","+str(current_trials)+" "+str(MAX_TRIALS)+" "+str(flag))




        if ((lsh_candidate in accept_mask) and flag) or current_trials >= MAX_TRIALS:
            if current_trials >= MAX_TRIALS:
                # logger.info(
                #     f'WARNING: desired semantic signature can\'t be sampled after max_trials {MAX_TRIALS}')
                print(f'CONTEXT: {text}', flush=True)
                print(
                    f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}', flush=True)
                maxedout_trials += 1
            debug_text_segments.append(
                (new_text, new_text_ids.size(1) - text_ids.size(1), lsh_candidate))
            current_trials = 0
            success_trials += 1

            # logger.info(new_text)
            old_text = text
            text = text.strip()+" "+new_text.strip()

            text = deal_text(text)

            text_ids = tokenizer.encode(text, return_tensors='pt').to(device)
            sent_end_criteria.update(text)
            

            lsh_seed = sqae_model.get_hash([my_sent_tokenize(text)[-1].strip().replace('\n', '').replace('"', '')])[0]
            all_lsh_seed.append(lsh_seed)

            lsh_seed = min(all_lsh_seed if len(all_lsh_seed)<min_width else all_lsh_seed[-min_width:])
            accept_mask = get_mask_from_seed(dim, lmbd, lsh_seed)


            if (len(text.split()) - prompt_length) >= gen_config.max_new_tokens-1:
                break
            
    return text, prompt, total_trials