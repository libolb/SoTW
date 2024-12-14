import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteria,LogitsProcessor, LogitsProcessorList
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby

MAX_TRIALS = 50
if torch.cuda.is_available():
    rng = torch.Generator("cuda")
else: 
    rng = torch.Generator("cpu")
hash_key = 15485863
PUNCTS = '\n!.?'
device = "cuda" if torch.cuda.is_available() else "cpu"


#def get_text_split(sentence,chunk_length=10):
#    words = sentence.split()
#    return [(" ".join(words[x: x + chunk_length])).strip() for x in range(0, len(words), chunk_length)]
#
#def my_sent_tokenize(text):
#
#    return get_text_split(text)
    
    
#def my_sent_tokenize(text):
#    all = []
#    for t in text.split("\n"):
#        for i in sent_tokenize(t):
#            all.append(i)
#    return all
    
def my_sent_tokenize(text):
    return sent_tokenize(text)


class SentenceEndCriteria(StoppingCriteria):
    """
    ONLY WORK WITH BATCH SIZE 1

    Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token). this is determined by nltk sent_tokenize.
    Only stop if ALL sentences in the batch is at least two sentences

    Args:
        tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0
        self.current_num_words = 0

    def update(self, current_text):
        self.current_num_sentences = len(my_sent_tokenize(current_text))
        self.current_num_words = len(current_text.split())
    

        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#        assert input_ids.size(0) == 1
#        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
#        print("-----------------------")
#        print(f"Decoded text so far: '{text}'")
#        print(len(my_sent_tokenize(text)),self.current_num_sentences + 1)
#        print("-----------------------")
#        #if len(sent_tokenize(text)) == self.current_num_sentences:
#        #    return True
#        return len(my_sent_tokenize(text)) > self.current_num_sentences + 1

        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # print("-----------------------")
        # print(f"Decoded text so far: '{text}'")
        # print(len(my_sent_tokenize(text)),self.current_num_sentences + 1)
        # print(len(text.split()),self.current_num_words)
        # print("-----------------------")
#        if len(my_sent_tokenize(text)) == self.current_num_sentences and len(my_sent_tokenize(text))!=1 and len(text.split())>self.current_num_words+1:
#            return True
        return len(my_sent_tokenize(text)) > self.current_num_sentences + 1


def discard_final_token_in_outputs(outputs):
    outputs.sequences = outputs.sequences[:, :-1]  # (bz, seqlen)
    return outputs


def extract_prompt_from_text(text, len_prompt):
    
    return my_sent_tokenize(text)[0]
#    tokens = text.split(' ')
#    tokens = tokens[:len_prompt]
#    new_text = ' '.join(tokens)
#    prompts = []
#    for p in PUNCTS:
#        if p==".":
#            idx=1
#            begin=0
#            while idx!=-1:
#                idx = new_text.find(p,begin)
#                if idx==-1:
#                    break
#                if len(new_text)>idx+2:
#                    if new_text[idx+1]!=" ":
#                        begin=idx+1
#                    else:
#                        break
#                else:
#                    break
#        else:
#            idx = new_text.find(p)
#
#        if idx != -1:
#            tokens = new_text[:idx + 1].split(" ")
#            # has to be greater than a minimum prompt
#            if len(tokens) > 3:
#                prompts.append(new_text[:idx + 1])
#    if len(prompts) == 0:
#        prompts.append(new_text + ".")
#    # select first (sub)sentence, deliminated by any of PUNCTS
#    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
#    return prompt



class ForbiddenCharsProcessor(LogitsProcessor):
    def __init__(self, tokenizer,forbidden_chars):
        self.forbidden_chars = forbidden_chars
        self.tokenizer = tokenizer
        

    def __call__(self, input_ids, scores):
        scores[:, self.forbidden_chars] = -float('inf')
        return scores



def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    forbidden_chars = []
    
    
#    if 'Llama-2-7b' in tokenizer.name_or_path:
#        forbidden_chars = [16156,18887,30010,16412,
#                        30086,5129,
#                        30015,9785,9533,14613,25035,1346,
#                        3178,8530,3995,8643,30024,9363,26622,6677,]
#    
#    if 'Baichuan-7B' in tokenizer.name_or_path:
#        forbidden_chars = [1185,1736,5209,8124,11462,15014,15152,15729,20119,22541,23274,24069,24911,27272,31164,
#                        889,2953,4403,5209,10397,27272,31162,
#                        7160,9384,9721,21388,31152,
#                        1473,31385,
#                        ]
#    
#    forbidden_chars.append(tokenizer.convert_tokens_to_ids("#"))
#    forbidden_chars.append(tokenizer.convert_tokens_to_ids("##"))
#    forbidden_chars_processor = ForbiddenCharsProcessor(tokenizer, forbidden_chars)
#    logits_processor = LogitsProcessorList([forbidden_chars_processor])
  
    outputs = model.generate(
            # input_ids,
            text_ids,
            generation_config=gen_config,
            #logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            
        )
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs.sequences
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

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

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]
