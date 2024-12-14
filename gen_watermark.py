'''
produce model generation
'''
import pprint
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from SQAE_model import SQAEModel
from sampling import reject_completion


import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True



setup_seed(1230)

PUNCTS = '.,!?'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--data_size', type=int,default=1)
    parser.add_argument('--out_dataset_path',
                        default="./results.json")
    
    parser.add_argument(
        '--data', type=str, help='dataset', default="./c4_500_len200_prompt.json")
    parser.add_argument(
        '--model', type=str, help='str model name to generate continuation. huggingface/openai',
        default='opt-1.3b')
    parser.add_argument(
        '--embedder', type=str, help='str model name to embed sentences', default='BAAI/bge-m3')
    parser.add_argument('--min_width', type=int, default=1,)

    parser.add_argument('--checkpoint', type=str, default="./SQAE/SQAE.pth")
    parser.add_argument('--checkpoint_dim', type=int, default=64)

    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--min_new_tokens', type=int, default=20)
    parser.add_argument('--lmbd', type=float, default=0.5,
                        help='ratio of valid sentences')


    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    # NOTE: currently, no batching
    is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get(
        'TRANSFORMERS_OFFLINE') == '1'


    import json
    with open(args.data, 'r', encoding="utf-8") as file:
        dataset = json.load(file)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True).to(args.device)
    model.eval()        
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad token to eos token")
    folder_name = os.path.join(args.data, args.embedder)
    # block \n
    bad_words_ids = tokenizer(
        "\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device='cuda').tolist()


    gen_config = GenerationConfig(
        return_dict_in_generate=True,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.05,
    )




    sqae_model = SQAEModel(model_path=args.embedder,
                              device=args.device, checkpoint=args.checkpoint,checkpoint_dim=args.checkpoint_dim)

    def deal_prompt(prompt):
        prompt = prompt.replace('\u201C', '"').replace('\u201D', '"')
        return prompt

    def text_to_generated_text(ex):
        prompt = ex['prompt']
        prompt = deal_prompt(prompt)
       # print(prompt)
        response = reject_completion(
            prompt,
            model, tokenizer, gen_config,
            sqae_model,
            lmbd=args.lmbd,
            device=args.device,
            min_width = args.min_width,
            dim= args.checkpoint_dim,
        )
        
        new_text = response[0][len(response[1]):].strip()
        return prompt,new_text


    output = []
    import json
    from tqdm import tqdm
    for i in tqdm(range(0,args.data_size)):
        prompt,text1 = text_to_generated_text(dataset[i])
        output.append({
            "ori_text":dataset[i]["text"],
            "prompt": prompt,
            'watermark_text': prompt+" "+text1,
        })


        result_file = args.out_dataset_path
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
