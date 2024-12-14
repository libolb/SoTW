
from SQAE_model import SQAEModel
from tqdm import trange
import argparse
from nltk.tokenize import sent_tokenize
from detection_utils import detect


def my_sent_tokenize(text):
    return sent_tokenize(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dataset_path', help='hf dataset containing text and para_text columns',
                        default="./results.json")

    parser.add_argument('--out_dataset_path', help='hf dataset containing text and para_text columns',
                        default="./results1.json")
    parser.add_argument('--embedder', type=str, help='sentence embedder',
                        default='BAAI/bge-m3')

    parser.add_argument('--lmbd', type=float, default=0.5, help='ratio of valid sentences')
    parser.add_argument('--data_size', type=int, default=1, help='ratio of valid sentences')
    parser.add_argument('--min_width', type=int, default=1,
                        help='dimension of the subspaces')
    parser.add_argument('--checkpoint', type=str, default="./SQAE/SQAE.pth",
                        help='')
    parser.add_argument('--checkpoint_dim', type=int, default=64)

    parser.add_argument('--hash_key', type=int, default=15485863,
                        help='dimension of the subspaces. default 3 for sstamp and 8 for ksstamp')

    args = parser.parse_args()
    return args

import re

def cut_sent(para):
    # 用于捕捉标点后直接是文字的情况，并进行分句
    para = re.sub('([。！？\?])([^”’，。！？《》\?])', r'\1split_sentence\2', para)
    # 处理引号内的标点符号后接文字的情况
    para = re.sub('([。！？\?][”’])([^，。！？《》\?\s])', r'\1split_sentence\2', para)
    # # 处理省略号或者多个点的情况，确保之后是文字才分句
    # para = re.sub('(\.{6}|\…{2})([^”’，。！？《》\?\s])', r'\1split_sentence\2', para)
    # 对最后一个标点符号后加换行符，以便正确分割
    para = re.sub('([。！？\?][”’])$', r'\1split_sentence', para)
    # 去除末尾的空白字符
    para = para.rstrip()
    # 分割字符串为句子列表
    # return para
    return para.split("split_sentence")


def split_japanese_sentences(text):
    # 使用正则表达式匹配句子结尾的标点符号
    sentences = re.split(r'(?<=[。！？])', text)
    # 去除空白句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences



if __name__ == '__main__':
    args = parse_args()
    import json
    with open(args.dataset_path, 'r', encoding="utf-8") as file:
       dataset = json.load(file)


    sqae_model = SQAEModel(model_path=args.embedder,
                              device=args.device, checkpoint=args.checkpoint,checkpoint_dim=args.checkpoint_dim)

    output = []
    for i in trange(0, args.data_size):
        ori_sents = my_sent_tokenize(dataset[i]['ori_text'])
        ori_z_score = detect(sents=ori_sents, model=sqae_model,
                                 lmbd=args.lmbd, dim=args.checkpoint_dim, min_width=args.min_width, hash_key=args.hash_key,
                                 )
        water_sents = my_sent_tokenize(dataset[i]["watermark_text"])
        water_z_score = detect(sents=water_sents, model=sqae_model,
                                   lmbd=args.lmbd, dim=args.checkpoint_dim, min_width=args.min_width, hash_key=args.hash_key,
                                   )

        output.append({
            "prompt": dataset[i]["prompt"],

            "ori_text": dataset[i]['ori_text'],
            'watermark_text': dataset[i]["watermark_text"],
            "z_score_ori": ori_z_score,
            "z_score_water": water_z_score,
        })

        with open(args.out_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
