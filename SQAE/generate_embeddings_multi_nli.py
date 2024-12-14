import numpy as np
import json
import torch
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
from FlagEmbedding import BGEM3FlagModel
import random


def cosine_similarity(x, y):
    # dot_product = torch.sum(x * y, dim=-1)
    # norm_x = torch.norm(x, p=2, dim=-1)
    # norm_y = torch.norm(y, p=2, dim=-1)
    # return dot_product / (norm_x * norm_y)
    return torch.cosine_similarity(x, y, dim=-1)
class SentenceEmbeddings:

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = BGEM3FlagModel(model_path,device=self.device)

    def get_embedding(self, sents) :
        all_embeddings = self.embedder.encode(sents, batch_size=1,max_length=512)['dense_vecs']
        all_embeddings = torch.tensor(all_embeddings, device=self.device,dtype=torch.float32)
        return all_embeddings

    def generate_embeddings(self, input_path, output_path, generate_size=10000):
        """Generate embeddings for all sentences in the input file."""
        all_embeddings = []

        en_embeddings=[]
        ch_embeddings=[]


        with open(input_path, 'r', encoding="utf-8") as file:
            lines = json.load(file)
        random.shuffle(lines)
        pbar = tqdm(total=generate_size, desc="Embeddings generated")
        print(len(lines))
        for line in lines:

            ch = self.get_embedding(line["Sentence1"])
            en = self.get_embedding(line["Sentence2"])

            all_embeddings.append(en.cpu().numpy())
            all_embeddings.append(ch.cpu().numpy())

            en_embeddings.append(en.cpu().numpy())
            ch_embeddings.append(ch.cpu().numpy())

            pbar.update(2)
            if len(all_embeddings) >= generate_size:
                break

        pbar.close()

        all_embeddings = np.vstack(all_embeddings)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, all_embeddings, delimiter=" ")

        en_embeddings = np.vstack(en_embeddings)
        np.savetxt("./embedding/multi_nli/train_embeddings_en.txt", en_embeddings, delimiter=" ")
        ch_embeddings = np.vstack(ch_embeddings)
        np.savetxt("./embedding/multi_nli/train_embeddings_ch.txt", ch_embeddings, delimiter=" ")

def main():

    size = 10000000
    parser = argparse.ArgumentParser(description='Generate embeddings for sentences.')
    parser.add_argument('--input_path', type=str, required=False, help='Input file path',default="./multi_nli.json")
    parser.add_argument('--output_path', type=str, required=False, help='Output file path',
                        default="./embedding/multi_nli/train_embeddings_{}.txt".format(size))
    parser.add_argument('--model_path', type=str, required=False, help='Path of the embedding model',default=r'BAAI\bge-m3')
    parser.add_argument('--size', type=int, required=False, default=size, help='Size of the data to generate embeddings for')
    args = parser.parse_args()

    sentence_embeddings = SentenceEmbeddings(args.model_path)
    sentence_embeddings.generate_embeddings(args.input_path, args.output_path, args.size)

if __name__ == '__main__':
    main()
