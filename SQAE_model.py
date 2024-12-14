
import numpy as np
from typing import List, Tuple, Callable, Optional, Iterator
from FlagEmbedding import BGEM3FlagModel
global Device





import torch.nn as nn
import torch
class SQAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings):
        super(SQAE, self).__init__()
        # Encoder part
        self.encoder_fc = nn.Linear(input_dim, latent_dim)

        # VectorQuantizer part
        self.embedding_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, latent_dim)
        # self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.embeddings.weight.data.uniform_(-1 / self.embedding_dim, 1 / self.embedding_dim)


        # Decoder part
        self.decoder_fc = nn.Linear(latent_dim, input_dim)

    def encoder(self, x):
        z = self.encoder_fc(x)
        return z

    def quantizer(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (z_flattened.pow(2).sum(1, keepdim=True)
                     + self.embeddings.weight.pow(2).sum(1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).int()
        z_q = torch.index_select(self.embeddings.weight, 0, encoding_indices)

        return z_q, encoding_indices

    def decoder(self, z_q):
        x_recon = self.decoder_fc(z_q)
        return x_recon

    def forward(self, x):
        z = self.encoder(x)
        z_q, encoding_indices = self.quantizer(z)
        x_recon = self.decoder(z + (z_q - z).detach())
        return x_recon, z, z_q, encoding_indices



class SQAEModel:
    def __init__(self, device, checkpoint="",checkpoint_dim=64, model_path=None, **kwargs):
        self.device = device
        self.embedder = BGEM3FlagModel(model_path,device=self.device)
        self.trans_model = SQAE(input_dim=1024, latent_dim=1000, num_embeddings=checkpoint_dim)
        self.trans_model.load_state_dict(torch.load(checkpoint))
        self.trans_model.to(torch.device("cuda"))
        self.trans_model.eval()
        self.batch_size = 1


    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        all_embeddings = self.embedder.encode(sents, batch_size=self.batch_size, max_length=512)['dense_vecs']
        x_recon, z, z_q, re = self.trans_model(torch.tensor(all_embeddings, device='cuda', dtype=torch.float32))
        return re

    def get_hash(self, sents: Iterator[str]) -> Iterator[str]:
        embd = self.get_embeddings(sents)
        return embd.tolist()