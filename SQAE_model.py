
import numpy as np
from typing import List, Tuple, Callable, Optional, Iterator
from FlagEmbedding import BGEM3FlagModel
global Device





import torch.nn as nn
import torch



fuc = "tanh"
n_layers = 8
MLP = True
enc_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # self.dropout = nn.Dropout(dropout_rate)
        if fuc=="tanh":
            self.relu = nn.Tanh()
        elif fuc=="relu":
            self.relu = nn.ReLU()
        else:
            raise ValueError("激活函数参数无效")

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc(out)
        out = self.norm(out)
        # out = out + x
        # return self.dropout(self.relu(out+x))
        return self.relu(out+x)

class Encoder(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=512, encoded_dim=256):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 输入层后加归一化
            # nn.Dropout(dropout_rate)
        ))
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, encoded_dim))

    def forward(self, x):
        # 保证数据类型和第一层一致
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers=4, encoded_dim=256, hidden_dim=512, output_dim=1024):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        # 将编码向量映射到隐藏层
        self.layers.append(nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 解码首层加归一化
            # nn.Dropout(dropout_rate)
        ))
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))
        # 最后重构到原始输入维度
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x


class SQAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings):
        super(SQAE, self).__init__()
        # Encoder part
        if MLP:
            self.encoder_fc =Encoder(n_layers, input_dim, latent_dim, enc_dim)
            self.encoder_fc.apply(self.init_weights)

            # self.encoder_fc = self.net = nn.Sequential(
            #     nn.Linear(input_dim, 2048),
            #     nn.Tanh(),
            #     nn.Linear(2048, latent_dim),
            # )

        else:
            self.encoder_fc = nn.Linear(input_dim, latent_dim)

        # VectorQuantizer part
        self.embedding_dim = enc_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, enc_dim)
        # self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.embeddings.weight.data.uniform_(-1. / self.num_embeddings, 1. / self.num_embeddings)


        # Decoder part
        if MLP:
            self.decoder_fc = Decoder(n_layers, enc_dim, latent_dim, input_dim)
            self.decoder_fc.apply(self.init_weights)
            # self.decoder_fc = nn.Linear(enc_dim, input_dim)
        else:
            self.decoder_fc = nn.Linear(latent_dim, input_dim)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            if fuc == "tanh":
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                # nn.init.zeros_(m.bias)
            elif fuc == "relu":
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.zeros_(m.bias)
            else:
                raise ValueError("激活函数参数无效")

    def encoder(self, x):
        z = self.encoder_fc(x)
        return z

    def quantizer(self, z):
        z_flattened = z.view(-1, self.embedding_dim)


        # # cos_sim
        # if distance=="cos":
        d = 1 - torch.matmul(z_flattened, self.embeddings.weight.t()) / (
                    z_flattened.norm(dim=1, keepdim=True) * self.embeddings.weight.norm(dim=1))
        # elif distance=="oushi":
        # # 欧氏距离
        #     d = torch.cdist(z_flattened, self.embeddings.weight, p=2)
        # else:
        #     raise ValueError("距离函数无效")


        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.num_embeddings).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings.weight).view(z.shape)


        return z_q, torch.argmin(d, dim=1).int()

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
        self.trans_model = SQAE(input_dim=1024, latent_dim=512, num_embeddings=checkpoint_dim)
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
