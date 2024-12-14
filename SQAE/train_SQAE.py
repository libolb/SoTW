import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from torch.optim import lr_scheduler
from tqdm import tqdm
# from tensorboardX import SummaryWriter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# 设置随机数种子



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



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
        self.embeddings.weight.data.uniform_(-1. / self.num_embeddings, 1. / self.num_embeddings)


        # Decoder part
        self.decoder_fc = nn.Linear(latent_dim, input_dim)

    def encoder(self, x):
        z = self.encoder_fc(x)
        return z

    def quantizer(self, z):
        z_flattened = z.view(-1, self.embedding_dim)

        # cos_sim
        d = 1 - torch.matmul(z_flattened, self.embeddings.weight.t()) / (
                    z_flattened.norm(dim=1, keepdim=True) * self.embeddings.weight.norm(dim=1))


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

def cos_distance(a,b):
    return (1.-torch.nn.functional.cosine_similarity(a,b))


def loss_function(x1, x_recon1, z1, z_q1, x2, x_recon2, z2, z_q2):

    recon_loss1 = torch.mean((x1 - x_recon1) ** 2)
    vq_loss1 = torch.mean((z1.detach() - z_q1)**2) + 0.25*torch.mean((z1 - z_q1.detach())**2)
    recon_loss2 = torch.mean((x2 - x_recon2) ** 2)
    vq_loss2 = torch.mean((z2.detach() - z_q2)**2) + 0.25*torch.mean((z2 - z_q2.detach())**2)



    recon_loss = (recon_loss1+recon_loss2)/2
    vq_loss = (vq_loss1+vq_loss2)/2



    margin = alpha
    cont_code_dist = cos_distance(x1, x2)
    discrete_code_dist = cos_distance(z1, z2)
    mask_a = (cont_code_dist < alpha).float()
    mask_b = (cont_code_dist >= alpha).float()
    # positive_loss = discrete_code_dist ** 2
    # negative_loss = torch.relu(margin - discrete_code_dist) ** 2
    positive_loss = torch.abs(discrete_code_dist)
    negative_loss = torch.abs(cont_code_dist - discrete_code_dist)

    contrastive_lossp = (mask_a * positive_loss).sum()/ torch.max(mask_a.sum(), torch.tensor(1.0))
    contrastive_lossn = (mask_b * negative_loss).sum()/ torch.max(mask_b.sum(), torch.tensor(1.0))
    contrastive_loss = (contrastive_lossp + contrastive_lossn) / 2

    return lambda1 *(recon_loss + vq_loss) + (1-lambda1)*contrastive_loss,recon_loss
    # return recon_loss + vq_loss,recon_loss


class VectorDataset(Dataset):
    def __init__(self, vectors):
        self.vectors = vectors
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return self.vectors[idx]
def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                pass
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        torch.save(model.state_dict(), args.output_model)
        self.val_loss_min = val_loss

if __name__ == '__main__':


    en_embedding_data_1 = np.loadtxt(
        "./embedding/multi_nli/train_embeddings_en.txt")
    ch_embedding_data_1 = np.loadtxt(
        "./embedding/multi_nli/train_embeddings_ch.txt")
    # en_embedding_data_1 = np.loadtxt(
    #     "./embedding/sts/train_embeddings_en.txt")
    # ch_embedding_data_1 = np.loadtxt(
    #     "./embedding/sts/train_embeddings_ch.txt")
    en_embedding_data = en_embedding_data_1
    ch_embedding_data = ch_embedding_data_1

    input_dim = 1024
    hidden_dim = 1024*4


    flag =1

    import json
    re = []
    for lr in [1e-05]:
        for num_embeddings in [64]:
            for lambda1 in [0.5] :
                for flag in [1]:
                    for batch_size in [64]:
                        for weight_decay in [1e-6]:

                            # torch.cuda.empty_cache()
                            epochs1 = 1000
                            latent_dim = 1000
                            scale=21
                            lambda2=1
                            # lambda1="912multinli"
                            alpha=0.3
                            setup_seed(1230)
                            # folder_path = 'runs_0909_VQ-VAE1/lr_{}_num_embedding_{}_lam_{}'.format(lr,num_embeddings,lambda1)
                            #
                            # writer = SummaryWriter(folder_path)



                            parser = argparse.ArgumentParser(description="Detect watermark in texts")
                            parser.add_argument("--output_model", type=str, default="./SQAE.pth")
                            parser.add_argument("--epochs", type=int, default=epochs1)

                            parser.add_argument("--lr", type=float, default=lr)
                            parser.add_argument("--input_dim", type=int, default=input_dim)



                            args = parser.parse_args()
                            model = SQAE(args.input_dim, latent_dim,num_embeddings)
                            model.to(device)


                            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
                            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
                            epochs = args.epochs

                            patience = 100
                            early_stopping = EarlyStopping(patience=patience, verbose=True)




                            en_data = torch.tensor(en_embedding_data, device='cuda', dtype=torch.float32)
                            en_dataset = VectorDataset(en_data)
                            en_dataloader = DataLoader(en_dataset, batch_size=batch_size, shuffle=True,
                                                       generator=torch.Generator().manual_seed(2024))
                            ch_data = torch.tensor(ch_embedding_data, device='cuda', dtype=torch.float32)
                            ch_dataset = VectorDataset(ch_data)
                            ch_dataloader = DataLoader(ch_dataset, batch_size=batch_size, shuffle=True,
                                                       generator=torch.Generator().manual_seed(2024))



                            if flag:
                                vali_en_embedding_data = np.loadtxt(
                                    "./embedding/sts/train_embeddings_en.txt")
                                vali_ch_embedding_data = np.loadtxt(
                                    "./embedding/sts/train_embeddings_ch.txt")
                                vali_en_data = torch.tensor(vali_en_embedding_data, device='cuda', dtype=torch.float32)
                                vali_en_dataset = VectorDataset(vali_en_data)
                                vali_en_dataloader = DataLoader(vali_en_dataset, batch_size=5000, shuffle=False,
                                                                generator=torch.Generator().manual_seed(2024))
                                vali_ch_data = torch.tensor(vali_ch_embedding_data, device='cuda', dtype=torch.float32)
                                vali_ch_dataset = VectorDataset(vali_ch_data)
                                vali_ch_dataloader = DataLoader(vali_ch_dataset, batch_size=5000, shuffle=False,
                                                                generator=torch.Generator().manual_seed(2024))
                            else:

                                vali_en_dataloader = DataLoader(en_dataset, batch_size=5000, shuffle=False,
                                                                generator=torch.Generator().manual_seed(2024))
                                vali_ch_dataloader = DataLoader(ch_dataset, batch_size=5000, shuffle=False,
                                                                generator=torch.Generator().manual_seed(2024))


                            for epoch in tqdm(range(epochs)):

                                max_app_num=50000000


                                train_all_num = torch.tensor([], device=device)


                                en_batch_iterator = iter(en_dataloader)
                                ch_batch_iterator = iter(ch_dataloader)
                                cnt = 0
                                model.train()

                                all_corret = 0
                                a_empty_tensor1 = torch.tensor([], device=device)
                                b_empty_tensor1 = torch.tensor([], device=device)
                                for _ in range(len(en_dataloader) - 1):

                                    cont_code1 = next(en_batch_iterator).to(device)
                                    cont_code2 = next(ch_batch_iterator).to(device)
                                    x_recon1, z1, z_q1, encoding_indices1 = model(cont_code1)
                                    x_recon2, z2, z_q2, encoding_indices2 = model(cont_code2)

                                    loss, a = \
                                        loss_function(cont_code1,x_recon1, z1, z_q1, cont_code2,x_recon2, z2, z_q2)

                                    a_empty_tensor1 = torch.cat((a_empty_tensor1, encoding_indices1), dim=0)
                                    b_empty_tensor1 = torch.cat((b_empty_tensor1, encoding_indices2), dim=0)

                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                                    cnt += 1

                                model.eval()
                                val_losses = []
                                vali_en_batch_iterator = iter(vali_en_dataloader)
                                vali_ch_batch_iterator = iter(vali_ch_dataloader)
                                with torch.no_grad():
                                    a_empty_tensor = torch.tensor([], device=device)
                                    b_empty_tensor = torch.tensor([], device=device)

                                    for _ in range(len(vali_en_dataloader)):
                                        en_input_a = next(vali_en_batch_iterator).to(device)
                                        ch_input_b = next(vali_ch_batch_iterator).to(device)
                                        x_recon, z, z_q, encoding_indices_en = model(en_input_a)
                                        x_recon, z, z_q, encoding_indices_ch = model(ch_input_b)

                                        a_empty_tensor = torch.cat((a_empty_tensor, encoding_indices_ch), dim=0)
                                        b_empty_tensor = torch.cat((b_empty_tensor, encoding_indices_en), dim=0)

                                    unique_values, counts = torch.unique(torch.cat([a_empty_tensor, b_empty_tensor]),
                                                                         dim=0, return_counts=True)

                                    inconsistencie = torch.abs(a_empty_tensor - b_empty_tensor)
                                    loss_a = torch.count_nonzero(inconsistencie)
                                    re_aa = torch.abs(
                                        torch.mean(2 * torch.cat([a_empty_tensor, b_empty_tensor], dim=0) - 1,
                                                   dim=0)).sum()

                                    unique_values1, counts1 = torch.unique(torch.cat([a_empty_tensor1, b_empty_tensor1]),
                                                                         dim=0, return_counts=True)

                                    if counts1.shape[0]!=num_embeddings:
                                        print(loss_a.item())
                                        val_losses.append(50000000)
                                    else:
                                        if flag:
                                            if max(counts).item()<max_app_num:
                                                max_app_num = max(counts).item()
                                        else:
                                            if max(counts1).item()<max_app_num:
                                                max_app_num = max(counts1).item()
                                        val_losses.append(max_app_num)
                                    print("---------------------------------------")
                                    print(re_aa.item(), loss_a.item())
                                    print(counts1, counts1.shape[0])

                                    std_dev = torch.std(counts.float())
                                    print(counts, counts.shape[0])



                                val_loss = np.mean(val_losses)
                                print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')
                                early_stopping(val_loss, model)
                                print("---------------------------------------")
                                if early_stopping.early_stop:
                                    print("Early stopping")
                                    break

