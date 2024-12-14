import sampling_utils
from sampling import get_mask_from_seed
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)




def flatten_gens_and_paras(gens, paras):
    new_gens = []
    new_paras = []
    for gen, para in zip(gens, paras):
        min_len = min(len(gen), len(para))
        new_gens.extend(gen[:min_len])
        new_paras.extend(para[:min_len])
    return new_gens, new_paras


def truncate_to_max_length(texts, max_length):
    new_texts = []
    for t in texts:
        t = " ".join(t.split(" ")[:max_length])
        if t[-1] not in sampling_utils.PUNCTS:
            t = t + "."
        new_texts.append(t)
    return new_texts






def detect(sents, model, lmbd, dim, cutoff=None, min_width=4, hash_key = 1,prompt=""):
    n_sent = len(sents)
    begin = 1
    text = sents[0]
    n_sent = n_sent - 1
    n_test_sent = n_sent - 1

    #text = sents[0]

    all_seed = []
    n_watermark = 0

    seed = model.get_hash([text.strip().replace('\n', '').replace('"', '')])[0]
    all_seed.append(seed)
    all_hash = []
    all_hash.append(seed)
    all_hash_sentence = []
    all_hash_sentence.append(seed)
    accept_mask = get_mask_from_seed(dim, lmbd, seed)

    cnt_g = 0

    for i in range(begin, len(sents)):
        if len(sents[i])<2:
            n_sent -= 1
            text = text.strip() + " " + sents[i].strip()

            continue
        candidate = model.get_hash([sents[i].strip().replace('\n', '').replace('"', '')])[0]
        all_hash_sentence.append(candidate)
        if candidate in accept_mask:


            cnt_g+=1
            n_watermark += 1

        text = text.strip()+" "+sents[i].strip()

        seed = candidate
        all_hash.append(seed)
        all_seed.append(seed)
        

        seed = min(all_seed if len(all_seed)<min_width else all_seed[-min_width:])
        
        

        accept_mask = get_mask_from_seed(dim, lmbd, seed)
        # accept_mask = [i for i in range(16)]



    #二项分布
    print(all_hash, all_hash_sentence)
    print(f'n_watermark: {n_watermark}, n_test_sent: {n_sent}')

    from scipy.stats import binom_test

    p_value = binom_test(n_watermark, n_sent, lmbd, alternative='greater')

    print(f"zscore: {1-p_value}")
    return 1-p_value


    return zscore








