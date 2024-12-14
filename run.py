import os

data_size = 500
min_width=1
lmbd=0.5

import torch

print("gpu:",torch.cuda.is_available())


models = [
"/home/sds/ustcllm/libo/code/model/meta-llama/Llama-2-7b/",
"/home/sds/ustcllm/libo/code/model/baichuan-inc/Baichuan-7B/",
"/home/sds/ustcllm/libo/code/model/opt-6.7b/",
"/home/sds/ustcllm/libo/code/model/meta-llama/Llama-2-7b-chat-hf/",
"/home/sds/ustcllm/libo/code/model/baichuan-inc/Baichuan2-7B-Chat/",
]

names = [
"Llama-2-7b",
"Baichuan-7B",
"opt-6.7b",
"Llama-2-7b-chat-hf",
"Baichuan2-7B-Chat"
]


for i in [4, 5]:
    name = names[i]
    model = models[i]

    file_name = "./result/1022/1023_qa_len200_short_{}_{}_lmbd_{}".format(name,data_size,lmbd)
    
    output_file=file_name + ".json"
    
    
    os.system("python watermark_240419.py --data_size {} --out_dataset_path {} --model {} --data_size {} --lmbd {} --min_width {}".format(data_size,output_file,model,data_size, lmbd, min_width))
    
    
    
    save_data_path =output_file
    #os.system("python dipper.py --save_data_path {}".format(save_data_path))
    
    
    dataset_path = file_name+".json"
    out_dataset_path=file_name+"_zscore_3.json"
    os.system("python detect_240419.py --dataset_path {} --out_dataset_path {} --lmbd {} --min_width {}".format(dataset_path,out_dataset_path, lmbd, min_width))