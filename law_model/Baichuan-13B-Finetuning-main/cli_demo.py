import os
import torch
import platform
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from utils import (
    load_pretrained,
    prepare_infer_args
)
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss

def load_data_and_model( index_path, json_path):
    # 1. 加载SentenceTransformer模型并转移到CUDA
    
    
    # 加载Faiss索引
    index = faiss.read_index(index_path)
    
    # 从JSON文件加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]
    
    return index, data_list

def query_faiss(model, index, data_list, query_text, k):
    # 2. 向量化查询文本
    query_vector = model.encode([query_text])[0]

    
    # 3. 使用Faiss进行检索
    distances, indices = index.search(np.array([query_vector]), k)
    
    # 返回检索到的文档的文字内容
    results = [data_list[idx]["value"] for idx in indices[0]]
    return results

# 使用方法：
model_path = "/home/jshi/chatlaw"
index_path1 = "/home/jshi/2023/faiss_FATIAO.idx"
json_path1 = "/home/jshi/output_file.json" # 请替换为您的JSON文件路径


def init_model():
    print("init model ...")
    
    model_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    model.generation_config = GenerationConfig.from_pretrained(
        model_args.model_name_or_path
    )
    model.generation_config.max_new_tokens = generating_args.max_new_tokens
    model.generation_config.temperature = generating_args.temperature
    model.generation_config.top_k = generating_args.top_k
    model.generation_config.top_p = generating_args.top_p
    model.generation_config.repetition_penalty = generating_args.repetition_penalty
    model.generation_config.do_sample = generating_args.do_sample
    model.generation_config.num_beams = generating_args.num_beams
    model.generation_config.length_penalty = generating_args.length_penalty
    
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    model1 = SentenceTransformer(model_path).to(device="cuda")
    messages = []
    prompt = input("请输入您的问题:\n")
    index, data_list = load_data_and_model(index_path1, json_path1)
    r = query_faiss(model1,index,data_list,prompt,5)
    print("检索到法条为：\n")
    print(r)
    prompt = prompt+"请依据"
    for a in r:
        prompt=prompt+a+'\n'
    prompt=prompt+'来进行回答'
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)
    print(response)
    messages.append({"role": "assistant", "content": response})
    


if __name__ == "__main__":
    main()
