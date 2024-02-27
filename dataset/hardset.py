import pickle
import torch
from torch.utils.data import DataLoader
from dataset.LVDataset import LVDataset
import numpy as np
import time
import random
import os
import openai
import multiprocessing
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
deployment_name='gpt35' 
print(openai.api_key, openai.api_base)
from dataset.hard_prompt import A, B

def random_set(entry_paths, size = 2000):
    random.shuffle(entry_paths)
    with open('./data/dataset/v1_random.pkl', 'wb') as f:
        pickle.dump(entry_paths[0:size], f)
    return entry_paths[0:size]

def test_prompt(languages):
    cnt = 0
    for language in languages:
        if gpt_query(language):
            cnt += 1
    print("{}/{}".format(cnt, len(languages)))
        
def gpt_query(language):
    completion = openai.ChatCompletion.create(
    engine=deployment_name,
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You should judge the following description based on the given information. The description is about a specific object and its parts. You should judge whether this object is category A or catergory B. Here are the characteristics of A and B. Material Usage: A (precious/uncommon) vs. B (common/practical); Friction Focus: A (ease of use/movement) vs. B (safety/stability); Density and Weight Considerations: A (heavier/solid) vs. B (lighter/varied); Fragility vs. Toughness: A (balanced) vs. B (durable/heavier use); Grasp Probability Guidance: A (specific) vs. B (general). Please Answer 'A' or 'B'. If you are not sure, please answer 'I don't know'."},
        # {"role": "system", "content": PT,},
        {"role": "user", "content": language},
        ],
    )
    input_tokens = completion.usage['prompt_tokens']
    output_tokens = completion.usage['completion_tokens']
    response = completion.choices[0].message['content']
    print(response) #,'Tokens = ',input_tokens,'+',output_tokens,'=',input_tokens+output_tokens,'Price =', 1e-3*input_tokens*0.01+1e-3*output_tokens*0.03)
    # convert response to boolean
    return 'A' in response

def get_hard(entry_paths):
    hard_entry_paths = []
    for entry_path in entry_paths:
        with open(entry_path, 'rb') as f:
            entry = pickle.load(f)
            language = entry.language
            if gpt_query(language):
                hard_entry_paths.append(entry_path)
            # time.sleep(1.0)
        # with open('./data/dataset/hard_entry_paths_p2.pkl', 'wb') as f:
        #     pickle.dump(hard_entry_paths, f)
    return hard_entry_paths

def main():
    hard_entry_paths = []
    with open('./data/dataset/test_dataset_v2.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    entry_paths = test_dataset.entry_paths
    hard_q = pickle.load(open('./evaluation/hard_entry_paths_q.pkl', 'rb'))
    hard_x = pickle.load(open('./evaluation/hard_entry_paths_x.pkl', 'rb'))
    random.shuffle(hard_q)
    random.shuffle(hard_x)
    hards = hard_q[:3000] + hard_x[:3000]
    entry_paths = list(set(entry_paths) - set(hards))
    random.shuffle(entry_paths)
    random.shuffle(hards)
    paths = entry_paths[:1800] + hards[:600]
    
    N_PROC = 2
    tasks = []
    for i in range(N_PROC):
        tasks.append(paths[i::N_PROC])
    pool = multiprocessing.Pool(processes=N_PROC)
    results = pool.map(get_hard, tasks)
    for result in results:
        hard_entry_paths += result
    with open('./data/dataset/hard_entry_paths_q2.pkl', 'wb') as f:
        pickle.dump(hard_entry_paths, f)
    print(len(hard_entry_paths))

if __name__ == "__main__":
    main()