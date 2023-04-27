import json
import os

DATA_PATH = "./data/vectors/tokenized_wiki.json"
OUT_PATH = "./data/training_files_four/shard_"
N_GPUS = 4

if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_full = json.load(f)
        print(f"Length of data: {len(data_full)}")


    x = len(data_full) // N_GPUS 


    for i in range(N_GPUS-1):
        with open(f"{OUT_PATH}{i}.json", "w", encoding="utf-8") as f:
            json.dump(data_full[i*x:(i+1)*x], f)


    with open(f"{OUT_PATH}{N_GPUS-1}.json", "w", encoding="utf-8") as f:
        json.dump(data_full[(N_GPUS-1)*x:], f)

    print("Done sharding. Lets take a glass of SHARDonnay, am I right??")
