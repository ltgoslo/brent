import torch
import math
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
import json
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
device = 'cuda:0' if torch.cuda.is_available() else 'mps' #use Mac M1 chip for local fun. 
#torch.backends.cuda.matmul.allow_tf32 = True #(RTX3090 and A100 GPU only)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed wikidump")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/vectors/tokenized_wiki.json",
        help="Path to the tokenized wiki json file"
    )

    parser.add_argument(
        "--no_text",
        action="store_true"
    )

    parser.add_argument(
    "--og_data_path",
    type=str,
    default="./data/nowiki/nowiki.txt",
    help="Path to the unprocessed wiki file "
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/vectors/vectors.npy",
        help="Output path of the computed embeddings"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltgoslo/norbert2"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=126 #Hyperparameter from o.g paper
    )

    args = parser.parse_args()
    return args


def embed(data_path, model, projector, device, output_path=False):
    '''
        Partitions wikidump by article and iterates with a sliding window of size chunk_size over the articles. Gets [CLS] for each chunk.
            args:
                data_path (str): path to the wikipedia dump, assumes articles delimited by newline
                output_path (str): path to store the produced embeddings in a .npy file
                model: embedding function
                tokenizer: tokenizer to use
                chunk_size: size of each chunk to embed
            returns:
                Saves a file of shape [n, 768] in npy format at output_path
    '''
    model.to(device)
    # projector.to(device)
    
    with torch.no_grad():
        vectors = []
        dataset = Wikidataset(data_path)
        dataloader = DataLoader(dataset, batch_size=128)
        model.eval()
        for (input_ids, attention_mask) in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.cuda.amp.autocast(True):
                embedding = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output #Get [CLS]
                # embedding = projector(embedding)
                # emb_norm = torch.linalg.norm(embedding, dim=1)
                # embedding = embedding / emb_norm.unsqueeze(-1)
            embedding = embedding.squeeze(0).cpu().detach().numpy().astype(np.float32).tolist()
            for emb in embedding:
                vectors.append(emb)
        vectors = np.array(vectors).astype(np.float32)
        if output_path:
            np.save(output_path, vectors)
        else:
            return vectors


def create_json_file(tokenizer, data_path, output_path, chunk_size, no_text=False):
    c = chunk_size
    docs = []
    with open(data_path, encoding="utf8") as f:
        raw = f.read()
        documents = raw.split("\n\n") #Docs are new line delimited
        for i, doc in enumerate(tqdm(documents)):
            tokenised_doc = tokenizer(doc, add_special_tokens=False)
            doc = tokenizer.decode(tokenised_doc["input_ids"], skip_special_tokens=False)
            tokenised_doc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
            n_chunks = math.floor(len(tokenised_doc["input_ids"])/c)
            q = 0
            for i in range(1, n_chunks+1):
                s = (i-1)*c
                e = (i)*c
                d_input_ids = tokenised_doc["input_ids"][s:e]
                d_attention_mask = tokenised_doc["attention_mask"][s:e]
                offset_mapping = tokenised_doc["offset_mapping"][s:e]
                offset_mapping = [[offset_mapping[i][0]-offset_mapping[0][0], offset_mapping[i][1]- offset_mapping[0][0]] for i in range(0, len(offset_mapping))]
                d_input_ids = [tokenizer.cls_token_id] + d_input_ids + [tokenizer.sep_token_id]
                d_attention_mask = [1] + d_attention_mask + [1]
                offset_mapping = [[0,0]] + offset_mapping + [[0,0]]

                text = tokenizer.decode(d_input_ids[1:-1], skip_special_tokens=False)
                if text.startswith("##"):
                    text = text[2:]
                if no_text:
                    docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask})
                else:
                    docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "offset_mapping": offset_mapping, "text": text})
                q = e
            d_input_ids = tokenised_doc["input_ids"][q:]
            d_input_ids = [tokenizer.cls_token_id] + d_input_ids + [tokenizer.sep_token_id]
            d_attention_mask = tokenised_doc["attention_mask"][q:]
            d_attention_mask = [1] + d_attention_mask + [1]
            offset_mapping = tokenised_doc["offset_mapping"][q:]
            offset_mapping = [[offset_mapping[i][0]-offset_mapping[0][0], offset_mapping[i][1]- offset_mapping[0][0]] for i in range(0, len(offset_mapping))]
            offset_mapping = [[0,0]] + offset_mapping + [[0,0]]

            text = tokenizer.decode(d_input_ids[1:-1], skip_special_tokens=False)
            if text.startswith("##"):
                    text = text[2:]
            for i in range(len(d_input_ids), c+2): #Pad remainder to c
                d_input_ids.append(0)
                d_attention_mask.append(0)
                offset_mapping.append([0,0])

            assert len(d_input_ids) == c +2
            assert len(offset_mapping) == c +2
            assert len(d_attention_mask) == c +2
            if no_text:
                docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask})
            else:
                docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "offset_mapping": offset_mapping, "text": text})
    

    with open(output_path, 'w', encoding="utf-8") as jf:
        json.dump(docs, jf, ensure_ascii=False)
        print(f"Number of chunks written to disk: {len(docs)} ")

class Wikidataset(Dataset):
    def __init__(self, path):
        f = open(path)
        self.documents = json.load(f)
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, index):
        i_ids = self.documents[index]["input_ids"]
        a_mask = self.documents[index]["attention_mask"]

        return torch.LongTensor(i_ids), torch.LongTensor(a_mask)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.no_text:
        create_json_file(tokenizer, args.og_data_path, args.data_path, args.chunk_size, no_text=True)
        #embed(args.data_path, model, device, args.output_path)
    else:
        create_json_file(tokenizer, args.og_data_path, args.data_path, args.chunk_size, no_text=False)
        #embed(args.data_path, model, device, args.output_path)
