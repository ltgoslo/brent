import math
import json
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Creation of Retrieval database")
    parser.add_argument("--data_path", type=str, default="./norec_sent/train.json", help="Path to data for the database")
    parser.add_argument("--data_json_format", action="store_true")
    parser.add_argument("--chunk_size", default=128, type=int, help="The max size of each document in the database")
    parser.add_argument("--output_path", type=str, default="./norec_sent/train_database.json", help="Output path of the database")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="ltgoslo/norbert2")
    parser.add_argument("--add_context", action="store_true")
    args = parser.parse_args()
    return args

def create_json_file_from_json(tokenizer, data_path, output_path, chunk_size):
    """
    data_path: json file
    output_path: json_file
    """
    c = chunk_size
    docs = []
    with open(data_path, encoding="utf8") as f:
        documents = json.load(f)
        for i, doc in enumerate(tqdm(documents)):
            tokenised_doc = tokenizer(doc["text"], add_special_tokens=True, truncation=True, padding="max_length", max_length=chunk_size)
            d_input_ids = tokenised_doc["input_ids"]
            d_attention_mask = tokenised_doc["attention_mask"]
            assert len(d_input_ids) == c
            docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "text": doc["text"], "label": doc["label"], "id": i})
    
    with open(output_path, 'w', encoding="utf-8") as jf:
        json.dump(docs, jf, ensure_ascii=False)
        print(f"Number of chunks written to disk: {len(docs)} ")

def create_json_file_from_json_with_label(tokenizer, data_path, output_path, chunk_size):
    """
    data_path: json file
    output_path: json_file
    """
    c = chunk_size
    docs = []
    with open(data_path, encoding="utf8") as f:
        documents = json.load(f)
        for i, doc in enumerate(tqdm(documents)):
            tokenised_doc = tokenizer(doc["label"] + " " + doc["text"], add_special_tokens=True, truncation=True, padding="max_length", max_length=chunk_size)
            d_input_ids = tokenised_doc["input_ids"]
            d_attention_mask = tokenised_doc["attention_mask"]
            assert len(d_input_ids) == c
            docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "text": doc["label"] + " " + doc["text"], "label": doc["label"], "id": i})
    
    with open(output_path, 'w', encoding="utf-8") as jf:
        json.dump(docs, jf, ensure_ascii=False)
        print(f"Number of chunks written to disk: {len(docs)} ")


def create_json_file(tokenizer, data_path, output_path, chunk_size, no_text=False):
    c = chunk_size
    docs = []
    with open(data_path, encoding="utf8") as f:
        raw = f.read()
        documents = raw.split("\n\n") #Docs are new line delimited
        for i, doc in enumerate(tqdm(documents)):
            tokenised_doc = tokenizer(doc, add_special_tokens=True, return_offsets_mapping=True)
            n_chunks = math.floor(len(tokenised_doc["input_ids"])/c)
            q = 0
            for i in range(1, n_chunks+1):
                s = (i-1)*c
                e = (i)*c
                d_input_ids = tokenised_doc["input_ids"][s:e]
                d_attention_mask = tokenised_doc["attention_mask"][s:e]
                offset_mapping = tokenised_doc["offset_mapping"][s:e]
                text = tokenizer.decode(d_input_ids, skip_special_tokens=True)
                if no_text:
                    docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask})
                else:
                    docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "offset_mapping": offset_mapping, "text": text})
                q = e
            d_input_ids = tokenised_doc["input_ids"][q:]
            d_attention_mask = tokenised_doc["attention_mask"][q:]
            offset_mapping = tokenised_doc["offset_mapping"][q:]
            text = tokenizer.decode(d_input_ids, skip_special_tokens=True)
            for i in range(len(d_input_ids), c): #Pad remainder to c
                d_input_ids.append(0)
                d_attention_mask.append(0)
            assert len(d_input_ids) == c
            if no_text:
                docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask})
            else:
                docs.append({"input_ids": d_input_ids, "attention_mask": d_attention_mask, "offset_mapping": offset_mapping, "text": text})
    
    with open(output_path, 'w', encoding="utf-8") as jf:
        json.dump(docs, jf, ensure_ascii=False)
        print(f"Number of chunks written to disk: {len(docs)} ")

if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    if args.data_json_format:
        if args.add_context:
            create_json_file_from_json_with_label(tokenizer, args.data_path, args.output_path, args.chunk_size)
        else:
            create_json_file_from_json(tokenizer, args.data_path, args.output_path, args.chunk_size)
    else:
        create_json_file(tokenizer, args.data_path, args.output_path, args.chunk_size)
