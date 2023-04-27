import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging
import json
import argparse
from tqdm import tqdm
import numpy as np
import nltk
import random

device = 'cuda:0' if torch.cuda.is_available() else 'mps' #use Mac M1 chip for local fun. 
#torch.backends.cuda.matmul.allow_tf32 = True #(RTX3090 and A100 GPU only)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Warmup the retriever with the ICT task over wikipedia articles")

    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/ict_model_chkpnt.pt",
        help="Output path of the warmed up model"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltgoslo/norbert2"
    )
    args = parser.parse_args()
    return args


class Embedder(nn.Module):
    def __init__(self, model_name_or_path) :
        super(Embedder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, x, x_mask) :
        return self.encoder(input_ids=x, attention_mask=x_mask).pooler_output


if __name__ == "__main__":

    args = parse_args()

    model = Embedder(args.model_name_or_path)

    ict_checkpoint = torch.load(args.output_path)

    model.load_state_dict(ict_checkpoint["model_state_dict"])

    torch.save({
        'model_state_dict': model.encoder.state_dict(),
    }, args.output_path)