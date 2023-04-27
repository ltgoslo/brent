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

'''

    This code implements the ICT pre-training task from https://arxiv.org/abs/1906.00300
    Our implementation is modified from https://github.com/donggyukimc/Inverse-cloze-task to fit the processed wikipedia article dumps from ./utils/create_ict_corpus.py 

'''

device = 'cuda:0' if torch.cuda.is_available() else 'mps' #use Mac M1 chip for local fun. 
#torch.backends.cuda.matmul.allow_tf32 = True #(RTX3090 and A100 GPU only)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Warmup the retriever with the ICT task over wikipedia articles")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/ict_warmup.json",
        help="Path to the processed wiki dump"
    )

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

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001 #Hyperparameter from o.g paper
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    args = parser.parse_args()
    return args


class Embedder(nn.Module):
    def __init__(self, model_name_or_path) :
        super(Embedder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, x, x_mask) :
        return self.encoder(input_ids=x, attention_mask=x_mask).pooler_output

def load_data(path):
    '''
        Returns a json object of the ict warmup corpus
    '''
    with open(path, 'r', encoding='utf8') as f:
        data = list(json.load(f))
    return data



def pad_sequence(x, max_seq=64, pad_idx=0, get_mask=True, decode=False, pad_max=False, device=device): # copied from https://github.com/donggyukimc/Inverse-cloze-task
    """
    padding given sequence with maximum length 
    generate padded sequence and mask
    """ 
    seq_len = np.array([min(len(seq), max_seq) for seq in x])
    if not pad_max :
        max_seq = max(seq_len)
    pad_seq = np.zeros((len(x), max_seq), dtype=np.int64)
    pad_seq.fill(pad_idx)
    for i, seq in enumerate(x):
        pad_seq[i, :seq_len[i]] = seq[:seq_len[i]]
    if get_mask :
        mask = make_mask(pad_seq, pad_idx, decode)
        mask = torch.from_numpy(mask).to(device)
    else :
        mask = None
    return torch.from_numpy(pad_seq).to(device), mask

def make_mask(x, pad_idx, decode=False): # copied from https://github.com/donggyukimc/Inverse-cloze-task
    "Create a mask to hide padding and future words."
    mask = (x!=pad_idx)
    if decode:
        size = x.shape[-1]
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        mask = np.expand_dims(mask, axis=1) & (subsequent_mask == 0)
    return mask.astype('uint8')

def create_contexts(data, tokenizer):
    '''
        Creates a list of lists with the sentences included in each document, tokenized.
    '''
    contexts = []
    for document in tqdm(data):
        sentences = nltk.sent_tokenize(document)
        s = {}
        a = []
        for sen in sentences:
            a.append(tokenizer(sen, add_special_tokens=False)["input_ids"])
        s["sentence"] = a
        contexts.append(s)

    return contexts


def create_data_loader(data, batch_size):
    '''
        Returns a DataLoader object that holds the indexes of which document is included in a batch. The batch size - 1 is thus equal to the number of neg. samples for a batch.
    '''
    return DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=True, drop_last=True)

def get_batch(index, contexts, start_token, tokenizer, debug=False): #modified https://github.com/donggyukimc/Inverse-cloze-task
    '''
        Creates padded and masked sequences of a batch of documents to retrieve from, with a target sentenced removed from the batch with probability p
    '''
    sentence = [contexts[i]["sentence"] for i in index] # get sentences of paragraphs from the documents in the batch
    target_sentence = [random.randint(0, len(sen)-1) for sen in sentence] # set target sentence for ICT training
    remove_target = [random.random()<(1-0.1) for _ in range(len(target_sentence))] # determine removal of original sentence as mentioned in og. paper
    target_context = [sen[:i]+sen[i+remove:] for i, sen, remove in zip(target_sentence, sentence, remove_target)] # set sentences of target context
    target_context = [[y for x in context for y in x] for context in target_context] # concat sentences of context
    target_context = [[start_token]+context for context in target_context]
    target_sentence = [sen[i] for i, sen in zip(target_sentence, sentence)]
    target_sentence = [[start_token]+sen for sen in target_sentence]
    s, s_mask = pad_sequence(target_sentence, max_seq=128)
    c, c_mask = pad_sequence(target_context, max_seq=128)
    if debug:
        target_sent = tokenizer.decode(s[0], skip_special_tokens=True)
        target_cont = [tokenizer.decode(t, skip_special_tokens=True) for t in c]
        print(target_sent) # A LOT of UNKs from the norwegian wiki???
        print(target_cont)
    return s, s_mask, c, c_mask


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    raw_data = load_data(args.data_path)
    contexts = create_contexts(raw_data, tokenizer)
    indexed_docs = np.array([i for i in range(len(contexts))])
    loader = create_data_loader(indexed_docs, args.batch_size)
    n = len(loader)
    epochs = args.epochs #According to the ORCA paper, they do a batch size of 4096 for 100k steps... As we do not have that much data, nor memory, we need to adjust this.
    softmax = nn.Softmax()

    model = Embedder(args.model_name_or_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    vocab = dict()
    for k, v in tokenizer.vocab.items() :
        vocab[k] = v
    start_token = vocab["[CLS]"]

    best_accuracy = 0.0

    for epoch in range(epochs):
            logging.info(f"Starting ICT training at epoch {epoch}/{epochs}")
            model.train()
            train_loss = 0.0
            targets_b = torch.Tensor([]).to(device)
            preds_b = torch.Tensor([]).to(device)

            for i, batch in enumerate(tqdm(loader)):
                '''
                    s = [batch_size, max sentence length in batch]
                    c = [batch_size, max_seq_length]
                    why? We have n target sentences (n=batch_size) in s, and in c we have n contexts, where each context are as many sentences fit within the max_seq_length and where the o.g sent have been removed with a proabability of 0.9
                '''
                optimizer.zero_grad()
                batch = batch[0] #ids of the documents
                s, s_mask, c, c_mask = get_batch(batch, contexts, start_token, tokenizer, debug=False)

                s_encode = model(x=s, x_mask=s_mask)
                c_encode = model(x=c, x_mask=c_mask)
                logit = torch.matmul(s_encode, c_encode.transpose(-2, -1)) #This finds the inner product between the n sentences and the n contexts
                target = torch.from_numpy(np.array([i for i in range(batch.size(0))])).long().to(device) #give each of the n documents an index
                
                #optimize for highest similarity for all ten sentences. We dont only check one sentence against 9 distractors once, we do it for all 10 sentences where the rest are distractors each time.
                loss_val = criterion(logit, target).mean() 
                train_loss += loss_val.item()
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                #retrieval accuracy
                preds = torch.argmax(softmax(logit), dim=1)
                targets_b = torch.cat((targets_b, target), 0)
                preds_b = torch.cat((preds_b, preds), 0)


            accuracy = torch.sum(preds_b == targets_b)/len(preds_b)
            logging.info(f"Batch acc: {accuracy}")
            logging.info("{} Epoch, train loss : {}".format(epoch+1, round(train_loss/n, 4)))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                logging.info(f"Saving model checkoint at {args.output_path}...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_path)
            else:
                break
            
