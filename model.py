from faiss import IndexFlatIP, IndexFlatL2
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM
import numpy as np
import math


# Model Architecture

## Retriever

### Index to represent embedded documents
### Query Embedder (Also the document embedder)
### Search function

## Reader

### Enccoder

## Params:

### input queries
### index
### k (number of documents to retrieve)

class REALM(nn.Module):

    def __init__(self, query_embedder, doc_embedder, reader_model, database, tokenizer, device, d, k, seq_len=128, share=False, use_null_doc=False):
        super().__init__()
        self.query = AutoModel.from_pretrained(query_embedder, local_files_only=True) # query embedder
        # self.W_in = nn.Linear(768, d)
        
        if share:
            self.doc = self.query
        else:
            self.doc = AutoModel.from_pretrained(doc_embedder, local_files_only=True) # doc embedder
            # self.W_doc = nn.Linear(768, d)

        self.db = database

        self.reader = AutoModelForMaskedLM.from_pretrained(reader_model, local_files_only=True)

        self.k = k
        self.d = d
        self.use_null_doc = use_null_doc
        #self.null_doc = nn.Parameter(torch.normal(torch.zeros(d), torch.ones(d)*0.2))
        self.null_doc = nn.Parameter(torch.zeros(d))
        self.null_tokens = tokenizer("", padding="max_length", max_length=seq_len, return_tensors="pt").to(device) 
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, queries, query_ids, pad_ids, seq_len=128, version="v1"):

        batch_size = len(queries['input_ids'])

        # Retrieval step
        input_ids = torch.tensor(queries['input_ids']).to(self.device)
        att_mask = torch.tensor(queries['attention_mask']).to(self.device)

        emb_in = self.query(input_ids=input_ids, attention_mask=att_mask).pooler_output
        emb_in = emb_in

        #emb_in = self.W_in(emb_in)

        #emb_in_norm = torch.linalg.norm(emb_in, dim=1)
        #emb_in = emb_in / emb_in_norm.unsqueeze(-1)

        _, doc_ids = self.index.search(emb_in.cpu().detach().numpy().astype(np.float32), self.k+1)

        retrieval_ids = np.zeros((doc_ids.shape[0], doc_ids.shape[1]-1), dtype=int)

        # Search to remove trivial documents
        for i, top_k_docs in enumerate(doc_ids):
            if query_ids[i].item() in top_k_docs:
                top_k_docs = np.delete(top_k_docs, np.argwhere(top_k_docs == query_ids[i].item()))
            retrieval_ids[i] = top_k_docs[:self.k]

        # Fetching the retrieval documents
        retrieval_docs = {'input_ids': [], 'attention_mask': []}

        for i in range(self.k):
            for r_ids in retrieval_ids:
                retrieval_docs['input_ids'].append(self.db[r_ids[i]]['input_ids'])
                retrieval_docs['attention_mask'].append(self.db[r_ids[i]]['attention_mask'])
        
        # Appending queries and documents

        query_context = {'input_ids': [], 'attention_mask': []}
        
        if version == "v1":
            for i in range(self.k):
                for j in range(batch_size):
                
                    to_pad = 0
                
                    d = retrieval_docs['input_ids'][i*batch_size+j][1:]
                    da = retrieval_docs['attention_mask'][i*batch_size+j][1:]
                
                    if pad_ids[j] is None:
                        q = queries['input_ids'][j]
                        qa = queries['attention_mask'][j]
                    else:
                        q = queries['input_ids'][j][:pad_ids[j]]
                        qa = queries['attention_mask'][j][:pad_ids[j]]
                        to_pad = seq_len - pad_ids[j]
                
                    if to_pad:
                        q_c = q + d + [self.tokenizer.pad_token_id]*to_pad
                        q_c_a = qa + da + [0]*to_pad
                    else:
                        q_c = q + d
                        q_c_a = qa + da
                    
                    query_context['input_ids'].append(q_c)
                    query_context['attention_mask'].append(q_c_a)

        elif version == "v2":
            for i in range(self.k):
                for j in range(batch_size):

                    d = retrieval_docs['input_ids'][i*batch_size+j][1:]
                    da = retrieval_docs['attention_mask'][i*batch_size+j][1:]

                    q = queries['input_ids'][j]
                    qa = queries['attention_mask'][j]

                    q_c = q + d
                    q_c_a = qa + da

                    query_context['input_ids'].append(q_c)
                    query_context['attention_mask'].append(q_c_a)

        # Adding null document

        for j in range(batch_size):
            
            q = queries['input_ids'][j]
            qa = queries['attention_mask'][j]
            
            query_context['input_ids'].append(q + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id]*(seq_len-2))
            query_context['attention_mask'].append(qa + [1] + [0]*(seq_len-2))

        # Embedding documents

        emb_doc = self.doc(input_ids=torch.tensor(retrieval_docs['input_ids']).to(self.device), 
                           attention_mask=torch.tensor(retrieval_docs['attention_mask']).to(self.device)).pooler_output

        # With the embedded empty string
        if self.use_null_doc:
            emb_doc = torch.cat((emb_doc, self.null_doc.expand(batch_size, -1)), 0)
        else:
            null = self.doc(**self.null_tokens).pooler_output    
            emb_doc = torch.cat((emb_doc, null.expand(batch_size, -1)), 0)
        
        # emb_doc = self.W_doc(emb_doc)

        # emb_doc_norm = torch.linalg.norm(emb_doc, dim=1)

        # emb_doc = emb_doc / emb_doc_norm.unsqueeze(-1)
        

        # Calculating logits for marginal distribution

        emb_doc = emb_doc.reshape(self.k+1, batch_size, -1)
        marginal_doc_logits = torch.einsum('BD, KBD -> BK', emb_in, emb_doc)
        marginal_doc_logits /= math.sqrt(self.d)
        #log_mar_doc_prob = nn.LogSoftmax(marginal_doc_prob)

        # Reading

        # [batch_size * k+1, seq_len, vocab_size]
        logits = self.reader(input_ids=torch.tensor(query_context['input_ids']).to(self.device),
                             attention_mask=torch.tensor(query_context['attention_mask']).to(self.device)).logits[:,:128]

        # mask_token_logits = torch.where(query_context['input_ids'] == self.tokenizer.mask_token_id)
        # masked_logits = 

        return marginal_doc_logits, logits, retrieval_ids

    
    def update_index(self, embeddings):

        self.index = IndexFlatIP(self.d)
        self.index.add(embeddings)




