import torch
import numpy as np
import torch.nn.functional as F

def retrieval_utility(pred_logits_per_doc, marginal_gold_log_probs, k, queries=None, targets=None, masked_token_ids=None, num_mask=None):

    if queries is not None and targets is not None:
        masked_token_ids = torch.tensor(queries['input_ids']) == tokenizer.mask_token_id
        num_mask = torch.sum(masked_token_ids)
    elif masked_token_ids is None or num_mask is None:
        print("Error! Requires either queries and targets or masked_token_ids and num_mask")
        return

    if targets:
        gold_preds_per_doc = F.cross_entropy(pred_logits_per_doc, targets)
    else:
        gold_preds_per_doc = pred_logits_per_doc

    batch_size = gold_preds_per_doc.shape[0] // k

    masked_gold_preds_per_doc = gold_preds_per_doc.masked_fill(~masked_token_ids.unsqueeze(1).bool(), 0)

    masked_gold_preds_null_doc = masked_gold_preds_per_doc[:, k-1]

    retrieval_utility = (masked_gold_preds_null_doc.unsqueeze(1) - masked_gold_preds_per_doc[:, :k-1]).sum(dim=[0,2]) / num_mask

    RU = (masked_gold_preds_null_doc - marginal_gold_log_probs).sum() / num_mask

    return retrieval_utility, retrieval_utility.max(), retrieval_utility.mean(), RU

def accuracy(pred_logits_per_doc, k, targets, queries=None, masked_token_ids=None, num_mask=None):

    if queries is not None:
        masked_token_ids = torch.tensor(queries['input_ids']) == tokenizer.mask_token_id
        num_mask = torch.sum(masked_token_ids)
    elif masked_token_ids is None or num_mask is None:
        print("Error! Requires either queries or masked_token_ids and num_mask")
        return
    
    batch_size = pred_logits_per_doc.shape[0] // k
    seq_len = pred_logits_per_doc.shape[1]

    pred_logits_per_doc = pred_logits_per_doc.view(k, batch_size, seq_len, -1).transpose(0,1)
    targets = targets.view(k, batch_size, seq_len).transpose(0,1)
    full_correct_per_doc = pred_logits_per_doc.argmax(dim=3) == targets

    masked_correct_per_doc = full_correct_per_doc.masked_fill(~masked_token_ids.unsqueeze(1).bool(), 0)

    document_accuracy = masked_correct_per_doc.sum(dim=[0,2]) / num_mask

    return document_accuracy

def ru_single_item(pred_logits_per_doc, marginal_gold_log_probs, k, targets=None):

    if targets:
        gold_preds_per_doc = F.cross_entropy(pred_logits_per_doc, targets)
    else:
        gold_preds_per_doc = pred_logits_per_doc

    batch_size = gold_preds_per_doc.shape[0] // k

    gold_preds_null_doc = gold_preds_per_doc[:, k-1]

    retrieval_utility = (gold_preds_null_doc.unsqueeze(1) - gold_preds_per_doc[:, :k-1]).mean(dim=[0])

    RU = (gold_preds_null_doc - marginal_gold_log_probs).mean()

    return retrieval_utility, retrieval_utility.max(), retrieval_utility.mean(), RU

def acc_single_item(pred_logits_per_doc, k, targets):

    batch_size = pred_logits_per_doc.shape[0] // k

    pred_logits_per_doc = pred_logits_per_doc.view(k, batch_size, -1).transpose(0,1)
    targets = targets.view(k, batch_size).transpose(0,1)
    full_correct_per_doc = pred_logits_per_doc.argmax(dim=2) == targets

    document_accuracy = full_correct_per_doc.sum(dim=[0]) / len(full_correct_per_doc)

    return document_accuracy

def ru_sequence(pred_logits_per_doc, marginal_gold_log_probs, k, targets=None):

    if targets:
        gold_preds_per_doc = F.cross_entropy(pred_logits_per_doc, targets, reduction="none")
        gold_preds_per_doc = gold_preds_per_doc.view(k, gold_preds_per_doc.shape[0] // k, -1).transpose(0,1)
    else:
        gold_preds_per_doc = pred_logits_per_doc

    gold_preds_null_doc = gold_preds_per_doc[:, k-1]

    retrieval_utility = (gold_preds_null_doc.unsqueeze(1) - gold_preds_per_doc[:, :k-1]).mean(dim=[0,2])

    RU = (gold_preds_null_doc - marginal_gold_log_probs).mean()

    return retrieval_utility, retrieval_utility.max(), retrieval_utility.mean(), RU

def acc_sequence(pred_logits_per_doc, k, targets, num_words):

    batch_size = pred_logits_per_doc.shape[0] // k
    seq_len = pred_logits_per_doc.shape[1]

    pred_logits_per_doc = pred_logits_per_doc.view(k, batch_size, seq_len, -1).transpose(0,1)
    targets = targets.view(k, batch_size, seq_len).transpose(0,1)
    full_correct_per_doc = pred_logits_per_doc.argmax(dim=3) == targets

    document_accuracy = full_correct_per_doc.sum(dim=[0,2]) / num_words

    return document_accuracy