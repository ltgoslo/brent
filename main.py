# coding=utf-8
import autofaiss
import argparse
import torch
from embedder import embed, Wikidataset
from model import REALM
from metrics import retrieval_utility, accuracy
import json
import torch.nn as nn
import numpy as np
from itertools import count
from mlm_dataset import MlmDataset
import math
from masker import Masker
import signal
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
import os
from socket import gethostname
import os.path
from utils import is_main_process, get_rank, seed_everything, get_world_size
from torch.nn.parallel import DistributedDataParallel
import time

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logging.info("Init")

#device = 'cuda:0' if torch.cuda.is_available() else 'mps' #use Mac M1 chip for local fun.

try:
    if int(os.environ["SLURM_PROCID"]) == 0:
        import wandb
except KeyError as e:
    logging.info("Not on slurm based system...")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-training a Retrieval-augmented Language Model")
    parser.add_argument("--wiki_base_path", type=str, default="./data/vectors/tokenized_wiki.json", help="Path to tokenized wikipedia dump")
    parser.add_argument("--shard_path", type=str, default="./data/training_files/shard_", help="Path to the sharded training data")
    parser.add_argument("--seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--output_path", type=str, default="./models/", help="Output path of the pre-trained model")
    parser.add_argument("--reader_model_name_or_path", type=str, default="ltgoslo/norbert2")
    parser.add_argument("--retrieval_model_name_or_path", type=str, default="./models/ict_model_chkpnt.pt")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.00003) #Hyperparameter from o.g paper
    parser.add_argument("--re_indexing_frequency", type=int, default=100, help="After how many steps will the index be refreshed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--use_ict", action="store_true")
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_steps", default=200000, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--device_max_steps", default=200000, type=int, help="Total number of training steps to perform per device")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--max_gradient", default=30.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True)
    parser.add_argument("--num_documents", default=726572, type=int)
    parser.add_argument("--num_embbed_dimensions", default=768, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--random_factor", default=1.0, type=float, help="Percentage of random masking compared to salient masking.")
    parser.add_argument("--salient_factor", default=1.0, type=float, help="Percentage of salient masking.")
    parser.add_argument("--mask_p", default=0.15, type=float, help="Percentage of spans to mask.")
    parser.add_argument("--wandb_name", default="Experiment_null", type=str, help="Name of the wandb experiment")
    args = parser.parse_args()
    return args

def prepare_model_and_optimizer(args, device, local_rank, model, checkpoint, load_ict=False):

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #wandb.config.update(config.to_dict())
        #wandb.config.update({"n_params": n_params})
        #logging.info(model)
        logging.info(f"NUMBER OF PARAMETERS: {n_params}\n")

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)
    
    # Load ICT pretrained model to the query and doc retriever
    if load_ict and checkpoint is None:
        logging.info("Loading ICT checkpoints as warmed-up models for the retriever...")
        ict_checkpoint = torch.load(args.retrieval_model_name_or_path)
        model.doc.load_state_dict(ict_checkpoint["model_state_dict"])
        model.query.load_state_dict(ict_checkpoint["model_state_dict"])

    model.to(device)

    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'embedding', 'null_doc']
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        logging.info("\nParameters without weight decay:")
        for n, p in no_decay_params:
            logging.info(n)
        
        logging.info("\nParameters with weight decay:")
        for n, p in decay_params:
            logging.info(n)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    if args.scheduler == "cosine":
        def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

                return lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)

    elif args.scheduler == "linear":
        scheduler = lr_scheduler.ChainedScheduler([
            lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=int(args.device_max_steps * args.warmup_proportion)),
            lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-9, total_iters=args.device_max_steps)
        ])

    elif args.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)


    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    
    if device != "cpu":
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )

        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

        return model, optimizer, scheduler, grad_scaler
    else:
        return model, optimizer, scheduler, None


def setup_training(args):
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()

        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        logging.info(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.")

        seed_everything(args.seed + rank)

        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        if rank == 0:
            logging.info(f"Group initialized? {torch.distributed.is_initialized()}")

        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logging.info(f"RCCL started on device {device}")
        logging.info(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

        if is_main_process():
            os.system(f"mkdir -p {args.output_path}")

        args.n_training_files = 1
        args.n_database_files = len(os.listdir(args.shard_path[:-6]))
        args.regular_shard_size = args.num_documents // args.n_database_files
        args.final_shard_size = args.num_documents - (get_world_size() - 1) * args.regular_shard_size
        if is_main_process():
            logging.info(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
            logging.info(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")
            logging.info(f"Found {args.n_training_files} training shards")
            logging.info(f"Found {args.n_database_files} database shards")
            logging.info(f"The regular size of a database shard is: {args.regular_shard_size}")
            logging.info(f"Final shard size: {args.final_shard_size}")

        args.device_max_steps = args.max_steps

        if is_main_process():
            wandb.init(
                name=args.wandb_name,
                config=args,
                id=args.wandb_id,
                project="nor_retriever",
                entity="nor-ret",
                resume="auto",
                mode=os.environ["WANDB_MODE"],
                allow_val_change=True,
                reinit=True
            )

        return device, local_rank
    else:
        return "cpu", 1

def realm_loss(marginal_logits, pred_logits, target, loss_fn, queries, tokenizer, batch_size, k):
    # [batch_size*k, seq_len]
    target = target.repeat(k, 1)

    # Count the number of words

    #num_mask = 0
    #mask_pos = []

    # [batch_size, seq_len]
    masked_token_ids = (torch.tensor(queries['input_ids']) == tokenizer.mask_token_id).to(device)
    num_mask = torch.sum(masked_token_ids)

    #for i in range(batch_size):
    #    masked_token_ids = torch.where(torch.tensor(queries['input_ids'][i]) == tokenizer.mask_token_id)[0]
    #    mask_pos.append(masked_token_ids)
    #    num_mask += len(masked_token_ids)

    # Calculate accuracy
    acc = accuracy(pred_logits, k, target, masked_token_ids=masked_token_ids, num_mask=num_mask)

    # Calculate the log probabilities of marginal logits

    # [batch_size * k, vocab_size, T]
    pred_logits = pred_logits.transpose(1,2)

    # [batch_size * k, seq_len]
    gold_preds_per_doc = -nn.functional.cross_entropy(pred_logits, target, reduction='none')

    # [batch_size, k, seq_len]
    gold_preds_per_doc = gold_preds_per_doc.view(k, batch_size, -1).transpose(0,1)

    # [batch_size, k]
    mar_log_prob = nn.functional.log_softmax(marginal_logits)

    # Calculate retrieval utility
    # RU, max_RU, mean_RU = retrieval_utility(-gold_preds_per_doc, k, masked_token_ids=masked_token_ids, num_mask=num_mask)

    # Calculate log(p(y|x))

    # [num_mask, k, vocab_size]
    #marginal_sum = torch.zeros(size=(num_mask, k, pred_log_prob.shape[-1]))
    # [batch_size, k, 1]
    mar_log_prob = mar_log_prob.unsqueeze(-1)

    #offset = 0
    #for i in range(batch_size):
    #    for masked_token in mask_pos[i]:
    #        marginal_sum[i+offset] = mar_log_prob[i] + pred_log_prob[i,:,masked_token]
    #        offset += 1
    #    offset -= 1

    # [batch_size, seq_len]
    gold_preds = torch.logsumexp(mar_log_prob + gold_preds_per_doc, 1)

    # [batch_size, seq_len]
    masked_gold_preds = gold_preds.masked_fill(~masked_token_ids, 0)

    RU, max_RU, mean_RU, real_RU = retrieval_utility(-gold_preds_per_doc, -masked_gold_preds, k, masked_token_ids=masked_token_ids, num_mask=num_mask)

    # []
    loss = -(masked_gold_preds.sum() / num_mask)

    # [num_mask, vocab_size]
    #marginal_pred = torch.logsumexp(marginal_sum, 1)

    # Calculate the negative log-likelihood

    # []
    #loss = loss_fn(marginal_pred, target)

    return loss, acc, RU, max_RU, mean_RU, real_RU



def create_train_dataloader(data, args, global_step, seed):
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=7 - 1,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True
    )
    return train_dataloader


def training_epoch(brent, masker, data, optimizer, loss_fn, scheduler, grad_scaler, global_step, epoch, args, device, max_local_steps, index_time=0, train_time=0):
    train_dataloader = create_train_dataloader(data, args, global_step, args.seed + get_rank() + epoch * get_world_size())

    brent = brent.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.device_max_steps)
        start = time.time()
    else:
        train_iter = train_dataloader

    for local_step, index in enumerate(train_iter):
        # optimizer.zero_grad()
        
        input_ids = [data.wiki[i]["input_ids"] for i in index]
        offsets = [data.wiki[i]["offset_mapping"] for i in index]
        text = [data.wiki[i]["text"] for i in index]
        attention_mask = [data.wiki[i]["attention_mask"] for i in index]

        # Document ids the queries come from
        query_ids = index

        #Targes and masking
        targets = torch.tensor(input_ids).to(device)
        qs = [masker(torch.tensor(i), t, o).numpy().tolist() for i, t, o in zip(input_ids, text, offsets)]

        # Postion of the first pad token in a query (list), None idicates no padding
        pad_ids = []
        for q in qs:
            try:
                i = q.index(0)
                pad_ids.append(i)
            except ValueError as e:
                pad_ids.append(None)

        queries = {
            "input_ids": qs,
            "attention_mask": attention_mask
        }
        
        # Train
        with torch.cuda.amp.autocast(args.mixed_precision):
            marginal_logits, pred_logits, retrieval_ids = brent(queries, query_ids, pad_ids, args.seq_length)
            loss, acc, RU, RU_max, RU_mean, RU_real = realm_loss(marginal_logits, pred_logits, targets, loss_fn, queries, brent.module.tokenizer, args.batch_size, brent.module.k+1)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(brent.parameters(), args.max_gradient)

        scheduler.step()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if is_main_process():
            train_iter.set_postfix_str(f"loss: {loss.item():.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")

            end = time.time()
            train_time += end - start

            if is_main_process():

                results_dict = {
                    "progress/epochs": epoch,
                    "progress/learning_rate": optimizer.param_groups[0]['lr'],
                    "info/train_time": train_time,
                    "info/index_time": index_time,
                    "train/loss": loss.item(),
                    "stats/RU_max": RU_max.item(),
                    "stats/RU_mean": RU_mean.item(),
                    "stats/RU": RU_real.item(),
                    "stats/perplexity": loss.exp().item(),
                }

                for i, accs in enumerate(acc):
                    results_dict[f"stats/accuracy_{i+1 if i < args.k-1 else '0'}"] =  accs.item()

                for i, rus in enumerate(RU):
                    results_dict[f"stats/RU_{i+1}"] = rus.item()

                for i, prob in enumerate(nn.functional.softmax(marginal_logits, dim=-1).mean(dim=0)):
                    results_dict[f"probs/Document{i+1 if i < args.k-1 else '0'}"] =  prob.item()

                wandb.log(results_dict, step=global_step)

            start = time.time()
                
        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            return global_step, train_time

    return global_step, train_time


def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_path}/model.bin"
    if is_main_process():
        if os.path.exists(checkpoint_path):
            os.rename(checkpoint_path, f"{checkpoint_path}_tmp")

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )

    def schedule_next_job(path):
        def signal_handler(sig, frame):
            if get_rank() == 0:
                print("SIGTERM detected, schedulling a new job and ending gracefully...")
                wandb.run.finish()
            if get_rank() == 1:
                os.system(f"sbatch --dependency=afterany:{os.getenv('SLURM_JOBID')} train_retriever.slurm --checkpoint_path {path}")
                exit()
        return signal_handler
    # signal.signal(signal.SIGTERM, schedule_next_job(checkpoint_path))

    return checkpoint_path


def save_step_model(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_path}/model_{global_step}.bin"
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "args": args,
            },
            checkpoint_path
        )
    return checkpoint_path



if __name__ == "__main__":
    args = parse_args()

    train_time = 0
    index_time = 0

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0
        args.wandb_id = wandb.util.generate_id() if int(os.environ["SLURM_PROCID"]) == 0 else 0

    #Setup training environment
    device, local_rank = setup_training(args)
    
    # Training corpus
    logging.info("Creating Dataset class...")
    data = MlmDataset(args.wiki_base_path)

    tokenizer = AutoTokenizer.from_pretrained(args.reader_model_name_or_path, local_files_only=True)

    # Build REALM model
    logging.info("Loading model...")
    m_init = REALM(
        query_embedder=args.reader_model_name_or_path,
        doc_embedder=args.reader_model_name_or_path,
        reader_model=args.reader_model_name_or_path,
        database=data.wiki,
        tokenizer=tokenizer,
        device=device,
        d=args.num_embbed_dimensions,
        k=args.k-1
    )

    brent, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, m_init, checkpoint, load_ict=args.use_ict)
    masker = Masker(tokenizer, mask_p=args.mask_p, random_factor=args.random_factor, salient_factor=args.salient_factor)

    # Set up loss and optimizer
    loss_fn = nn.NLLLoss()
    #optimizer = torch.optim.AdamW(brent.parameters(), lr=args.lr
    world_size = int(os.environ["WORLD_SIZE"])
    # Do initial indexing
    logging.info("Initial indexing, this might take a a couple of minutes")

    start = time.time()

    partial_embedded_vectors = embed(args.shard_path + str(get_rank()) + ".json", brent.module.doc, None, device)
    # Test
    print(f"Hi, I am GPU {get_rank()}, here is the length of the partial embedded vectors: {len(partial_embedded_vectors)}")
    # Convert to tensor and put on device
    partial_embedded_vectors = torch.tensor(partial_embedded_vectors, dtype=torch.float32).to(device)[:args.regular_shard_size]
    # Creating the full tensor
    # re_embedded_vectors = torch.zeros((args.num_document, args.num_embbed_dimensions), dtype=np.float32) # Real line
    if world_size <= args.n_database_files:
        re_embedded_vectors = [torch.zeros((args.regular_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device) for _ in range(world_size)] # test
    else:
        re_embedded_vectors = [torch.zeros((args.regular_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device) for _ in range(world_size-1)] # test
        re_embedded_vectors.append(torch.zeros((args.final_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device))
    # re_embedded_vectors = [torch.zeros((args.num_documents // args.n_gpu, args.num_embbed_dimensions), dtype=torch.float32).to(device) for _ in range(args.n_gpu-1)] # test
    # re_embedded_vectors += [torch.zeros(args.num_documents - (args.n_gpu  -1)* (args.num_documents // args.n_gpu), dtype=torch.float32)]
    # Gathering all the tensors together
    torch.distributed.all_gather(re_embedded_vectors, partial_embedded_vectors)
    # Changing back numpy array
    re_embedded_vectors = torch.cat(tuple(re_embedded_vectors), 0).cpu().numpy()
    # Remove the partial vectors from memory
    del partial_embedded_vectors
    # Do initial indexing
    brent.module.update_index(re_embedded_vectors)

    end = time.time()
    index_time += end - start

    # Test
    #print(f"Hi, I am GPU {get_rank()}, here is the length of the re-embedded vectors: {len(re_embedded_vectors)}")
    for epoch in count(initial_epoch):
        if epoch == 0:
            train_data, min_length = data, torch.tensor(max(500, args.re_indexing_frequency), dtype=torch.long, device=device)
        else:
            train_data, min_length = data, torch.tensor(args.re_indexing_frequency, dtype=torch.long, device=device)
        global_step, train_time = training_epoch(brent, masker, train_data, optimizer, loss_fn, scheduler, grad_scaler, global_step, epoch, args, device, min_length, index_time, train_time)
        logging.info(f"At global step: {global_step}")
        if global_step % 10000 == 0: #test
            logging.info(f"Saving model at step: {global_step}")
            path = save_step_model(brent, optimizer, grad_scaler, scheduler, global_step, epoch, args)

        checkpoint_path = save(brent, optimizer, grad_scaler, scheduler, global_step, epoch, args)

        start = time.time()

        # Memory management
        del re_embedded_vectors

        partial_embedded_vectors = embed(args.shard_path + str(get_rank()) + ".json", brent.module.doc, None, device)
        #print(f"Hi, I am GPU {get_rank()}, here is the length of the partial embedded vectors: {len(partial_embedded_vectors)}")
        # Convert to tensor and put on device
        partial_embedded_vectors = torch.tensor(partial_embedded_vectors, dtype=torch.float32).to(device)[:args.regular_shard_size]
        # Creating the full tensor
        # re_embedded_vectors = torch.zeros((args.num_document, args.num_embbed_dimensions), dtype=np.float32) # Real line
        if world_size <= args.n_database_files:
            re_embedded_vectors = [torch.zeros((args.regular_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device) for _ in range(world_size)] # test
        else:
            re_embedded_vectors = [torch.zeros((args.regular_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device) for _ in range(world_size-1)] # test
            re_embedded_vectors.append(torch.zeros((args.final_shard_size, args.num_embbed_dimensions), dtype=torch.float32).to(device))
        # Gathering all the tensors together
        torch.distributed.all_gather(re_embedded_vectors, partial_embedded_vectors)
        # Changing back numpy array
        re_embedded_vectors = torch.cat(tuple(re_embedded_vectors), 0).cpu().numpy()
        # Remove the partial vectors from memory
        del partial_embedded_vectors
        # Do initial indexing
        brent.module.update_index(re_embedded_vectors)

        end = time.time()
        index_time += end - start


        if global_step >= args.device_max_steps:
            break

    
