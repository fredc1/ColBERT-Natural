import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AdamW
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress #, manage_checkpoints



def train(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    if args.lazy:
        reader = LazyBatcher(args)
    else:
        reader = EagerBatcher(args)

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    colbert = colbert.to(DEVICE)
    colbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    running_train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in tqdm(zip(range(start_batch_idx, args.maxsteps), reader)):
        this_batch_loss = 0.0
        last_scores = None
        for queries, passages in BatchSteps:
            with amp.context():
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps
                last_scores = scores

            amp.backward(loss)

            running_train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        num_examples_seen = (batch_idx - start_batch_idx) * args.bsize
        elapsed = float(time.time() - start_time)

        if batch_idx % 50 == 0:
            print_progress(last_scores, running_train_loss/50, batch_idx, colbert, args.bsize)
            running_train_loss = 0.0
            # print(f"Avg loss so far: {avg_loss}  Batch# {batch_idx + 1}")
        # print_message(batch_idx, avg_loss)
        # manage_checkpoints(args, colbert, optimizer, batch_idx+1)
    