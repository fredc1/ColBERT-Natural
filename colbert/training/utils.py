import os
import torch

from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.evaluation.mrr import NQ_Validator

def print_progress(scores, running_loss, batch, model, args_bsize):
    validator = NQ_Validator(model, "/workspace/ColBERT-Natural/data/nq-dev-all.jsonl", args_bsize)
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print(f"Batch: {batch} AvgScore(+): {positive_avg} AvgScore(-): {negative_avg} Delta: {positive_avg - negative_avg} Running Avg Loss: {running_loss}, Validation MRR: {validator.get_mrr()}")
    #assert 0
'''
def manage_checkpoints(args, colbert, optimizer, batch_idx):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % 2000 == 0:
        name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
'''