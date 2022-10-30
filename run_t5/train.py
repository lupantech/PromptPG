import re
import os
import sys
import time
import json
import random
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils
from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print("GPU device:", torch.cuda.device_count())  # Check how many CUDA capable devices you have, 2)

unifiedqa_models = {
    "small": "allenai/unifiedqa-t5-small",  # 231M
    "base": "allenai/unifiedqa-t5-base",  # 850M
    "large": "allenai/unifiedqa-t5-large",  # 2.7G
    "3b": "allenai/unifiedqa-t5-3b",  # 10.6G
    "11b": "allenai/unifiedqa-t5-11b",  # 42.1G
}


def evaluate(model, dataloader, tokenizer, device, args):

    # model.eval()

    eval_loss = 0

    for pids, batch in iter(dataloader):
        input_strings = batch["input_strings"]
        output_strings = batch["output_strings"]

        input_ids = tokenizer(input_strings,
                              padding="longest",
                              return_tensors="pt",
                              truncation=True,
                              max_length=args.max_input_length).input_ids.to(device)

        output_ids = tokenizer(output_strings,
                               padding="longest",
                               return_tensors="pt",
                               truncation=True,
                               max_length=args.max_output_length).input_ids.to(device)

        ## loss
        outputs = model(input_ids=input_ids, labels=output_ids)
        loss = outputs.loss * outputs.logits.shape[0]
        eval_loss += loss.item()

    eval_loss /= len(dataloader.dataset)

    return eval_loss


def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--save_all', action='store_true', help='save all model checkpoints or not')
    # input and output
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output', type=str, default='../saved_models/unifinedqa')
    parser.add_argument('--train_split', type=str, default='train', choices=['train'])
    parser.add_argument('--val_split', type=str, default='dev', choices=['dev', 'dev1k'])
    parser.add_argument('--quick_check', action='store_true')
    parser.add_argument('--check_point', type=str, default=None)
    # model
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='t5-small')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_input_length', type=int, default=200)
    parser.add_argument('--max_output_length', type=int, default=100)
    args = parser.parse_args()

    if args.quick_check:
        args.epochs = 1
        args.model = 't5-small'
        args.train_split = 'dev1k'
        args.val_split = 'dev1k'

    # print and save the args
    args.output = os.path.join(args.output, args.label)
    utils.create_dir(args.output)
    logger = utils.Logger(args.output + '/args.txt')

    print('====Input Arguments====')
    logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    ## [1] Define the model
    model_name = args.model
    if model_name in unifiedqa_models:
        model_name = unifiedqa_models[model_name]  # Unifiedqa models
    print("# model:", model_name)
    print("# Loading the model ...")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print('\n# model parameters:', sum(param.numel() for param in model.parameters()))

    if args.check_point != None:
        model_path = os.path.join(args.output, args.check_point)
        print(f"# Loading the check point: {model_path}")
        model.load_state_dict(torch.load(model_path))

    ## [2] Data loader
    train_dset = TMQADataset(args.data_root, args.train_split, tokenizer, args)
    eval_dset = TMQADataset(args.data_root, args.val_split, tokenizer, args)

    train_dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dset, batch_size=args.eval_batch_size, shuffle=False)

    ## GPU
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    model.to(device)

    ## [3] Train a model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_loss = 100000
    best_epoch = 0

    print("\nStart training: ")
    model.train()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        total_loss = 0
        t = time.time()

        for pids, batch in iter(train_dataloader):
            input_strings = batch["input_strings"]
            output_strings = batch["output_strings"]

            input_ids = tokenizer(input_strings,
                                  padding="longest",
                                  return_tensors="pt",
                                  truncation=True,
                                  max_length=args.max_input_length).input_ids.to(device)

            output_ids = tokenizer(output_strings,
                                   padding="longest",
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=args.max_output_length).input_ids.to(device)

            # print(input_ids.shape)
            # print(output_ids.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=input_ids, labels=output_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * outputs.logits.shape[0]

        total_loss /= len(train_dataloader.dataset)

        # evaluation
        model.train(False)
        eval_loss = evaluate(model, eval_dataloader, tokenizer, device, args)
        model.train(True)

        # save the model
        if eval_loss < best_eval_loss:
            model_path = os.path.join(args.output, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_loss = eval_loss
            best_epoch = epoch

        if args.save_all:
            model_path = os.path.join(args.output, 'model_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path)

        # print results
        logger.write('epoch %d, time: %.2f\t' % (epoch, time.time() - t) + 'train_loss: %.3f\t' % (total_loss) +
                     'val_loss:  %.3f\t' % (eval_loss))

    logger.write('\tBEST evaluation loss: %.3f @ %d' % (best_eval_loss, best_epoch))
