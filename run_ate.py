import argparse
from pathlib import Path

from train import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
sys.path.insert(0, "data")
sys.path.insert(0, "datasets")

from dataset import dataset_ATE
from data_utils import dataset_ATE_txt, da_dataset_ATE_txt
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random

# laptops_train_ds = dataset_ATE(pd.read_csv("data/laptops_train.csv"), tokenizer)
# laptops_test_ds = dataset_ATE(pd.read_csv("data/laptops_test.csv"), tokenizer)
# restaurants_train_ds = dataset_ATE(pd.read_csv("data/restaurants_train.csv"), tokenizer)
# restaurants_test_ds = dataset_ATE(pd.read_csv("data/restaurants_test.csv"), tokenizer)
# twitter_train_ds = dataset_ATE(pd.read_csv("data/twitter_train.csv"), tokenizer)
# twitter_test_ds = dataset_ATE(pd.read_csv("data/twitter_test.csv"), tokenizer)

# device_train_ds = dataset_ATE_txt("datasets/device.train.txt", tokenizer)
# device_test_ds = dataset_ATE_txt("datasets/device.test.txt", tokenizer)
# service_train_ds = dataset_ATE_txt("datasets/service.train.txt", tokenizer)
# service_test_ds = dataset_ATE_txt("datasets/service.test.txt", tokenizer)

# train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
# test_ds = ConcatDataset([laptops_test_ds, restaurants_test_ds, twitter_test_ds])

if __name__ == '__main__':


    domains = ["laptop", "rest", "mams"]
    arg_parser = argparse.ArgumentParser(description='Domain adaptation')
    arg_parser.add_argument('--batch_size', type=int, default=8)
    arg_parser.add_argument('--da_epochs', type=int, default=1)
    arg_parser.add_argument('--threshold', type=float, default=0.3)
    arg_parser.add_argument('--bank_size', type=int, default=300)
    arg_parser.add_argument('--prob', type=float, default=1)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--source_domain', type=str, required=True)
    arg_parser.add_argument('--target_domain', type=str, required=True)
    arg_parser.add_argument('--bert_model', type=str, required=True)
    arg_parser.add_argument('--out_dir', type=Path, required=True)
    arg_parser.add_argument('--pd_loss', type=bool, default=False)
    arg_parser.add_argument('--da_first', action="store_true")
    args = arg_parser.parse_args()

    print("***************loading pd_rep**********************")
    source_domain = args.source_domain
    target_domain = args.target_domain
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    source_ds = dataset_ATE_txt(os.path.join("datasets", source_domain+".train.txt"), tokenizer, source_domain)
    target_ds = dataset_ATE_txt(os.path.join("datasets", target_domain+".test.txt"), tokenizer, source_domain)

    train_loader = DataLoader(source_ds, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle = True)
    test_loader = DataLoader(target_ds, batch_size=50, collate_fn=create_mini_batch, shuffle = True)

    da_source_ds = da_dataset_ATE_txt(os.path.join("datasets", source_domain+".train.txt"), tokenizer, source_domain, target_domain, args.threshold, args.bank_size, args.prob)
    da_train_loader = DataLoader(da_source_ds, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle = True)

    model_ATE = bert_ATE(args.bert_model, args.pd_loss).to(DEVICE)


    print("***************Start training**********************")
    if args.da_first:
        train_model_ATE(da_train_loader, model_ATE, args.da_epochs, args, None)
        train_model_ATE(train_loader, model_ATE, args.epochs, args, test_loader)
    else:
        train_model_ATE(train_loader, model_ATE, args.epochs, args, test_loader)
        train_model_ATE(da_train_loader, model_ATE, args.da_epochs, args, None)

    # precision, recall, F1 = evaluate_ate(test_loader, model_ATE)

    # out = "source: {} target: {} batch_size: {} epochs: {} da_epochs: {} threshold: {} bank_size: {}\n precision: {}, recall: {}, F1: {} \n".format(source_domain, target_domain, args.batch_size, args.epochs, args.da_epochs, args.threshold, args.bank_size, precision, recall, F1)

    # with open(args.out_dir, "a+", encoding="utf-8") as f:
    #     f.write(out)

    # print(out)
