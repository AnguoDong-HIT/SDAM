from threading import main_thread
from model.bert import bert_ATE, bert_ABSA
from datasets.data_utils import da_dataset_ATE_txt
from torch.utils.data import DataLoader
import os
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time
import numpy as np


DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# pretrain_model_name = "pretrained/ep2"
# tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
# model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
# optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
# model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
# optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)

def evl_time(t):
    min, sec= divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
    return model
    
def save_model(model, name):
    torch.save(model.state_dict(), name)

def create_mini_batch(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    tags_tensors = [s[2] for s in samples]
    tags_tensors = pad_sequence(tags_tensors, batch_first=True)

    # pols_tensors = [s[3] for s in samples]
    # pols_tensors = pad_sequence(pols_tensors, batch_first=True)

    pd_s_tensors = [s[4] for s in samples]
    pd_s_tensors = pad_sequence(pd_s_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    
    return ids_tensors, tags_tensors, None, masks_tensors, pd_s_tensors


def train_model_ATE(loader, model, epochs, args, test_loader):
    weight_decay = 1e-2
    epsilon = 1e-8
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay' : weight_decay
        },
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay' : 0.0
        }
    ]
    optimizer_ATE = torch.optim.AdamW(optimizer_grouped_parameters, lr = lr, eps = epsilon)

    # optimizer_ATE = torch.optim.Adam(model.parameters(), lr=lr)
    all_data = len(loader)
    F1_set = []
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []
        correct_predictions = 0
        
        for data in loader:
            t0 = time.time()
            ids_tensors, tags_tensors, _, masks_tensors, pd_s = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)
            pd_s = pd_s.to(DEVICE)

            loss = model(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors, pd_s=pd_s)
            losses.append(loss.item())
            loss.backward()
            optimizer_ATE.step()
            optimizer_ATE.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         

        if test_loader is not None:
            precision, recall, F1 = evaluate_ate(test_loader, model)
            F1_set.append([epoch, F1])
            out = "source: {} target: {} batch_size: {} epoch: {} da_epochs: {} threshold: {} bank_size: {}\n precision: {}, recall: {}, F1: {} \n".format(args.source_domain, args.target_domain, args.batch_size, epoch, args.da_epochs, args.threshold, args.bank_size, precision, recall, F1)
            with open(args.out_dir, "a+", encoding="utf-8") as f:
                f.write(out)
    if test_loader is not None:
        with open(args.out_dir, "a+", encoding="utf-8") as f:
            maxf = max(F1_set, key=lambda x:x[1])
            f.write(f"\nmax F1:{maxf[1]}  epoch:{maxf[0]} \n")
    # save_model(model, 'bert_ATE_{}.pkl'.format(epochs))

def da_train_model_ATE(args, tokenizer, model):
    optimizer_ATE = torch.optim.Adam(model.parameters(), lr=lr)
    all_data = 0
    for epoch in range(args.epochs):
        finish_data = 0
        losses = []
        current_times = []
        correct_predictions = 0

        da_ds = da_dataset_ATE_txt(os.path.join("datasets", args.source_domain+".train.txt"), tokenizer, args.source_domain, args.target_domain, args.threshold, args.bank_size, args.prob)
        da_loader = DataLoader(da_ds, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle = True)
        all_data = len(da_loader)

        for data in da_loader:
            t0 = time.time()
            ids_tensors, tags_tensors, _, masks_tensors, pd_s = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)
            pd_s = pd_s.to(DEVICE)

            loss = model(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors, pd_s=pd_s)
            losses.append(loss.item())
            if np.mean(losses) < 0.001:
                return
            loss.backward()
            optimizer_ATE.step()
            optimizer_ATE.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(args.epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         



def tag2aspect(tag_sequence):
    """
    convert BIO tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIO tagging schema
    :return:
    """
    ts_sequence = []
    beg = -1
    for index, ts_tag in enumerate(tag_sequence):
        # 如果本tag为O且前面有个B，把B到O-1加进去
        if ts_tag == 0:
            if beg != -1:
                ts_sequence.append((beg, index-1))
                beg = -1
        else:
            cur = ts_tag
            # 如果本tag为B，把begin置为B。如果是I就继续
            if cur == 1:
                if beg != -1:
                    ts_sequence.append((beg, index-1))
                beg = index

    if beg != -1:
        ts_sequence.append((beg, index))
    return ts_sequence

def match(pred, gold):
    true_count = 0
    for t in pred:
        if t in gold:
            true_count += 1
    return true_count

def evaluate_ate(loader, model):
    TP, FN, FP = 0, 0, 0
    with torch.no_grad():
        for data in loader:

            ids_tensors, tags_tensors, _, masks_tensors, pd_s = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)      # batch_size, seq_len
            masks_tensors = masks_tensors.to(DEVICE)

            outputs = model(ids_tensors=ids_tensors, tags_tensors=None, masks_tensors=masks_tensors)

            _, predictions = torch.max(outputs, dim=2)

            for i in range(len(tags_tensors)):
                gold = tags_tensors[i].tolist()
                pred = predictions[i].tolist()
                gold_aspects = tag2aspect(gold)
                pred_aspects = tag2aspect(pred)
                n_hit = match(pred=pred_aspects, gold=gold_aspects)
                TP += n_hit
                FP += (len(pred_aspects) - n_hit)
                FN += (len(gold_aspects) - n_hit)
    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.0001)
    F1 = 2 * precision * recall / (precision + recall + 0.00001)
    return precision, recall, F1

