import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from torch.utils.data import Dataset
import torch
from deal_pd import *
from create_prototype_bank import *
from random import randint, random

# print("**************loading bank****************")
# laptop2rest_single_word_bank, laptop2rest_multi_word_bank = get_bank("laptop", "rest")
# laptop2mams_single_word_bank, laptop2mams_multi_word_bank = get_bank("laptop", "mams")
# rest2laptop_single_word_bank, rest2laptop_multi_word_bank = get_bank("rest", "laptop")
# mams2laptop_single_word_bank, mams2laptop_multi_word_bank = get_bank("mams", "laptop")

# def bank(source_domain, target_domain):
#     if source_domain == "laptop" and target_domain == "rest":
#         return laptop2rest_single_word_bank, laptop2rest_multi_word_bank
#     elif source_domain == "laptop" and target_domain == "mams":
#         return laptop2mams_single_word_bank, laptop2mams_multi_word_bank
#     elif source_domain == "rest" and target_domain == "laptop":
#         return rest2laptop_single_word_bank, rest2laptop_multi_word_bank
#     else :
#         return mams2laptop_single_word_bank, mams2laptop_multi_word_bank


class dataset_ATE_txt(Dataset):

    def __init__(self, input_file, tokenizer, source_domain):
        self.lines = self.read_txt(input_file)
        self.tokenizer = tokenizer
        self.pd_rep = get_aspect_pd(source_domain)

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            try:
                sentence, label, tag_label = t.split('***')
            except:
                sentence, label = t.split('***')
                tag_label = label
            label = label.split()
            sentence = sentence.lower().split()
            tag_label = tag_label.split()
            assert len(label) == len(sentence) == len(tag_label), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines
    
    def ot2bio(self, tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = tag_sequence[i]
            if '-' not in cur_ts_tag:
                new_ts_sequence.append(0)
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append(2) 
                else:
                    new_ts_sequence.append(1)
            prev_pos = cur_pos
        return new_ts_sequence
    
    def __getitem__(self, idx):
        tokens = self.lines[idx]["sentence"]
        label = self.lines[idx]["label"]
        tags = self.ot2bio(label)

        nlp_doc = nlp(" ".join(tokens))
        pd_s = pd_similarity(self.pd_rep, nlp_doc)

        bert_tokens = []
        bert_tags = []
        bert_pd_s = []
        for i in range(len(tokens)):
            # 分词
            t = self.tokenizer.tokenize(tokens[i])
            # 分开的每个词标签相同
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pd_s += [float(pd_s[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pd_s = torch.tensor(bert_pd_s)

        return bert_tokens, ids_tensor, tags_tensor, None, pd_s
    
    def __len__(self):
        return len(self.lines)



class da_dataset_ATE_txt(Dataset):

    def __init__(self, input_file, tokenizer, source_domain, target_domain, threshold, bank_size, prob):
        self.lines = self.read_txt(input_file)
        self.tokenizer = tokenizer
        self.pd_rep = get_aspect_pd(source_domain)

        single_word_bank, multi_word_bank, multi_word_dict = get_bank(source_domain, target_domain, 0.3, 0.3, 300, 300)
        
        single_len = len(single_word_bank)
        multi_len = len(multi_word_bank)

        self.single_word_bank = single_word_bank
        self.multi_word_bank = multi_word_bank
        self.multi_word_dict = multi_word_dict
        self.single_len = single_len
        self.multi_len = multi_len
        self.prob = prob

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            try:
                sentence, label, tag_label = t.split('***')
            except:
                sentence, label = t.split('***')
                tag_label = label
            label = label.split()
            sentence = sentence.lower().split()
            tag_label = tag_label.split()
            assert len(label) == len(sentence) == len(tag_label), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines
    
    def ot2bio(self, tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = tag_sequence[i]
            if '-' not in cur_ts_tag:
                new_ts_sequence.append(0)
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append(2) 
                else:
                    new_ts_sequence.append(1)
            prev_pos = cur_pos
        return new_ts_sequence

    def tag2aspect(self, tag_sequence):
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
    
    def __getitem__(self, idx):
        tokens = self.lines[idx]["sentence"]
        label = self.lines[idx]["label"]
        tags = self.ot2bio(label)
        ts_sequence = self.tag2aspect(tags)

        nlp_doc = nlp(" ".join(tokens))
        pd_s = pd_similarity(self.pd_rep, nlp_doc)

        bert_tokens = []
        bert_tags = []
        bert_pd_s = []

        for (beg, end) in ts_sequence:
            if random() < self.prob:
                if beg == end:
                    tokens[beg] = self.single_word_bank[randint(0, self.single_len-1)]
                elif self.multi_len != 0 and (end - beg + 1) in self.multi_word_dict.keys():
                    phraze = self.multi_word_dict[end - beg + 1][randint(0, len(self.multi_word_dict[end - beg + 1])-1)]
                    phraze = phraze.split(' ')
                    for i in range(beg, end+1):
                        tokens[i] = phraze[i-beg]

        for i in range(len(tokens)):
            # 分词
            t = self.tokenizer.tokenize(tokens[i])
            # 分开的每个词标签相同
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pd_s += [float(pd_s[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pd_s = torch.tensor(bert_pd_s)

        return bert_tokens, ids_tensor, tags_tensor, None, pd_s
    
    def __len__(self):
        return len(self.lines)



class dataset_ABSA_txt(Dataset):

    def __init__(self, input_file, tokenizer, source_domain):
        self.lines = self.read_txt(input_file)
        self.tokenizer = tokenizer
        self.pd_rep = get_aspect_pd(source_domain)

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            try:
                sentence, label, tag_label = t.split('***')
            except:
                sentence, label = t.split('***')
                tag_label = label
            label = label.split()
            sentence = sentence.lower().split()
            tag_label = tag_label.split()
            assert len(label) == len(sentence) == len(tag_label), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines
    
    def ot2bio(self, tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = tag_sequence[i]
            if '-' not in cur_ts_tag:
                new_ts_sequence.append(0)
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    if cur_sentiment == 'POS' :
                        new_ts_sequence.append(2)
                    elif cur_sentiment == 'NEU' :
                        new_ts_sequence.append(4)
                    elif cur_sentiment == 'NEG' :
                        new_ts_sequence.append(6)
                    # prev_pos is T
                    # new_ts_sequence.append(2) 
                else:
                    if cur_sentiment == 'POS' :
                        new_ts_sequence.append(1)
                    elif cur_sentiment == 'NEU' :
                        new_ts_sequence.append(3)
                    elif cur_sentiment == 'NEG' :
                        new_ts_sequence.append(5)
            prev_pos = cur_pos
        return new_ts_sequence
    
    def __getitem__(self, idx):
        tokens = self.lines[idx]["sentence"]
        label = self.lines[idx]["label"]
        tags = self.ot2bio(label)

        nlp_doc = nlp(" ".join(tokens))
        pd_s = pd_similarity(self.pd_rep, nlp_doc)

        bert_tokens = []
        bert_tags = []
        bert_pd_s = []
        for i in range(len(tokens)):
            # 分词
            t = self.tokenizer.tokenize(tokens[i])
            # 分开的每个词标签相同
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pd_s += [float(pd_s[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pd_s = torch.tensor(bert_pd_s)

        return bert_tokens, ids_tensor, tags_tensor, None, pd_s
    
    def __len__(self):
        return len(self.lines)



class da_dataset_ABSA_txt(Dataset):

    def __init__(self, input_file, tokenizer, source_domain, target_domain, threshold, bank_size, prob):
        self.lines = self.read_txt(input_file)
        self.tokenizer = tokenizer
        self.pd_rep = get_aspect_pd(source_domain)

        single_word_bank, multi_word_bank, multi_word_dict = get_bank(source_domain, target_domain, 0.3, 0.3, 300, 300)
        
        single_len = len(single_word_bank)
        multi_len = len(multi_word_bank)

        self.single_word_bank = single_word_bank
        self.multi_word_bank = multi_word_bank
        self.multi_word_dict = multi_word_dict
        self.single_len = single_len
        self.multi_len = multi_len
        self.prob = prob

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            try:
                sentence, label, tag_label = t.split('***')
            except:
                sentence, label = t.split('***')
                tag_label = label
            label = label.split()
            sentence = sentence.lower().split()
            tag_label = tag_label.split()
            assert len(label) == len(sentence) == len(tag_label), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines
    
    def ot2bio(self, tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = tag_sequence[i]
            if '-' not in cur_ts_tag:
                new_ts_sequence.append(0)
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    if cur_sentiment == 'POS' :
                        new_ts_sequence.append(2)
                    elif cur_sentiment == 'NEU' :
                        new_ts_sequence.append(4)
                    elif cur_sentiment == 'NEG' :
                        new_ts_sequence.append(6)
                    # prev_pos is T
                    # new_ts_sequence.append(2) 
                else:
                    if cur_sentiment == 'POS' :
                        new_ts_sequence.append(1)
                    elif cur_sentiment == 'NEU' :
                        new_ts_sequence.append(3)
                    elif cur_sentiment == 'NEG' :
                        new_ts_sequence.append(5)
            prev_pos = cur_pos
        return new_ts_sequence

    def tag2aspect(self, tag_sequence):
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
                    ts_sequence.append((beg, index-1, tag_sequence[beg]))
                    beg = -1
            else:
                cur = ts_tag
                # 如果本tag为B，把begin置为B。如果是I就继续
                if cur == 1 or cur == 3 or cur == 5:
                    if beg != -1:
                        ts_sequence.append((beg, index-1, tag_sequence[beg]))
                    beg = index

        if beg != -1:
            ts_sequence.append((beg, index, tag_sequence[beg]))
        return ts_sequence
    
    def __getitem__(self, idx):
        tokens = self.lines[idx]["sentence"]
        label = self.lines[idx]["label"]
        tags = self.ot2bio(label)
        ts_sequence = self.tag2aspect(tags)

        nlp_doc = nlp(" ".join(tokens))
        pd_s = pd_similarity(self.pd_rep, nlp_doc)

        bert_tokens = []
        bert_tags = []
        bert_pd_s = []

        for (beg, end, sen) in ts_sequence:
            if random() < self.prob:
                if beg == end:
                    tokens[beg] = self.single_word_bank[randint(0, self.single_len-1)]
                elif self.multi_len != 0 and (end - beg + 1) in self.multi_word_dict.keys():
                    phraze = self.multi_word_dict[end - beg + 1][randint(0, len(self.multi_word_dict[end - beg + 1])-1)]
                    phraze = phraze.split(' ')
                    for i in range(beg, end+1):
                        tokens[i] = phraze[i-beg]

        for i in range(len(tokens)):
            # 分词
            t = self.tokenizer.tokenize(tokens[i])
            # 分开的每个词标签相同
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)
            bert_pd_s += [float(pd_s[i])]*len(t)
        
        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)
        pd_s = torch.tensor(bert_pd_s)

        return bert_tokens, ids_tensor, tags_tensor, None, pd_s
    
    def __len__(self):
        return len(self.lines)