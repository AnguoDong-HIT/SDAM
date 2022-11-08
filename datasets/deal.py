import sys
import os
import argparse
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from create_prototype_bank import *
import deal_pd
from random import randint, random


def read_txt(input_file):
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
    
def ot2bio(tag_sequence):
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


parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--bank_size', type=int, default=300)
args = parser.parse_args()
s = "laptop"
t = "rest"

lines = read_txt(f'datasets/{s}.train.txt')
pd_rep = get_aspect_pd(s)

single_bank_size = args.bank_size
multi_bank_size = args.bank_size
single_threshold = args.threshold
multi_gram_threshold = args.threshold
single_word_bank, multi_word_bank, multi_word_dict = get_bank(s, t, single_threshold, multi_gram_threshold, single_bank_size, multi_bank_size)

single_len = len(single_word_bank)
multi_len = len(multi_word_bank)

new_lines = []
new_labels = []
for idx in range(len(lines)):
    line = lines[idx]
    tokens = line["sentence"]
    label = line["label"]
    tags = ot2bio(label)
    ts_sequence = tag2aspect(tags)

    nlp_doc = nlp(" ".join(tokens))
    pd_s = pd_similarity(pd_rep, nlp_doc)

    for (beg, end) in ts_sequence:
        if beg == end:
            tokens[beg] = single_word_bank[randint(0, single_len-1)]
        elif multi_len != 0 and (end - beg + 1) in multi_word_dict.keys():
            phraze = multi_word_dict[end - beg + 1][randint(0, len(multi_word_dict[end - beg + 1])-1)]
            phraze = phraze.split(' ')
            for i in range(beg, end+1):
                tokens[i] = phraze[i-beg]
            # 替换多词,多词长度不一，还要更新tag和pd_s
            # pro = multi_word_bank[randint(0, multi_len-1)]
            # if len(pro) == end - beg + 1:
            #     for i in range(beg, end+1):
            #         tokens[i] = pro[i-beg]

    new_lines.append(tokens)
    new_labels.append(label)

with open(f'datasets/{args.threshold}-{args.bank_size}.txt', 'a+') as f:
    for tokens, label in zip(new_lines, new_labels):
        f.write(' '.join(tokens) + '***' + ' '.join(label) + '\n')


# s_d = ["laptop", "mams"]
# for s in s_d:
#     for t in s_d:
#         if (s==t):
#             continue
#         if (s=="laptop" and t=="mams"):
#             continue
#         if (s=="device" and t=="laptop"):
#             continue
        
        
#         lines = read_txt(f'datasets/{s}.train.txt')
#         pd_rep = get_aspect_pd(s)

#         single_bank_size = 300
#         multi_bank_size = 300
#         single_threshold = 0.3
#         multi_gram_threshold = 0.3
#         single_word_bank, multi_word_bank, multi_word_dict = get_bank(s, t, single_threshold, multi_gram_threshold, single_bank_size, multi_bank_size)
        
#         single_len = len(single_word_bank)
#         multi_len = len(multi_word_bank)

#         new_lines = []
#         new_labels = []
#         for idx in range(len(lines)):
#             line = lines[idx]
#             tokens = line["sentence"]
#             label = line["label"]
#             tags = ot2bio(label)
#             ts_sequence = tag2aspect(tags)

#             nlp_doc = nlp(" ".join(tokens))
#             pd_s = pd_similarity(pd_rep, nlp_doc)

#             for (beg, end) in ts_sequence:
#                 if beg == end:
#                     tokens[beg] = single_word_bank[randint(0, single_len-1)]
#                 elif multi_len != 0 and (end - beg + 1) in multi_word_dict.keys():
#                     phraze = multi_word_dict[end - beg + 1][randint(0, len(multi_word_dict[end - beg + 1])-1)]
#                     phraze = phraze.split(' ')
#                     for i in range(beg, end+1):
#                         tokens[i] = phraze[i-beg]
#                     # 替换多词,多词长度不一，还要更新tag和pd_s
#                     # pro = multi_word_bank[randint(0, multi_len-1)]
#                     # if len(pro) == end - beg + 1:
#                     #     for i in range(beg, end+1):
#                     #         tokens[i] = pro[i-beg]

#             new_lines.append(tokens)
#             new_labels.append(label)

#         with open(f'datasets/{s}-{t}.txt', 'a+') as f:
#             for tokens, label in zip(new_lines, new_labels):
#                 f.write(' '.join(tokens) + '***' + ' '.join(label) + '\n')
        
        