import os
import numpy as np
import spacy
from tqdm import tqdm

with open("datasets/dep_vocab.txt", 'r', encoding="utf-8") as f:
    dep_vocab = f.read()
dep_vocab = dep_vocab.split("\n")
dep_to_idx = dict([(dep, i) for i, dep in enumerate(dep_vocab)])
dep_vocab_size = len(dep_vocab)

with open("datasets/tag_vocab.txt", 'r', encoding="utf-8") as f:
    pos_vocab = f.read()
pos_vocab = pos_vocab.split("\n")
pos_to_idx = dict([(pos, i) for i, pos in enumerate(pos_vocab)])
pos_vocab_size = len(pos_vocab)


# 输入一个pos_，只有自己的
def one_hot(x, n_class):
    onehot = [0.] * n_class
    if x not in pos_vocab:
        x = "[UNK]"
    onehot[pos_to_idx[x]] = 1
    return np.array(onehot)

# 输入一列dep_,包括父子
def multi_hot(x, n_class):
    multihot = [0.] * n_class
    for dep in x:
        if dep not in dep_vocab:
            dep = "[UNK]"
        multihot[dep_to_idx[dep]] = 1
    return np.array(multihot)

# print(one_hot(pos_vocab[5], len(pos_vocab)))
# print(multi_hot(dep_vocab[5:10], len(dep_vocab)))


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
            tag_label = tag_label.split()
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines

def get_aspect_pd():
    aspect_pos_rep = [0.] * pos_vocab_size
    aspect_dep_rep = [0.] * dep_vocab_size
    domains = ["device", "laptop", "rest"]
    aspect_amount = 0
    for domain in domains:
        for suffix in [".train.txt", ".test.txt"]:
            data_file = os.path.join("datasets", domain+suffix)
            lines = read_txt(data_file)
            for idx in tqdm(range(len(lines)), desc="reading lines from {}".format(data_file)):
                line = lines[idx]
                sentence, label = line["sentence"], line["label"]
                n_tag = len(label)
                nlp = spacy.load("en_core_web_sm")
                nlp_doc = nlp(sentence)
                for i in range(n_tag):
                    cur_ot_tag = label[i]
                    if 'T' in cur_ot_tag:
                        cur_tag = nlp_doc[i].tag_
                        cur_deps = [nlp_doc[i].dep_]
                        cur_deps.extend([child.dep_ for child in nlp_doc[i].children])
                        tag = one_hot(cur_tag, pos_vocab_size)
                        dep = multi_hot(cur_deps, dep_vocab_size)
                        aspect_pos_rep = [i + j for i, j in zip(aspect_pos_rep, tag)]
                        aspect_dep_rep = [i + j for i, j in zip(aspect_dep_rep, dep)]
                        aspect_amount += 1

    aspect_pos_rep = [i / aspect_amount for i in aspect_pos_rep]
    aspect_dep_rep = [i / aspect_amount for i in aspect_dep_rep]
    return aspect_pos_rep, aspect_dep_rep

if __name__ == '__main__':
    aspect_pos_rep, aspect_dep_rep = get_aspect_pd()
    print(aspect_pos_rep, aspect_dep_rep)

