import numpy as np
import math
from pytorch_transformers.tokenization_bert import BertTokenizer
import spacy
from tqdm import tqdm
import os
import collections

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

nlp = spacy.load("en_core_web_sm")

# 输入一个pos_，只有自己的
def one_hot(x, n_class):
    onehot = [0.] * n_class
    onehot[pos_to_idx[x]] = 1
    return np.array(onehot)

# 输入一列dep_,包括父子
def multi_hot(x, n_class):
    multihot = [0.] * n_class
    for dep in x:
        multihot[dep_to_idx[dep]] = 1
    return np.array(multihot)

# def get_pd():
#     with open("data/pd_rep.txt", 'r') as f:
#         lines = f.read().split('\n')
#         pos_rep, dep_rep = lines[0], lines[1]
#         pos_rep = pos_rep.strip('][').split(', ')
#         dep_rep = dep_rep.strip('][').split(', ')

#         pos_rep = np.array(pos_rep, dtype=np.float64)
#         dep_rep = np.array(dep_rep, dtype=np.float64)
#         return (pos_rep, dep_rep)

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

def get_percentage(domain):
    NN = [0.] * pos_vocab_size
    NN[pos_to_idx['NN']] = 1
    rel = [0.] * dep_vocab_size
    rel[dep_to_idx['det']] = 1
    rel[dep_to_idx['nsubj']] = 1

    aspect_amount = 0
    instance_amount = 0
    instances = []

    for suffix in [".train.txt", ".test.txt"]:
        data_file = os.path.join("datasets", domain+suffix)
        lines = read_txt(data_file)
        # for idx in tqdm(range(len(lines)), desc="reading lines from {}".format(data_file)):
        for idx in range(len(lines)):
            line = lines[idx]
            sentence, label = line["sentence"], line["label"]
            n_tag = len(label)
            nlp_doc = nlp(sentence)
            for i in range(n_tag):
                cur_ot_tag = label[i]
                if '-' in cur_ot_tag:
                    cur_tag = nlp_doc[i].tag_
                    cur_deps = [nlp_doc[i].dep_]
                    cur_deps.extend([child.dep_ for child in nlp_doc[i].children])
                    tag = one_hot(cur_tag, pos_vocab_size)
                    dep = multi_hot(cur_deps, dep_vocab_size)
                    if (tag.tolist() == NN and dep.tolist() == rel) :
                        instance_amount += 1
                        instances.append(nlp_doc[i].text)
                    aspect_amount += 1

    # return (instance_amount+0.0001) / (aspect_amount+0.0001)
    instances = collections.Counter(instances)
    instances = instances.most_common(10)
    return instances


def get_aspect_pd(domain):
    aspect_pos_rep = [0.] * pos_vocab_size
    aspect_dep_rep = [0.] * dep_vocab_size
    # domains = ["laptop", "rest", "device", "service"]
    aspect_amount = 0

    for suffix in [".train.txt", ".test.txt"]:
        data_file = os.path.join("datasets", domain+suffix)
        lines = read_txt(data_file)
        # for idx in tqdm(range(len(lines)), desc="reading lines from {}".format(data_file)):
        for idx in range(len(lines)):
            line = lines[idx]
            sentence, label = line["sentence"], line["label"]
            n_tag = len(label)
            nlp_doc = nlp(sentence)
            for i in range(n_tag):
                cur_ot_tag = label[i]
                if '-' in cur_ot_tag:
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
    aspect_pos_rep = np.array(aspect_pos_rep, dtype=np.float64)
    aspect_dep_rep = np.array(aspect_dep_rep, dtype=np.float64)
    return (aspect_pos_rep, aspect_dep_rep)

def cos_similarity(a, b):
    num = a.dot(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if math.isclose(denom, 0, rel_tol=1e-5):
        return 0
    cos = num / denom
    return cos

def pd_similarity(pd_rep, nlp_doc):
    pos_rep, dep_rep = pd_rep[0], pd_rep[1]
    doc_pos = [t.tag_ for t in nlp_doc]
    pos_similarity = [cos_similarity(pos_rep, one_hot(token_pos, pos_vocab_size)) for token_pos in doc_pos]
    dep_similarity = []
    for token in nlp_doc:
        cur_deps = [token.dep_]
        cur_deps.extend([child.dep_ for child in token.children])
        cur_deps = [dep for dep in cur_deps if dep !='']
        # print("tokens: {}, deps: {}".format(token, cur_deps))
        dep_similarity.append(cos_similarity(dep_rep, multi_hot(cur_deps, dep_vocab_size)))
    # return list(zip(pos_similarity, dep_similarity))
    return [pos * dep for pos, dep in zip(pos_similarity, dep_similarity)]

if __name__ == "__main__":
    domains = ["rest", "laptop", "mams", "service", "device"]
    for domain in domains:
        print(f'{domain}: {get_percentage(domain)}')

    # for idx, source_domain in enumerate(domains):
    #     for jdx, target_domain in enumerate(domains):
    #         if idx == jdx:
    #             continue
    #         print("相似度between{} and {}".format(source_domain, target_domain), cos_similarity(get_aspect_pd(source_domain)[0], get_aspect_pd(target_domain)[0])
    #                                                                                         * cos_similarity(get_aspect_pd(source_domain)[1], get_aspect_pd(target_domain)[1]))