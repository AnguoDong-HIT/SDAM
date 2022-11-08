import spacy
import numpy as np

from create_aux_data import multi_hot, one_hot, pos_vocab_size, dep_vocab_size

nlp = spacy.load("en_core_web_sm")
txt = "I love the way the entire suite of software works together ."
nlp_doc = nlp(txt)

# for t in nlp_doc:
#     print("{0}({1}) <-- {2} -- {3}({4})".format(t.text, t.tag_, t.dep_, t.head.text, t.head.tag_))

# labels = nlp.get_pipe("tagger").labels
# with open("datasets/tag_vocab.txt", "w+", encoding="utf-8") as f:
#     for tag in labels:
#         print(tag)
#         f.write(tag + '\n')

def get_pd():
    with open("pd_rep", 'r') as f:
        lines = f.read().split('\n')
        pos_rep, dep_rep = lines[0], lines[1]
        pos_rep = pos_rep.strip('][').split(', ')
        dep_rep = dep_rep.strip('][').split(', ')

        pos_rep = np.array(pos_rep, dtype=np.float64)
        dep_rep = np.array(dep_rep, dtype=np.float64)
        return (pos_rep, dep_rep)

def cos_similarity(a, b):
    num = a.dot(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = num / denom
    return 0.5 + 0.5 * cos

def pd_similarity(pd_rep, nlp_doc):
    pos_rep, dep_rep = pd_rep[0], pd_rep[1]
    doc_pos = [t.tag_ for t in nlp_doc]
    pos_similarity = [cos_similarity(pos_rep, one_hot(token_pos, pos_vocab_size)) for token_pos in doc_pos]
    dep_similarity = []
    for token in nlp_doc:
        cur_deps = [token.dep_]
        cur_deps.extend([child.dep_ for child in token.children])
        dep_similarity.append(cos_similarity(dep_rep, multi_hot(cur_deps, dep_vocab_size)))
    return list(zip(pos_similarity, dep_similarity))

if __name__ == "__main__":
    pd_rep = get_pd()
    pd_s = pd_similarity(pd_rep, nlp_doc)
    pd_s = [(0., 0.)] + pd_s + [(0., 0.)]
    scores = [t[0] * t[1] for t in pd_s]
    idx_to_score = dict([(idx, score) for idx, score in enumerate(scores)])
    g = sorted(idx_to_score.items(), key=lambda d:d[1], reverse=True)
    print(g)


        

