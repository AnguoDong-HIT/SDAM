import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from argparse import ArgumentParser
from pathlib import Path
import spacy
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool

from random import random, randrange, randint, shuffle, choice

from create_aux_data import multi_hot, one_hot, pos_vocab_size, dep_vocab_size
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import json
import collections
import math

class SubTokenizer(BertTokenizer):
    '''
    bert toknizer： only do sub word tokenizer here
    '''
    def subword_tokenize(self, tokens, pd_s):  # for tag split
        # input : tokens list, pd_s list; output: token list, pd_s list, idx_map;
        split_tokens, split_pd_s = [], []
        # idx_map是原idx在分词后的顺序
        idx_map = []
        first_token_map = dict()
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                split_pd_s.append(pd_s[ix])  # the subwords share same pd_s
                idx_map.append(ix)
        return split_tokens, split_pd_s, idx_map

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
            self.c_ss = None
        else:
            self.documents = []
            self.c_ss = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document, c_s):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
            self.c_ss.append(c_s)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item], self.c_ss[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def truncate_seq(tokens, pd_s, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break

        trunc_tokens, trunc_pd_s = tokens, pd_s

        # 在列表末尾删除单词
        trunc_tokens.pop()
        trunc_pd_s.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def get_pd():
    with open("pd_rep_mams", 'r') as f:
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
    if math.isclose(denom, 0, rel_tol=1e-5):
        return 0
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
        cur_deps = [dep for dep in cur_deps if dep !='']
        # print("tokens: {}, deps: {}".format(token, cur_deps))
        dep_similarity.append(cos_similarity(dep_rep, multi_hot(cur_deps, dep_vocab_size)))
    return list(zip(pos_similarity, dep_similarity))
    
#Done 利用pd_similarity来mask词, pd_s是（p_s, d_s）列表
def create_masked_lm_predictions(tokens, pd_s, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    
    # 先来个简单的相乘
    scores = [t[0] * t[1] for t in pd_s]
    idx_to_score = dict([(idx, score) for idx, score in enumerate(scores)])
    sorted_scores = sorted(idx_to_score.items(), key=lambda d:d[1], reverse=True)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if (sorted_scores[num_to_mask-1][1] == sorted_scores[num_to_mask][1]):
        cand_indices = [sorted_scores[i][0] for i in range(num_to_mask-1)]
    else:
        cand_indices = [sorted_scores[i][0] for i in range(num_to_mask)]

    shuffle(cand_indices)
    masked_lms = []
    for index in cand_indices:
        masked_token = None
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        tokens[index] = masked_token
    
    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


#Done 去掉NSP
def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document, document_c_s = doc_database[doc_idx]
    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    i = 0
    while i < len(document):
        segment, segment_c_s = document[i], document_c_s[i]
        
        if segment:
            truncate_seq(segment, segment_c_s, max_num_tokens)
            tokens = ["[CLS]"] + segment + ["[SEP]"]
            pd_s = [(0., 0.)] + segment_c_s + [(0., 0.)]
            assert len(tokens) == len(pd_s)

            # 对语法序列进行破坏
            tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                tokens, pd_s, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)
            instance = {
                    "tokens": tokens,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
            instances.append(instance)
        i += 1

    return instances


def create_training_file(docs, vocab_list, args, epoch_num):
    epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_file = args.output_dir / "epoch_{}_metrics.json".format(epoch_num)
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = SubTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    nlp = spacy.load("en_core_web_sm")
    pd_rep = get_pd()
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            c_s = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip().lower()
                if line == "":
                    docs.add_document(doc, c_s)
                    doc = []
                    c_s = []
                    # continue
                else:
                    nlp_doc = nlp(line)
                    tokens = [t.text for t in nlp_doc]
                    pd_s = pd_similarity(pd_rep, nlp_doc)
                    # print("tokens: ", tokens)
                    # print("pd_s: ", pd_s)
                    tokens, token_pd_s, _ = tokenizer.subword_tokenize(tokens, pd_s)
                    doc.append(tokens)
                    c_s.append(token_pd_s)
            if doc:
                docs.add_document(doc, c_s)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")
        args.output_dir.mkdir(exist_ok=True)

        if args.num_workers > 1:
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            arguments = [(docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(docs, vocab_list, args, epoch)


if __name__ == '__main__':
    main()
