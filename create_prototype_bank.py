from deal_pd import *
import collections

domains = ["laptop", "rest", "mams"]


def get_bank(source_domain, target_domain, single_threshold, multi_gram_threshold, single_bank_size, multi_bank_size):
    source_pd_rep = get_aspect_pd(source_domain)
    source_vocab = set()
    for suffix in [".train.txt", ".test.txt"]:
        data_file = os.path.join("datasets", source_domain+suffix)
        lines = read_txt(data_file)
        for idx in range(len(lines)):
            line = lines[idx]
            sentence, label = line["sentence"], line["label"]
            tokens = set(sentence.lower().split(' '))
            source_vocab.update(tokens)


    target_file = os.path.join("datasets", target_domain+".train.txt")

    single_word_bank = []
    multi_word_bank = []
    bank = []

    lines = read_txt(target_file)
    for idx in range(len(lines)):
        line = lines[idx]
        sentence = line["sentence"]
        tokens = sentence.lower().split()
        nlp_doc = nlp(sentence)
        pd_s = pd_similarity(source_pd_rep, nlp_doc)
        pre_s = 0
        for token, s in zip(tokens, pd_s):
            if pre_s > multi_gram_threshold and s > multi_gram_threshold:
                multi_word_bank[-1] = multi_word_bank[-1] + ' ' + token
            elif s > multi_gram_threshold:
                multi_word_bank.append(token)
            if s > single_threshold:
                single_word_bank.append(token)
            pre_s = s

    multi_word_bank = filter(lambda phraze: ' ' in phraze, multi_word_bank)
    single_word_bank = filter(lambda token: token not in source_vocab, single_word_bank)
    
    single_word_bank = collections.Counter(single_word_bank)
    single_word_bank = single_word_bank.most_common(single_bank_size)
    single_word_bank = [word[0] for word in single_word_bank]

    multi_word_bank = collections.Counter(multi_word_bank)
    multi_word_bank = multi_word_bank.most_common(multi_bank_size)
    multi_word_bank = [word[0] for word in multi_word_bank]

    multi_word_dict = dict()
    for phraze in multi_word_bank:
        s = phraze.split(' ')
        if (len(s) in multi_word_dict.keys()):
            multi_word_dict[len(s)].append(phraze)
        else :
            multi_word_dict[len(s)] = []
            multi_word_dict[len(s)].append(phraze)

    # if target_domain == "laptop":
    #     for word in ["laptop", "computer", "pc", "mac", "notebook", "netbook", "macbook" "desktop"]:
    #         if word in single_word_bank:
    #             single_word_bank.remove(word)

    return single_word_bank, multi_word_bank, multi_word_dict

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

if __name__ == "__main__":
    source_domain = "rest"
    target_domain = "laptop"
    single_bank_size = 300
    multi_bank_size = 300
    single_threshold = 0.3
    multi_gram_threshold = 0.3
    single_word_bank, multi_word_bank, multi_word_dict = get_bank(source_domain, target_domain, single_threshold, multi_gram_threshold, single_bank_size, multi_bank_size)
    
    print("single_word_bank: ")
    print(len(single_word_bank))
    print(single_word_bank)
    print("multi_word_dict: ")
    print(len(multi_word_bank))
    print(multi_word_dict)
    
    # target_aspect_set = set()
    # target_file = os.path.join("datasets", target_domain+".train.txt")
    # lines = read_txt(target_file)
    # for idx in range(len(lines)):
    #     tokens = lines[idx]["sentence"]
    #     label = lines[idx]["label"]
    #     tags = ot2bio(label)
    #     ts_sequence = tag2aspect(tags)
    #     tokens = tokens.split(' ')
    #     for (beg, end) in ts_sequence:
    #         target_aspect_set.add(' '.join(tokens[beg: end+1]))
    
    # print(target_aspect_set)    ##478个
    # l1 = len(single_word_bank) 
    # sum = 0
    # for idx in range(l1):
    #     if (single_word_bank[idx] in target_aspect_set):
    #         sum += 1
    # print((sum+0.0001) / l1)

    # print(single_word_bank)
    # print(l1)
    # print(multi_word_bank)
    # print(len(multi_word_bank))

        

