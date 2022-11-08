from pathlib import Path
from absa.model import bert_ABSA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import collections
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, RawFinalResult, RawSpanResult, span_annotate_candidates, wrapped_get_final_text, id_to_label
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from squad.squad_evaluate import exact_match_score, f1_score
import time
import numpy as np
from transformers import BertTokenizer
try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evl_time(t):
    min, sec= divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

def metric_max_over_ground_truths(metric_fn, term, polarity, gold_terms, gold_polarities):
    hit = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        score = metric_fn(term, gold_term)
        if score and polarity == gold_polarity:
            hit = 1
    return hit

def read_train_data(args, tokenizer, logger):
    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, verbose_logging=args.verbose_logging)
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_examples, train_features, train_data

def read_da_train_data(args, tokenizer, logger):
    train_path = os.path.join(args.data_dir, args.da_train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, verbose_logging=args.verbose_logging)
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_examples, train_features, train_data

def read_eval_data(args, tokenizer, logger):
    eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, verbose_logging=args.verbose_logging)

    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def eval_absa(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred_terms': pred_terms, 'pred_polarities': pred_polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(pred_terms, pred_polarities):
            common += metric_max_over_ground_truths(exact_match_score, term, polarity, example.term_texts, example.polarities)
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def evaluate(args, model, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, example_indices = batch

        with torch.no_grad():
            batch_start_logits, batch_end_logits, sequence_output = model('extract_inference', ids_tensors=input_ids, masks_tensors=input_mask)


        batch_features, batch_results = [], []
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            batch_features.append(eval_feature)
            batch_results.append(RawSpanResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        span_starts, span_ends, _, label_masks = span_annotate_candidates(eval_examples, batch_features, batch_results,
                                                                          args.filter_type, False,
                                                                          args.use_heuristics, args.use_nms,
                                                                          args.logit_threshold, args.n_best_size,
                                                                          args.max_answer_length, args.do_lower_case,
                                                                          args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        span_starts = span_starts.to(DEVICE)
        span_ends = span_ends.to(DEVICE)
        sequence_output = sequence_output.to(DEVICE)
        with torch.no_grad():
            batch_ac_logits = model('classify_inference', masks_tensors=input_mask, span_starts=span_starts,
                                    span_ends=span_ends, sequence_input=sequence_output)    # [N, M, 4]

        for j, example_index in enumerate(example_indices):
            cls_pred = batch_ac_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)
    # if write_pred:
    #     output_file = os.path.join(args.output_dir, "predictions.json")
    #     with open(output_file, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    #     logger.info("Writing predictions to: %s" % (output_file))
    return metrics


def train_absa(args, model, train_examples, train_features, train_dataloader,
                    eval_examples, eval_features, eval_dataloader):
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = 2e-5, eps = epsilon)
    all_data = len(train_dataloader)
    F1_set = []
    epochs = args.epochs
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []

        for step, batch in enumerate(train_dataloader):
            t0 = time.time()
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices = batch
            
            batch_start_logits, batch_end_logits, enc = model('extract_inference', ids_tensors=input_ids, masks_tensors=input_mask)

            batch_features, batch_results = [], []
            for j, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[j].detach().cpu().tolist()
                end_logits = batch_end_logits[j].detach().cpu().tolist()
                train_feature = train_features[example_index.item()]
                unique_id = int(train_feature.unique_id)
                batch_features.append(train_feature)
                batch_results.append(RawSpanResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

            span_starts, span_ends, labels, label_masks = span_annotate_candidates(train_examples, batch_features,
                                                                                batch_results,
                                                                                args.filter_type, True,
                                                                                args.use_heuristics,
                                                                                args.use_nms,
                                                                                args.logit_threshold,
                                                                                args.n_best_size,
                                                                                args.max_answer_length,
                                                                                args.do_lower_case,
                                                                                args.verbose_logging, logger)

            span_starts = torch.tensor(span_starts, dtype=torch.long)
            span_ends = torch.tensor(span_ends, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            label_masks = torch.tensor(label_masks, dtype=torch.long)
            span_starts = span_starts.to(DEVICE)
            span_ends = span_ends.to(DEVICE)
            labels = labels.to(DEVICE)
            label_masks = label_masks.to(DEVICE)

            loss = model('train', ids_tensors=input_ids, masks_tensors=input_mask, start_positions=start_positions, end_positions=end_positions,
                        span_starts=span_starts, span_ends=span_ends,
                        polarity_labels=labels, label_masks=label_masks)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         
        
        metrics = evaluate(args, model, eval_examples, eval_features, eval_dataloader, logger)
        F1_set.append([epoch, metrics['f1']])
        with open(args.out_dir, "a+", encoding="utf-8") as f:
                f.write(str(epoch) + ' ' + str(metrics) + '\n')
    
    with open(args.out_dir, "a+", encoding="utf-8") as f:
            maxf = max(F1_set, key=lambda x:x[1])
            f.write(f"\nmax F1:{maxf[1]}  epoch:{maxf[0]} \n")

def da_train_absa(args, model, train_examples, train_features, train_dataloader,
                    eval_examples, eval_features, eval_dataloader):
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = 2e-5, eps = epsilon)
    all_data = len(train_dataloader)
    F1_set = []
    epochs = args.da_epochs
    for epoch in range(epochs):
        finish_data = 0
        losses = []
        current_times = []

        for step, batch in enumerate(train_dataloader):
            t0 = time.time()
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices = batch
            
            batch_start_logits, batch_end_logits, enc = model('extract_inference', ids_tensors=input_ids, masks_tensors=input_mask)

            batch_features, batch_results = [], []
            for j, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[j].detach().cpu().tolist()
                end_logits = batch_end_logits[j].detach().cpu().tolist()
                train_feature = train_features[example_index.item()]
                unique_id = int(train_feature.unique_id)
                batch_features.append(train_feature)
                batch_results.append(RawSpanResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

            span_starts, span_ends, labels, label_masks = span_annotate_candidates(train_examples, batch_features,
                                                                                batch_results,
                                                                                args.filter_type, True,
                                                                                args.use_heuristics,
                                                                                args.use_nms,
                                                                                args.logit_threshold,
                                                                                args.n_best_size,
                                                                                args.max_answer_length,
                                                                                args.do_lower_case,
                                                                                args.verbose_logging, logger)

            span_starts = torch.tensor(span_starts, dtype=torch.long)
            span_ends = torch.tensor(span_ends, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            label_masks = torch.tensor(label_masks, dtype=torch.long)
            span_starts = span_starts.to(DEVICE)
            span_ends = span_ends.to(DEVICE)
            labels = labels.to(DEVICE)
            label_masks = label_masks.to(DEVICE)

            loss = model('train', ids_tensors=input_ids, masks_tensors=input_mask, start_positions=start_positions, end_positions=end_positions,
                        span_starts=span_starts, span_ends=span_ends,
                        polarity_labels=labels, label_masks=label_masks)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            finish_data += 1
            current_times.append(round(time.time()-t0,3))
            current = np.mean(current_times)
            hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
            print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         
        
        metrics = evaluate(args, model, eval_examples, eval_features, eval_dataloader, logger)
        F1_set.append([epoch, metrics['f1']])
        with open(args.out_dir, "a+", encoding="utf-8") as f:
                f.write('da' + str(epoch) + ' ' + str(metrics) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default=None, type=str, help="SemEval xml for training")
    parser.add_argument("--da_train_file", default='l-r.txt', type=str, help="da training")
    parser.add_argument("--predict_file", default=None, type=str, help="SemEval csv for prediction")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--logit_threshold", default=8., type=float,
                        help="Logit threshold for annotating labels.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--bert_model', default='pretrained/dev_corpus_finetuned_dp_rest', type=str)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--da_epochs', type=int, default=5)

    args = parser.parse_args()

    pretrained_model = args.bert_model
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logger.info('out_dir: {}'.format(args.out_dir))

    logger.info("***** Preparing model *****")
    model = bert_ABSA(pretrained_model).to(DEVICE)


    logger.info("***** Preparing training *****")
    train_examples, train_features, train_data = read_train_data(args, tokenizer, logger)
    da_train_examples, da_train_features, da_train_data = read_da_train_data(args, tokenizer, logger)

    train_data = ConcatDataset([train_data, da_train_data])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    logger.info("***** Preparing evaluation *****")
    eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)

    train_absa(args, model, train_examples, train_features, train_dataloader, eval_examples, eval_features, eval_dataloader)

if __name__ == '__main__':
    main()