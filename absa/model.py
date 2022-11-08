from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchcrf import CRF
from torch.nn import CrossEntropyLoss

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.reshape([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.reshape([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.reshape([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.reshape([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.reshape([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.reshape([-1])  # [N*M]
    span_ends_offset = span_ends_offset.reshape([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    # print(f'input.shape:{ input.shape}, input_mask: {input_mask.shape}')
    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask


class bert_ABSA(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ABSA, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.lstm = torch.nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=768//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.unary_affine = nn.Linear(768, 1)
        self.linear = nn.Linear(768, 2)
        # self.crf = CRF(self.tagset_size, batch_first=True)
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(768, 5)

    def forward(self, mode, masks_tensors, ids_tensors=None, start_positions=None, end_positions=None, span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, sequence_input=None):
        
        if mode == 'train':
            bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
            enc, _ = self.lstm(bert_outputs)
            ae_logits = self.linear(enc)

            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            ae_loss = (start_loss + end_loss) / 2

            span_output, span_mask = get_span_representation(span_starts, span_ends, enc,
                                                             masks_tensors)  # [N*M, JR, D], [N*M, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]
            
            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            return ae_loss + ac_loss

        elif mode == 'extract_inference':
            bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
            enc, _ = self.lstm(bert_outputs)
            ae_logits = self.linear(enc)

            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits, enc
        
        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             masks_tensors)  # [N*M, JR, D], [N*M, JR]

            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts)

    
