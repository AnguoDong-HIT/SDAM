import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from pytorch_transformers import BertPreTrainedModel, BertModel
import math

# 域分类头 -> 2
class DomainPredictionHead(nn.Module):
    def __init__(self, config):
        super(DomainPredictionHead, self).__init__()
        
        self.decoder = nn.Sequential(
            GradientReversal(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
        
class ABSABert(BertPreTrainedModel):
    def __init__(self, config):
        super(ABSABert, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)          # 任务分类器，token-level

        self.domain_cls = DomainPredictionHead(config)  # 域分类器
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                domain_label=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # bert输出
        sequence_output, pooled_output = outputs[:2] 
        sequence_output = self.dropout(sequence_output)

        if domain_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            max_len = sequence_output.size()[1]
            avg_pool = nn.functional.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=max_len).squeeze()
            domain_predicte_scores = self.domain_cls(avg_pool)
            if len(domain_predicte_scores.size()) == 1:
                domain_predicte_scores = domain_predicte_scores.unsqueeze(0)
            domain_loss = loss_fct(domain_predicte_scores, domain_label)
            return domain_loss
        else:
            logits = self.classifier(sequence_output)
            loss_fct = InstanceLoss()
            if labels is not None:
                with torch.no_grad():
                    # 如果是训练数据，利用域分类器的结果对每个词进行加权 batch * len * 2
                    factors = self.domain_cls(sequence_output)
                    factors = F.softmax(factors, dim=-1)
                    factors, _ = torch.min(factors, dim=-1, keepdim=True)
                    factors = factors.squeeze()  # batch*seq_len
                    factors = F.softmax(factors, dim=-1) 
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), factors.view(-1))
                return loss, factors
            else:
                return logits
