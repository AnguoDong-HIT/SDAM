from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceLoss(nn.Module):
    def __init__(self, eps=0.2, reduction='mean', ignore_index=-1):
        super(InstanceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, target, weights):
        '''
        input: (batch*seq_len, 3) 
        target: (batch*seq_len) labels for every token
        weights: (batch*seq_len) weights for every token
        '''
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = F.nll_loss(log_preds, target, reduction='none', ignore_index=self.ignore_index)
        weights = F.softmax(weights, dim=-1)
        return torch.matmul(weights.view(-1), loss)

class bert_ATE(torch.nn.Module):
    def __init__(self, pretrain_model, pd_loss):
        super(bert_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.pd_loss = pd_loss

    def forward(self, ids_tensors, tags_tensors, masks_tensors, pd_s = None):
        bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
        # print(bert_outputs.size())
        linear_outputs = self.linear(bert_outputs)
        # print(linear_outputs.size())

        if tags_tensors is not None:
            # TODO 将pd_s作为每个词的loss加权
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1,3)
            # print(linear_outputs.size())
            # print(tags_tensors.size())

            loss = self.loss_fn(linear_outputs, tags_tensors)
            if pd_s is not None and self.pd_loss == True:
                loss_fct = InstanceLoss()
                pd_s = pd_s.view(-1)
                loss = loss_fct(linear_outputs, tags_tensors, pd_s)
            return loss
        else:
            return linear_outputs


class bert_ABSA(torch.nn.Module):
    def __init__(self, pretrain_model, pd_loss):
        super(bert_ABSA, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 7)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.pd_loss = pd_loss

    def forward(self, ids_tensors, tags_tensors, masks_tensors, pd_s = None):
        bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
        # print(bert_outputs.size())
        linear_outputs = self.linear(bert_outputs)
        # print(linear_outputs.size())

        if tags_tensors is not None:
            # TODO 将pd_s作为每个词的loss加权
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1,7)
            # print(linear_outputs.size())
            # print(tags_tensors.size())

            loss = self.loss_fn(linear_outputs, tags_tensors)
            if pd_s is not None and self.pd_loss == True:
                loss_fct = InstanceLoss()
                pd_s = pd_s.view(-1)
                loss = loss_fct(linear_outputs, tags_tensors, pd_s)
            return loss
        else:
            return linear_outputs
