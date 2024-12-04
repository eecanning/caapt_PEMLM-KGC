import torch
from torch import nn
from transformers import BertTokenizer

from utils import read_triplets_from_txt


class Classifier(torch.nn.Module):
    def __init__(self, hidden_size, num_label):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, num_label)

    def forward(self, x):
        x = self.linear(x)
        return x


class MainModel(nn.Module):
    def __init__(self, bert, hidden_size, num_label,tokenizer,device):
        super(MainModel, self).__init__()
        self.bert = bert.to(device)
        self.classifier = Classifier(hidden_size, num_label).to(device)
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.token_to_id('[MASK]')


    def forward(self, inputs_embeds,input_ids):
        output = self.bert(inputs_embeds =inputs_embeds)
        mask_positions_list = [(id == self.mask_id).nonzero().squeeze() for id in input_ids]
        mask_hidden_state = [hidden_state[mask_positions] for hidden_state, mask_positions in
                             zip(output['last_hidden_state'], mask_positions_list)]
        mask_hidden_state = torch.stack(mask_hidden_state)
        logits = self.classifier(mask_hidden_state)
        return logits



