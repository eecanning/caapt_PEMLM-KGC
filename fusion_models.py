import torch.nn.functional as F
from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size,hidden_size,output_size,num_label):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Linear(output_size, num_label)



    def forward(self, semantic_x,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + semantic_x
        x = self.classifier(x)
        return x
#
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, num_label)

    def forward(self, x):
        x = self.linear(x)
        return x

class simpleMLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(simpleMLP,self).__init__()
        self.fc = nn.Linear(input_size,output_size)
    def forward(self,x):
        x = self.fc(x)
        return x

class PEMLM_F(nn.Module):
    def __init__(self, bert,transE,simpleMLP,tokenizer,classifier,negative_nums, device):
        super(PEMLM_F, self).__init__()
        self.bert = bert.to(device)
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.token_to_id('[MASK]')
        self.transE = transE
        self.classifier = classifier
        self.simplemlp = simpleMLP
        self.negative_nums = negative_nums


    def forward(self, inputs_embeds,input_ids,heads,relations,tails):
        head_embeddings,relation_embeddings,tail_embeddings = self.transE(heads,relations,tails)
        #pre fusion
        semantic_head_embedding = inputs_embeds[:,1,:].clone()
        semantic_relation_embedding = inputs_embeds[:,2,:].clone()
        fusion_head = torch.cat((semantic_head_embedding,head_embeddings),dim=-1)
        fusion_relation = torch.cat((semantic_relation_embedding,relation_embeddings),dim=-1)
        fusion_head_output = self.simplemlp(fusion_head)
        fusion_relation_output = self.simplemlp(fusion_relation)
        inputs_embeds[:, 1, :] = fusion_head_output
        inputs_embeds[:, 2, :] = fusion_relation_output


        output = self.bert(inputs_embeds =inputs_embeds)
        mask_positions_list = [(id == self.mask_id).nonzero().squeeze() for id in input_ids]
        mask_hidden_state = [hidden_state[mask_positions] for hidden_state, mask_positions in
                             zip(output['last_hidden_state'], mask_positions_list)]
        mask_hidden_state = torch.stack(mask_hidden_state)
        structure_loss = self.transE.InfoNCE_loss(heads,relations,tails, self.negative_nums)
        logit = self.classifier(mask_hidden_state)

        return logit,structure_loss