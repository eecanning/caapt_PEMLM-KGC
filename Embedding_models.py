import torch.nn.functional as F
from torch import nn
import torch

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, b=7.0,temperature = 1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.b = b
        self.temperature = temperature
        #margin_loss
        # self.temperature = nn.Parameter(torch.tensor(temperature))



        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads, relations, tails):
        # print(heads)
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        return head_embeddings, relation_embeddings, tail_embeddings
    def get_loss(self,heads, relations, tails):
        batch_size = heads.shape[0]
        negative_tails = torch.Tensor().to("cuda:0")

        for i in range(len(heads)):
            other_tails = torch.cat((tails[:i], tails[i + 1:]))
            negative_tails = torch.cat((negative_tails, other_tails))
        negative_tails = negative_tails.type(torch.int)

        negative_tails = negative_tails.reshape(batch_size,-1)



        positive_score = self.score_function(self.entity_embeddings(heads),self.relation_embeddings(relations),self.entity_embeddings(tails))

        heads_col = heads.reshape(-1,1)
        relations_col = relations.reshape(-1,1)
        negative_score = self.score_function(self.entity_embeddings(heads_col),self.relation_embeddings(relations_col),self.entity_embeddings(negative_tails))
        # Calculate InfoNCE Loss
        # loss = -torch.log(torch.exp(positive_score/self.temperature) / (torch.exp(positive_score/self.temperature) + torch.sum(torch.exp(negative_score/self.temperature), dim=-1)))
        #marginLoss
        # print(positive_score)
        loss = torch.relu(positive_score.unsqueeze(1) - negative_score + 1)
        return loss.mean()

    def score_function(self,head_embeddings,relation_embeddings,tail_embeddings):
        score = torch.cosine_similarity(head_embeddings+relation_embeddings,tail_embeddings,dim=-1)#相似度分数
        # score = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=1, dim=-1)
        # return self.b - 0.5 * (torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=2, dim=-1)**2)
        return score
    def get_output(self,heads,relations):
        output = self.entity_embeddings(heads)+self.relation_embeddings(relations)
        logits = torch.cosine_similarity(output.unsqueeze(1),self.entity_embeddings.weight.data,dim=-1)
        return logits
class NewTransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim,groundtruth,entity2id, b=7.0,temperature = 1.0):
        super(NewTransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.b = b
        self.temperature = temperature
        self.groundtruth = groundtruth
        self.entity2id = entity2id
        self.entities = range(len(entity2id))
        # self.temperature = nn.Parameter(torch.tensor(temperature))



        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads, relations, tails):
        # print(heads)
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # score = self.b - 0.5 * (torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=2, dim=-1)**2)

        return head_embeddings, relation_embeddings, tail_embeddings
    def InfoNCE_loss(self,heads,relations,labels):
        # random_indices = torch.randint(0,len(self.entity2id),(135,))
        # sample_embedding = self.entity_embeddings.weight.data[random_indices]
        sample_embedding = self.entity_embeddings.weight.data
        h_embedding = self.entity_embeddings(heads.reshape(-1,1))
        r_embedding = self.relation_embeddings(relations.reshape(-1,1))

        total_score = self.score_function(h_embedding,r_embedding,sample_embedding)
        positive_score = self.score_function(self.entity_embeddings(heads),self.relation_embeddings(relations),self.entity_embeddings(labels))
        loss = -torch.log(torch.exp(positive_score)/torch.sum(torch.exp(total_score),dim=-1))

        return loss.mean()

    def score_function(self,head_embeddings,relation_embeddings,tail_embeddings):
        score = torch.cosine_similarity(head_embeddings+relation_embeddings,tail_embeddings,dim=-1)
        return score
        # return self.b - 0.5 * (torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=2, dim=-1)**2)

