import torch.nn.functional as F
from torch import nn
import torch

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim,groundtruth,entity2id, b=7.0,temperature = 1.0):
        super(TransE, self).__init__()
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

