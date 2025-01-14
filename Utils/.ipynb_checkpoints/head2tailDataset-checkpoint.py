
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def get_data_from_rawdata(tokenizer,triple_data,entity2id,modes):
    CLS_id = tokenizer.token_to_id('[CLS]')
    SEP_id = tokenizer.token_to_id('[SEP]')
    MASK_id = tokenizer.token_to_id('[MASK]')
    triple_data = triple_data
    input_ids = []
    entity2id = entity2id
    labels = []
    if modes == 'tail-batch':
        for line in tqdm(triple_data,total=len(triple_data)):
            head, relation, tail = line
            if len(line) != 3:
                print(f"Ignoring invalid line: {line}")
                continue
            # get label
            label = entity2id[tail]
            labels.append(label)
            # get input_ids
            head_ids = tokenizer.token_to_id(head)
            relation_ids = tokenizer.token_to_id(relation)
            tail_ids = [MASK_id]
            input_id = [CLS_id] + [head_ids] + [relation_ids] + tail_ids + [SEP_id]

            input_ids.append(input_id)


    return input_ids,labels

class OnlyTailDataset(Dataset):
    def __init__(self,input_ids,labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, item):
        input_id = torch.tensor(self.input_ids[item])
        label = torch.tensor(self.labels[item])

        return input_id,label

def get_data_from_rawdata_fusion(tokenizer,triple_data,entity2id,relation2id,modes):
    CLS_id = tokenizer.token_to_id('[CLS]')
    SEP_id = tokenizer.token_to_id('[SEP]')
    MASK_id = tokenizer.token_to_id('[MASK]')
    triple_data = triple_data
    input_ids = []
    entity2id = entity2id
    labels = []
    heads = []
    relations = []
    if modes == 'tail-batch':
        for line in tqdm(triple_data,total=len(triple_data)):
            head, relation, tail = line
            if len(line) != 3:
                print(f"Ignoring invalid line: {line}")
                continue
            #label获取
            label = entity2id[tail]
            labels.append(label)
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            heads.append(head_id)
            relations.append(relation_id)
            # input_id获取
            head_ids = tokenizer.token_to_id(head)
            relation_ids = tokenizer.token_to_id(relation)
            tail_ids = [MASK_id]
            input_id = [CLS_id] + [head_ids] + [relation_ids] + tail_ids + [SEP_id]

            input_ids.append(input_id)


    return input_ids,heads,relations,labels

class OnlyTailDataset_fusion(Dataset):
    def __init__(self,input_ids,heads,relations,labels):
        self.input_ids = input_ids
        self.labels = labels
        self.heads = heads
        self.relations = relations


    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, item):
        input_ids = torch.tensor(self.input_ids[item])
        labels = torch.tensor(self.labels[item])
        heads = torch.tensor(self.heads[item])
        relations = torch.tensor(self.relations[item])
        return input_ids,heads,relations,labels
if __name__ == '__main__':
    from KGE.CoKE.utils import *
    from tokenizers import Tokenizer

    train_data = read_triplets_from_txt('data/FB15k-237/train_filter.tsv')
    dev_data = read_triplets_from_txt('data/FB15k-237/dev_long.tsv')
    test_data = read_triplets_from_txt('data/FB15k-237/test_long.tsv')
    entity2id = read_entity('data/FB15k-237/long_entities.txt')
    reverse_train_data = []
    for triplet in train_data:
        reverse_train_data.append([triplet[2], triplet[1], triplet[0]])
    reverse_test_data = []
    for triplet in test_data:
        reverse_test_data.append([triplet[2], triplet[1], triplet[0]])
    reverse_val_data = []
    for triplet in dev_data:
        reverse_val_data.append([triplet[2], triplet[1], triplet[0]])

    onlyTail_train_data = train_data + reverse_train_data
    onlyTail_test_data = test_data + reverse_test_data
    onlyTail_val_data = dev_data + reverse_val_data

    tokenizer = Tokenizer.from_file("../model/new_tokenizer.json")
    input_ids, labels = get_data_from_rawdata(tokenizer, onlyTail_train_data, entity2id,
                                                                               'tail-batch')

    dataset = OnlyTailDataset(input_ids, labels)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        print(batch)
        break

