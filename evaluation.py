import numpy as np
import torch
device = "cuda"
def count_groundtruth(train_set ,valid_set ,test_set):
    groundtruth = { split: {'head': {}, 'rel': {}, 'tail': {}} for split in ['all', 'train', 'valid', 'test']}
    possible_entities = { split: {'head': {}, 'tail': {}} for split in ['train']}

    for triple in train_set:
        h, r, t = triple
        groundtruth['all']['head'].setdefault((r, t), [])
        groundtruth['all']['head'][(r, t)].append(h)
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['all']['rel'].setdefault((h, t), [])
        groundtruth['all']['rel'][(h, t)].append(r)
        groundtruth['train']['head'].setdefault((r, t), [])
        groundtruth['train']['head'][(r, t)].append(h)
        groundtruth['train']['tail'].setdefault((r, h), [])
        groundtruth['train']['tail'][(r, h)].append(t)
        groundtruth['train']['rel'].setdefault((h, t), [])
        groundtruth['train']['rel'][(h, t)].append(r)
        possible_entities['train']['head'].setdefault(r, set())
        possible_entities['train']['head'][r].add(h)
        possible_entities['train']['tail'].setdefault(r, set())
        possible_entities['train']['tail'][r].add(t)
    for triple in valid_set:
        h, r, t = triple
        groundtruth['all']['head'].setdefault((r, t), [])
        groundtruth['all']['head'][(r, t)].append(h)
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['all']['rel'].setdefault((h, t), [])
        groundtruth['all']['rel'][(h, t)].append(r)
        groundtruth['valid']['head'].setdefault((r, t), [])
        groundtruth['valid']['head'][(r, t)].append(h)
        groundtruth['valid']['tail'].setdefault((r, h), [])
        groundtruth['valid']['tail'][(r, h)].append(t)

    for triple in test_set:
        h, r, t = triple


        groundtruth['all']['head'].setdefault((r, t), [])
        groundtruth['all']['head'][(r, t)].append(h)
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['all']['rel'].setdefault((h, t), [])
        groundtruth['all']['rel'][(h, t)].append(r)
        groundtruth['test']['head'].setdefault((r, t), [])
        groundtruth['test']['head'][(r, t)].append(h)
        groundtruth['test']['tail'].setdefault((r, h), [])
        groundtruth['test']['tail'][(r, h)].append(t)
    return groundtruth, possible_entities


def evaluation(output,groundtruth,entity2id,r_ht,labels,modes):
    '''
    input:groundtruth,entity_set,entity2id,h_t mean h or t
    '''
    not_candidate_ents_id = []
    if modes != 'head' or modes != 'tail':
        ValueError('modes should be head or tail.')
    for i in range(len(r_ht)):
        r = r_ht[i][0]
        ht = r_ht[i][1]
        not_candidate_ent = set(groundtruth['all'][modes][(r, ht)])
        not_candidate_ents_id.append([entity2id[ent] for ent in not_candidate_ent])
    output_logit = output
    label_num = len(entity2id)

    for index,(not_candidate_ent,label) in enumerate(zip(not_candidate_ents_id,labels)):
        keys_to_modify = list(set(range(label_num)) & set(not_candidate_ent) - {label.item()})
        output_logit[index,keys_to_modify] = float('-inf')
    softmax = torch.softmax(output_logit, dim=1)
    ranks = torch.argsort(softmax, dim=1, descending=True)
    return ranks
def evaluationWithSoftmax(softmax_value,groundtruth,entity2id,r_ht,labels,modes):
    '''
    input:groundtruth,entity_set,entity2id,h_t mean h or t
    '''
    not_candidate_ents_id = []
    if modes != 'head' or modes != 'tail':
        ValueError('modes should be head or tail.')
    for i in range(len(r_ht)):
        r = r_ht[i][0]
        ht = r_ht[i][1]
        not_candidate_ent = set(groundtruth['all'][modes][(r, ht)])
        not_candidate_ents_id.append([entity2id[ent] for ent in not_candidate_ent])
    output_softmax_value = softmax_value
    label_num = len(entity2id)
    sum=output_softmax_value.sum()

    for index,(not_candidate_ent,label) in enumerate(zip(not_candidate_ents_id,labels)):
        keys_to_modify = list(set(range(label_num)) & set(not_candidate_ent) - {label.item()})
        output_softmax_value[index,keys_to_modify] = 0
    # softmax = torch.softmax(output_logit, dim=1)
    ranks = torch.argsort(output_softmax_value, dim=1, descending=True)
    return ranks



