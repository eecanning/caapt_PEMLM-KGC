import argparse

from sklearn.decomposition import PCA

from utils import *
from transformers import RobertaModel, RobertaTokenizer, AlbertModel, AlbertTokenizer, BertModel
from tqdm import tqdm
import torch
import json
from tokenizers import Tokenizer,models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build word embedding.")
    parser.add_argument('--PreEncoding_Model', type=str, default='bert-base-uncased', help="Pre-Encoding model path")
    parser.add_argument('--main_model',  type=str, default='bert-base-uncased', help="main model path")
    parser.add_argument('--entity2text_path', type=str, required=True, help="entity2text file path")
    parser.add_argument('--relation2text_path', type=str, required=True, help="relation2text file path")
    parser.add_argument('--embedding_json_path', type=str, required=True, help="Pre-Encoding embedding json save path")
    parser.add_argument('--tokenizer_save_path', type=str, required=True, help="new tokenizer path")
    parser.add_argument('--embedding_path', type=str, required=True, help="Pre-Encoding embedding save path")




    arguments = parser.parse_args()
    model = arguments.PreEncoding_Model

    model_des = BertModel.from_pretrained(model)
    tokenizer = BertModel.from_pretrained(model)

    entity2text_path = arguments.entity2text_path
    relation2text_path = arguments.relation2text_path
    entity2text = read_text_from_txt(entity2text_path)
    relation2text = read_text_from_txt(relation2text_path)
    total2text = {}
    total2text.update(entity2text)
    total2text.update(relation2text)

    model_des = model_des.to("cuda:0")
    word_embeddings = {}
    for item in tqdm(total2text.items()):
        key = item[0]
        des = item[1]
        input_des = tokenizer(des,return_tensors='pt',max_length=128,truncation=True,padding='max_length').to("cuda:0")
        # output = model_des(**input_des)['pooler_output']
        # mean pool
        output = model_des(**input_des)
        hidden_states = output.last_hidden_state
        mean_pooler = torch.mean(hidden_states,dim=1)

        # PCA decomposit For ALBERt
        # mean_pooler = mean_pooler.view(mean_pooler.size(0),6,128)
        # transform_mean_pooler = mean_pooler.mean(dim=1)

        # Max Pool
        # _,max_pooled = torch.max(hidden_states,dim=1)

        # cls Pool
        # cls_hidden_state = output.last_hidden_state[:,0,:]

        word_embeddings[key] = mean_pooler.detach().cpu().numpy().tolist()


    # with open(arguments.embedding_json_path,'w') as f:
    #     json.dump(word_embeddings,f)
    #
    # with open(arguments.embedding_json_path) as f:
    #     word_embeddings = json.load(f)

    vocab = []
    word_embeddings_weight = torch.Tensor()
    for key, embedding in tqdm(word_embeddings.items()):
        vocab.append(key)
        word_embeddings_weight = torch.cat((word_embeddings_weight, torch.Tensor(embedding)), dim=0)

    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.add_tokens(vocab)
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

    tokenizer_vocab = tokenizer.get_vocab()

    main_model = BertModel.from_pretrained(arguments.main_model)
    main_tokenizer = BertModel.from_pretrained(arguments.main_model)

    pad_id = main_tokenizer.pad_token_id
    cls_id = main_tokenizer.cls_token_id
    sep_id = main_tokenizer.sep_token_id
    mask_id = main_tokenizer.mask_token_id
    unk_id = main_tokenizer.unk_token_id

    pad_embedding = model.embeddings.word_embeddings.weight[pad_id]
    cls_embedding = model.embeddings.word_embeddings.weight[cls_id]
    sep_embedding = model.embeddings.word_embeddings.weight[sep_id]
    mask_embedding = model.embeddings.word_embeddings.weight[mask_id]
    unk_embedding = model.embeddings.word_embeddings.weight[unk_id]

    word_embeddings_weight = torch.cat((word_embeddings_weight, pad_embedding.unsqueeze(0)), dim=0)
    word_embeddings_weight = torch.cat((word_embeddings_weight, cls_embedding.unsqueeze(0)), dim=0)
    word_embeddings_weight = torch.cat((word_embeddings_weight, sep_embedding.unsqueeze(0)), dim=0)
    word_embeddings_weight = torch.cat((word_embeddings_weight, mask_embedding.unsqueeze(0)), dim=0)
    word_embeddings_weight = torch.cat((word_embeddings_weight, unk_embedding.unsqueeze(0)), dim=0)

    tokenizer.save(arguments.tokenizer_save_path)
    torch.save(word_embeddings_weight, arguments.embedding_path)
