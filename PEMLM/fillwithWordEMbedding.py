import argparse
import json
import torch
from tqdm import tqdm
from tokenizers import Tokenizer,models
from transformers import BertModel,BertTokenizer,RobertaModel,RobertaTokenizer,AlbertModel,AlbertTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build word embedding.")
    parser.add_argument('--PreEncoding_Model', type=str, default='bert-base-uncased', help="Pre-Encoding model path")
    parser.add_argument('--entity2text_path', type=str, required=True, help="entity2text file path")
    parser.add_argument('--relation2text_path', type=str, required=True, help="relation2text file path")
    parser.add_argument('--embedding_path', type=str, required=True, help="Pre-Encoding embedding save path")
    arguments = parser.parse_args()

    with open(arguments.embedding_path) as f:
        word_embeddings = json.load(f)

    vocab = []
    word_embeddings_weight = torch.Tensor()
    for key,embedding in tqdm(word_embeddings.items()):
        vocab.append(key)
        print(len(embedding[0]))
        word_embeddings_weight = torch.cat((word_embeddings_weight,torch.Tensor(embedding)),dim=0)

    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.add_tokens(vocab)
    tokenizer.add_special_tokens(["[PAD]","[CLS]","[SEP]","[MASK]","[UNK]"])

    tokenizer_vocab = tokenizer.get_vocab()


    model = AlbertModel.from_pretrained('model/ALBERT')
    bert_tokenizer = AlbertTokenizer.from_pretrained('model/ALBERT')
    pad_id = bert_tokenizer.pad_token_id
    cls_id = bert_tokenizer.cls_token_id
    sep_id = bert_tokenizer.sep_token_id
    mask_id = bert_tokenizer.mask_token_id
    unk_id = bert_tokenizer.unk_token_id

    pad_embedding = model.embeddings.word_embeddings.weight[pad_id]
    cls_embedding = model.embeddings.word_embeddings.weight[cls_id]
    sep_embedding = model.embeddings.word_embeddings.weight[sep_id]
    mask_embedding = model.embeddings.word_embeddings.weight[mask_id]
    unk_embedding = model.embeddings.word_embeddings.weight[unk_id]

    word_embeddings_weight = torch.cat((word_embeddings_weight,pad_embedding.unsqueeze(0)),dim = 0)
    word_embeddings_weight = torch.cat((word_embeddings_weight,cls_embedding.unsqueeze(0)),dim = 0)
    word_embeddings_weight = torch.cat((word_embeddings_weight,sep_embedding.unsqueeze(0)),dim = 0)
    word_embeddings_weight = torch.cat((word_embeddings_weight,mask_embedding.unsqueeze(0)),dim = 0)
    word_embeddings_weight = torch.cat((word_embeddings_weight,unk_embedding.unsqueeze(0)),dim = 0)


    tokenizer.save('model/WN18RR_ALBERT_tokenizer.json')
    torch.save(word_embeddings_weight,'model/WN18RR_word_embeddings_ALBERT.pt')