import argparse
import json
from Utils.Embedding_models import *
import torch.nn
from tokenizers import Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from Utils.evaluation import *
from Utils.fusion_models import *
from torch.utils.data import DataLoader
import os
from Utils.head2tailDataset import *
from Utils.model import *
from Utils.utils import *

def train_val_epochs(bert,model,train_dataloader,val_dataloader,epochs,lr,label_num,device):
    max_MRR = 0
    #交叉熵损失函数
    criterion = CrossEntropyLoss(label_smoothing=0.8)
    params = [
        {'params':model.transE.entity_embeddings.weight,'lr':arguments.tran_lr},
        {'params':model.transE.relation_embeddings.weight,'lr':arguments.tran_lr},
        {'params':[param for name,param in model.named_parameters() if 'transE' not in name],'lr':lr}
    ]
    optimizer = torch.optim.Adam(params,lr=lr)
    for epoch in range(epochs):
        model.train()
        train_tail_loss = 0
        bar = tqdm(total = len(train_dataloader),desc=f'Epoch{epoch+1}/{epochs}', ncols=100)
        for iter,batch in enumerate(train_dataloader):
            input_ids,heads,relations,labels = batch
            input_ids = input_ids.to(device)
            heads = heads.to(device)
            relations = relations.to(device)
            labels = labels.to(device)
            word_embeds = bert.embeddings.word_embeddings(input_ids)
            pos_embeds = bert.embeddings.position_embeddings(torch.arange(0,input_ids.shape[1],dtype = torch.long).to(device))
            input_embeddings = word_embeds + pos_embeds
            output,structure_loss = model(input_embeddings,input_ids,heads,relations,labels)
            classify_loss = criterion(output, labels)
            loss = classify_loss + arguments.alpha * structure_loss
            train_tail_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description(f'train_tail_Epoch {epoch + 1}/{epochs}')
            bar.set_postfix(c_loss=f'{classify_loss.item():.4f}',s_loss = f'{structure_loss.item():.4f}')
            bar.update(1)
        bar.close()
        mean_loss = train_tail_loss/len(train_dataloader)
        print('train_epoch: {}, loss:{}'.format(epoch+1,mean_loss))
        train_epoch_result = {
            "epoch":epoch+1,
            "loss":mean_loss
        }
        #写入json
        with open(arguments.train_result_json_path,'a') as jsf:
            json.dump(train_epoch_result,jsf)
            jsf.write('\n')


        model.eval()
        # #tail-barch validation
        total_rank = []
        val_loss = 0.0
        bar = tqdm(total = len(val_dataloader),desc=f'Epoch{epoch+1}/{epochs}', ncols=100)
        for iter, batch in enumerate(val_dataloader):
            input_ids, heads, relations, labels = batch
            input_ids = input_ids.to(device)
            heads = heads.to(device)
            relations = relations.to(device)
            labels = labels.to(device)
            word_embeds = bert.embeddings.word_embeddings(input_ids)
            pos_embeds = bert.embeddings.position_embeddings(torch.arange(0,input_ids.shape[1],dtype = torch.long).to(device))
            input_embeddings = word_embeds + pos_embeds
            r_h = []
            for input_id in input_ids:
                r = input_id[2].item()
                h = input_id[1].item()
                r_token = tokenizer.id_to_token(r)
                h_token = tokenizer.id_to_token(h)
                r_h.append((r_token, h_token))

            output,structure_loss = model(input_embeddings,input_ids,heads,relations,labels)
            classify_loss = criterion(output, labels)
            loss = classify_loss + arguments.alpha * structure_loss
            val_loss += loss.item()
            ranks = evaluation(output, groundtruth, entity2id, r_h, labels, modes='tail')

            for label, rank in zip(labels, ranks):
                the_rank = torch.where(rank == label)[0].item() + 1
                total_rank.append(the_rank)
            bar.set_description(f'test_tail_Epoch {epoch + 1}/{epochs}')
            bar.set_postfix(loss=f'{loss.item()}')
            bar.update(1)
        bar.close()

        total_rank = torch.tensor(total_rank)
        tail_MR = total_rank.sum().item() / len(total_rank)
        tail_MRR = torch.sum(1 / total_rank).item() / len(total_rank)
        tail_hit10 = torch.sum(total_rank <= 10).item() / len(total_rank)
        tail_hit3 = torch.sum(total_rank <= 3).item() / len(total_rank)
        tail_hit1 = torch.sum(total_rank <= 1).item() / len(total_rank)
        tail_test_loss = val_loss / len(val_dataloader)
        print('MR:{} ,MRR:{} ,HIT10:{} ,HIT3:{} ,HIT1:{}'.format(
                                                                                                       tail_MR,
                                                                                                       tail_MRR,
                                                                                                       tail_hit10,
                                                                                                       tail_hit3,
                                                                                                       tail_hit1))
        test_epoch_result = {
            "epoch": epoch+1,
            "loss": tail_test_loss,
            "MR": tail_MR,
            "MRR": tail_MRR,
            "Hit@10": tail_hit10,
            "Hit@3": tail_hit3,
            "Hit@1": tail_hit1,

        }
        with open(arguments.valid_result_json_path, 'a') as jsf:
            json.dump(test_epoch_result, jsf)
            jsf.write('\n')
        #save model if mean MRR > max_MRR
        if tail_MRR > max_MRR:
            max_MRR = tail_MRR
            torch.save(model.state_dict(), arguments.weight_path)
            print('model saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for training')
    parser.add_argument('--tran_lr', type=float, default=1e-3, help='Learning rate for transModel')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--weight_path', type=str, default='parameter/WN18RR_PEMLM-F(triple).pth', help='model_weight_path')
    parser.add_argument('--model_path', type=str, default='bert-base-uncased', help='original model path')
    parser.add_argument('--embedding_path', type=str, default='model/WN18RR_word_embeddings.pt',
                        help='WN18RR embedding path')
    parser.add_argument('--embedding_size', type = int, default = 50, help = 'embedding size of embedding model.')
    parser.add_argument('--tokenizer_path', type=str, default='model/WN18RR_tokenizer.json',
                        help='tokenizer path')
    parser.add_argument('--relation2id', type=str, default='data/WN18RR/reverse_relations.txt', help='relation2id path')
    parser.add_argument('--entity_path', type=str, default='data/WN18RR/entities.txt', help='entity path')
    parser.add_argument('--train_data_path', type=str, default='data/WN18RR/train.tsv', help='train data path')
    parser.add_argument('--valid_data_path', type=str, default='data/WN18RR/valid_filter.tsv', help='valid data path')
    parser.add_argument('--test_data_path', type=str, default='data/WN18RR/test_filter.tsv', help='test data path')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='num_attention_heads')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='num_hidden_layers')
    parser.add_argument('--max_length', type=int, default=128, help='max_length')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--train_result_json_path', type=str, default='log/WN18RR/WN18RR_PEMLM-F_trainResult(triple).json',
                        help='train_result_json_path')
    parser.add_argument('--valid_result_json_path', type=str, default='log/WN18RR/WN18RR_PEMLM-F_valResult(triple).json',
                        help='valid_result_json_path')
    parser.add_argument('--test_result_json_path', type=str, default='log/WN18RR/WN18RR_PEMLM-F_testResult(triple).json',
                        help='test_result_json_path')
    parser.add_argument('--alpha', type=float, default=1.0, help='fusion loss weight')
    parser.add_argument('--negative_nums', type=int, default=256, help='numbers of negative samples ')
    parser.add_argument('--device', type=str, default='cuda:4',
                        help="Device to use for computation (e.g., 'cuda:0/1/2/3/4/5/6/7' or 'cpu')")


    arguments = parser.parse_args()

    epochs = arguments.epochs
    lr = arguments.lr
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = arguments.device

    train_data = read_triplets_from_txt(arguments.train_data_path)
    valid_data = read_triplets_from_txt(arguments.valid_data_path)
    test_data = read_triplets_from_txt(arguments.test_data_path)
    train_data_reverse = []
    for triplet in train_data:
        train_data_reverse.append([triplet[2], 'be' + triplet[1], triplet[0]])
    onlyTail_train_data = train_data + train_data_reverse
    valid_data_reverse = []
    for triplet in valid_data:
        valid_data_reverse.append([triplet[2], 'be' + triplet[1], triplet[0]])
    onlyTail_val_data = valid_data + valid_data_reverse
    test_data_reverse = []
    for triplet in test_data:
        test_data_reverse.append([triplet[2], 'be' + triplet[1], triplet[0]])
    onlyTail_test_data = test_data + test_data_reverse
    entity2id = read_entity(arguments.entity_path)
    relation2id = read_entity(arguments.relation2id)
    entity_set = set(entity2id.keys())
    label_num = len(entity2id)
    relation_num = len(relation2id)
    groundtruth = count_groundtruth(onlyTail_train_data, onlyTail_val_data, onlyTail_test_data)
    Bert = BertModel.from_pretrained(arguments.model_path).to(device)
    #freeze embedding layer
    # for param in Bert_model.embeddings.word_embeddings.parameters():
    #     param.requires_grad = False
    tokenizer = Tokenizer.from_file(arguments.tokenizer_path)
    vocab = tokenizer.get_vocab()
    entity2id = read_entity(arguments.entity_path)
    label_num = len(entity2id)
    # TransE model
    # transe = TransE(label_num,relation_num,arguments.hidden_size,groundtruth,entity2id)
    transe = TransE(label_num,relation_num,arguments.embedding_size,groundtruth,entity2id)

    classifier = Classifier(arguments.hidden_size,label_num)
    #simpleMlp
    simpleMlp = simpleMLP(arguments.hidden_size + arguments.embedding_size, arguments.hidden_size)

    new_word_embeddings_weight = torch.load(arguments.embedding_path)
    # if weight exists, load it
    weight_path = arguments.weight_path
    # PEMLM -F
    Bert.embeddings.word_embeddings.weight = torch.nn.Parameter(new_word_embeddings_weight)
    model = PEMLM_F(Bert,transe,simpleMlp,tokenizer,classifier, arguments.negative_nums,device).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('model has loaded.')
    else:
        print('model dont exists.')
        print('init model...')

    #train_data

    input_ids,heads,relations,labels = get_data_from_rawdata_fusion(tokenizer, onlyTail_train_data, entity2id,relation2id,'tail-batch')
    train_dataset = OnlyTailDataset_fusion(input_ids,heads,relations,labels)

    input_ids,heads,relations,labels = get_data_from_rawdata_fusion(tokenizer, onlyTail_test_data, entity2id,relation2id,'tail-batch')
    test_dataset = OnlyTailDataset_fusion(input_ids,heads,relations,labels)

    train_dataloader = DataLoader(train_dataset,batch_size = arguments.train_batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size = arguments.val_batch_size,shuffle=False)

    train_val_epochs(Bert,model,train_dataloader,test_dataloader,epochs,lr,label_num,device)