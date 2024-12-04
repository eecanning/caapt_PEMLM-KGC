import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('--train_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--weight_path', type=str, default='parameter/WN18RR_combinehr.pth', help='model_weight_path')
    parser.add_argument('--model_path', type=str, default='model/bert-base-uncased-model', help='original model path')
    parser.add_argument('--tokenizer_path', type=str, default='model/WN18RR_tokenizer_reverse.json', help='tokenizer path')
    parser.add_argument('--entity_path', type=str, default='data/WN18RR/entities.txt', help='entity path')
    parser.add_argument('--train_data_path', type=str, default='data/WN18RR/train.tsv', help='train data path')
    parser.add_argument('--valid_data_path', type=str, default='data/WN18RR/valid_filter.tsv', help='valid data path')
    parser.add_argument('--test_data_path', type=str, default='data/WN18RR/test_filter.tsv', help='test data path')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='num_attention_heads')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='num_hidden_layers')
    parser.add_argument('--max_length', type=int, default=128, help='max_length')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--train_result_json_path', type=str,default='log/WN18RR_train_combinehr.json', help='train_result_json_path')
    parser.add_argument('--valid_result_json_path', type=str,default='log/WN18RR_valid_combinehr.json', help='valid_result_json_path')
    parser.add_argument('--test_result_json_path', type=str,default='log/WN18RR_test_combinehr.json', help='test_result_json_path')

    args = parser.parse_args()
    return args