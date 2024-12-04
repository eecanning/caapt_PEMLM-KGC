# PEMLM-KGC
> Joint Pre-Encoding Representation and Sturcture Embedding for Efficient and Low-Resource Knowledge Graph Completion

[//]: # (![image]&#40;https://github.com/qiucy23/PEMLM-KGC/blob/main/model_pic.png&#41;)



[//]: # ([![NPM Version][npm-image]][npm-url])

[//]: # ([![Build Status][travis-image]][travis-url])

[//]: # ([![Downloads Stats][npm-downloads]][npm-url])

An efficient description-based model for KGC
![](https://github.com/qiucy23/PEMLM-KGC/blob/main/model_pic.png)

## Getting Started 

### requirements

you can use pip install from requirements.txt
```
pip install -r requirements.txt
```





### Pre-Encoding step

You can get three BERT-base precoding embeddings by running the following commands

#### FB15k-237 Dataset
```
python buildwordEmbedding.py --entity2text_path data/FB15k-237/entity2textlong_filter.txt' --relation2text data/FB15k-237/reverse_relation2text.txt --tokenizer_save_path model/FB15k237_tokenizer.json --embedding_path model/FB15k237_word_embeddings.pt
```
#### WN18RR Dataset
```aiignore
python buildwordEmbedding.py --entity2text_path data/WN18RR/entity2text_filter.txt' --relation2text data/WN18RR/reverse_relation2text.txt --tokenizer_save_path model/WN18RR_tokenizer.json --embedding_path model/WN18RR_word_embeddings.pt
```
#### UMLS Dataset
```aiignore
python buildwordEmbedding.py --entity2text_path data/WN18RR/entity2text_filter.txt' --relation2text data/WN18RR/reverse_relation2text.txt --tokenizer_save_path model/WN18RR_tokenizer.json --embedding_path model/WN18RR_word_embeddings.pt
```
## Deployment 部署方法

部署到生产环境注意事项。

## Contributing 贡献指南

Please read [CONTRIBUTING.md](#) for details on our code of conduct, and the process for submitting pull requests to us.

清阅读 [CONTRIBUTING.md](#) 了解如何向这个项目贡献代码

## Release History 版本历史

* 0.2.1
    * CHANGE: Update docs
* 0.2.0
    * CHANGE: Remove `README.md`
* 0.1.0
    * Work in progress

## Authors 关于作者

* **WangYan** - *Initial work* - [WangYan](https://wangyan.org)

查看更多关于这个项目的贡献者，请阅读 [contributors](#) 

## License 授权协议

这个项目 MIT 协议， 请点击 [LICENSE.md](LICENSE.md) 了解更多细节。