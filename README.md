# PEMLM-KGC [ORGANIZING]

---
> Joint Pre-Encoding Representation and Sturcture Embedding for Efficient and Low-Resource Knowledge Graph Completion

[//]: # (![image]&#40;https://github.com/qiucy23/PEMLM-KGC/blob/main/model_pic.png&#41;)




[//]: # ([![NPM Version][npm-image]][npm-url])

[//]: # ([![Build Status][travis-image]][travis-url])

[//]: # ([![Downloads Stats][npm-downloads]][npm-url])

An efficient description-based model for KGC
![](https://github.com/qiucy23/PEMLM-KGC/blob/main/pic/model_pic.png)
---
![](https://github.com/qiucy23/PEMLM-KGC/blob/main/pic/PEMLM-F.png)

## Update
- 06/12/24 - The code is tentatively organized.
## Installation

---

### requirements

you can use pip install from requirements.txt
```
pip install -r requirements.txt
```


## Getting Start
### Something you may want to know

* **entity2text.txt** means **short** text entity description.
* **entity2textlong.txt** means **long** text entity description.
* **entity2textlong_filter.txt (if exists)** means the cleaned text description file only contains the long entity description existing in the training, validation, and test sets.
* **entity2textlong_unseen.txt (if exists but NO USE)** represents the entities that have been filtered out. These entities do not all exist in the training set, the validation set, and the test set.

* **train/valid/test_filter.csv (if exists)** including triples contain only the entities whose description existence in entity2textlong.txt
* **train/valid/test_unseen.csv (if exists but NO USE)** represents the triplets of entities do not all exist in the training set, the validation set, and the test set.
* **reverse_relation2text.txt** including [relation+reverse_relation] and their description.
---
### Pre-Encoding step

You can get three BERT-base precoding embeddings by running the following commands

#### FB15k-237 Dataset
```aiignore
python buildwordEmbedding.py \
--entity2text_path data/FB15k-237/entity2textlong_filter.txt \
--relation2text data/FB15k-237/reverse_relation2text.txt \
--tokenizer_save_path model/FB15k237_tokenizer.json \
--embedding_path model/FB15k237_word_embeddings.pt
```
#### WN18RR Dataset
```aiignore
python buildwordEmbedding.py \
--entity2text_path data/WN18RR/entity2text_filter.txt \
--relation2text data/WN18RR/reverse_relation2text.txt \
--tokenizer_save_path model/WN18RR_tokenizer.json \
--embedding_path model/WN18RR_word_embeddings.pt
```
#### UMLS Dataset
```aiignore
python buildwordEmbedding.py \
--entity2text_path data/UMLS/entity2textlong.txt \
--relation2text data/UMLS/reverse_relation2text.txt \
--tokenizer_save_path model/UMLS_tokenizer.json \
--embedding_path model/UMLS_word_embeddings.pt
```
### Running with PEMLM

---
#### Running PEMLM for **FB15k-237**
```aiignore
python FB15k237main.py --train_batch_size 256 --lr 1e-5 --epochs 200 
```
the trained weight is saved in **parameter/FB15k237_PEMLM.pth** and train/val log will be saved in **log/FB15K237_PEMLM_trainResult.json and log/FB15K237_PEMLM_valResult.json**

you can add --weight_path xxx.pth --train_result_json_path xxx.json --valid_result_json_path xxx.json for customize your save path

---
#### Running PEMLM for **WN18RR**


```aiignore
python WN18RRmain.py --train_batch_size 256 --lr 3e-5 --epochs 200 
```
---
### Running PEMLM for **UMLS**
```aiignore
python UNLSmain.py --train_batch_size 256 --lr 1e-5 --epochs 200
```

Running with PEMLM
---
### Running PEMLM-F for **FB15k-237**

```aiignore
python FB15k237main-F.py \
--train_batch_size 256 \
-hidden_size 768  \
--lr 1e-5 \
--tran_lr 1e-4 \
--epochs 200 \
--alpha 1.0
```

---
### Running PEMLM-F for **WN18RR**

```aiignore
python WN18RRmain-F.py \
--train_batch_size 256 \
--hidden_size 768 \
--lr 3e-5 \
--tran_lr 1e-3 \
--epochs 200 \
--alpha 2.0
```
---
### Running PEMLM-F for **UMLS**

```aiignore
python UMLSmain-F.py \
--train_batch_size 256 \
--hidden_size 768 \
--lr 1e-5 \
--tran_lr 1e-4 \
--epochs 200 \
--alpha 0.5
```


## Citation
#### If our work inspires you or use our code in your research, we would greatly appreciate it if you could **STAR** this repository. 

    @inproceedings{qiu2024joint,
        title={Joint Pre-Encoding Representation and Structure Embedding for Efficient and Low-Resource Knowledge Graph Completion},
        author={Qiu, Chenyu and Qian, Pengjiang and Wang, Chuang and Yao, Jian and Liu, Li and Wei, Fang and Eddie, Eddie},
        booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
        pages={15257--15269},
            year={2024}
    }

[//]: # (## Contributing 贡献指南)

[//]: # ()
[//]: # (Please read [CONTRIBUTING.md]&#40;#&#41; for details on our code of conduct, and the process for submitting pull requests to us.)

[//]: # ()
[//]: # (清阅读 [CONTRIBUTING.md]&#40;#&#41; 了解如何向这个项目贡献代码)

[//]: # ()
[//]: # (## Release History 版本历史)

[//]: # ()
[//]: # (* 0.2.1)

[//]: # (    * CHANGE: Update docs)

[//]: # (* 0.2.0)

[//]: # (    * CHANGE: Remove `README.md`)

[//]: # (* 0.1.0)

[//]: # (    * Work in progress)

[//]: # ()
[//]: # (## Authors 关于作者)

[//]: # ()
[//]: # (* **WangYan** - *Initial work* - [WangYan]&#40;https://wangyan.org&#41;)

[//]: # ()
[//]: # (查看更多关于这个项目的贡献者，请阅读 [contributors]&#40;#&#41; )

[//]: # ()
[//]: # (## License 授权协议)

[//]: # ()
[//]: # (这个项目 MIT 协议， 请点击 [LICENSE.md]&#40;LICENSE.md&#41; 了解更多细节。)