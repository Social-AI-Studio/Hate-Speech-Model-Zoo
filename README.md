# Hate Speech Model Zoo 

A public repository compilation of existing methods and commonly used datasets for hate speech detection. This repository is used for ASONAM20 Tutorial: "Perils and Promises of Automated Hate Speech Detection"

## Problem Overview  

Hate speech is defined as: public speech that expresses hate or en- courages violence towards a person or group based on something such as race, religion, sex, or sexual orientation. The goal of hate speech detection is to correctly identify hate speech among texts. It can be regarded as a classification task. 

## Dependencies:  

- Python 3.6.9
- Pytorch 1.6.0

## Data for Hate Speech Detection
Datasets are provided in the folder named as **data**. For each dataset, we provide data which has been preprocessed and split into 5-folders. For raw data, please refer to their original project.

| Dataset | Label (Count)                                     |
| :-----: | :-----------------------------------------------: | 
| [WZ](https://www.aclweb.org/anthology/W17-3006.pdf)      | racism (1,923) sexism (3,079) neither (11,033)                     |
| [DT](https://arxiv.org/pdf/1703.04009.pdf)      | hate (1,430) offensive (19,190) beither (4,163)   |
| [FOUNTA](https://arxiv.org/pdf/1802.00393.pdf)  | hate (3,907) abusive (19,232) spam (13,840) normal (53,011)      |  

## Methods for Hate Speech Detection 
Here is the current list of models currently implemented in the project:

- LSTM: It applies LSTM for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- CNN: It applies CNN for extracting textual information. The semantic information will be fed to a classification layer for the final prediction. 
- Bert: It applies Pretrained-Bert model for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- [CNN-GRU](http://eprints.whiterose.ac.uk/128405/8/chase.pdf): It combines CNN and GRU for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- [HybridCNN](https://www.aclweb.org/anthology/W17-3006.pdf): It exploits both the word-level information and the character-level information. Two CNNs are used to get each level of representation. Word-level and character-level representations are concatenated and sent to the classification layer.
- [DeepHate](https://dl.acm.org/doi/fullHtml/10.1145/3394231.3397890): It considers multi-faceted text representations for hate speech detection: word embeddings, sentiment embeddings and the topic information. Different information are aggregated through gate attention and the joint representation is fed to the classification layer. For more detail, please refer to the [project](https://gitlab.com/bottle_shop/safe/deephate/-/tree/master/)

## Existing Problems 
Most recent models for hate speech are supervised and highly rely on annotated datasets. As a consequence, the quality of datasets affects the performance of models. However, most of the datasets are imbalanced. To be more specific, hate tweets are fewer compared with tweets in other classes. There are two main solutions: 1) data augmentation 2) debias of models.

- **Data Augmentation**: Adding more data is the most straightforward way to solve the problem of imbalance of datasets. [Rizos et al., 2019](https://www.researchgate.net/publication/337018946_Augment_to_Prevent_Short-Text_Data_Augmentation_in_Deep_Learning_for_Hate-Speech_Classification) tried substitution and swapping words of original tweets for augmentation. They are detrimental to the fluency of sentences. They also tried to generate tweets with pretrained language model, however, ignoring attributes of hate tweets. [HateGAN](https://www.aclweb.org/anthology/2020.coling-main.557.pdf) is a deep generative reinforcement learning model, which addresses the problem of imbalance by augmenting the dataset with hateful tweets.  

- **Debias of Models**: Since models are trained on an imbalanced dataset, they suffers from the problem of biased prediction. The problem of bias is visualized and code is shared under the folder of **bias**. In order to solve the problem, [Bert+SOC](https://www.aclweb.org/anthology/2020.acl-main.483/) has been used for debias in hate speech detection by adding the regularization term.

## Citation
If you use our codes in your project, please cite:

### To cite 
```
@article{lee2020tutorial,
  title={Perils and Promises of Automated Hate Speech Detection},
  author={Lee, Roy Ka-Wei and Cao, Rui},
  journal={arXiv preprint},
  year={2020}
}
```
