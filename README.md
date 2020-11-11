# Hate Speech Model Zoo 

A public repository compilation of existing methods and commonly used datasets for hate speech detection. This repository is used for ASONAM20 Tutorial: "Perils and Promises of Automated Hate Speech Detection"

## Problem Overview  

Hate speech is defined as: public speech that expresses hate or en- courages violence towards a person or group based on something such as race, religion, sex, or sexual orientation. The goal of hate speech detection is to correctly identify hate speech among texts. It can be regarded as a classification task. 

## Dependencies:  

- Python 3.6.9
- Pytorch 1.6.0

## Data for Training HateGAN
Datasets are provided in the folder named as **data**. For each dataset, we provide two versions of it: the raw data and the split dataset (split into 5-folders randomly).

| Dataset | Label (Count)                                     |
| :-----: | :-----------------------------------------------: | 
| [WZ](https://www.aclweb.org/anthology/W17-3006.pdf)      | hate (3,435) non-hate (9,767)                     |
| [DT](https://arxiv.org/pdf/1703.04009.pdf)      | hate (1,430) offensive (19,190) beither (4,163)   |
| [FOUNTA](https://arxiv.org/pdf/1802.00393.pdf)  | hate (3,907) abusive (19,232) spam (13,840) normal (53,011)      |  

## Methods for Hate Speech Detection 
Here is the current list of models currently implemented in the project:

- LSTM: It applies LSTM for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- CNN: It applies CNN for extracting textual information. The semantic information will be fed to a classification layer for the final prediction. 
- Bert: It applies Pretrained-Bert model for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- [CNN-GRU](http://eprints.whiterose.ac.uk/128405/8/chase.pdf): It combines CNN and GRU for extracting textual information. The semantic information will be fed to a classification layer for the final prediction.
- [HybridCNN](https://www.aclweb.org/anthology/W17-3006.pdf): It exploits both the word-level information and the character-level information. Two CNNs are used to get each level of representation. Word-level and character-level representations are concatenated and sent to the classification layer.
- [DeepHate](https://dl.acm.org/doi/fullHtml/10.1145/3394231.3397890): It considers multi-faceted text representations for hate speech detection: word embeddings, sentiment embeddings and the topic information. Different information are aggregated through gate attention and the joint representation is fed to the classification layer. 

## Existing Problems 

- **Imbalance of datasets**: Most existing datasets for hate speech detection are highly imbalanced as there are fewer hate texts compared with other classes. One solution is to do data augmentation. However, it is expensive and time-consuming. A few pieces of work have tried some automantic methods of data augmentation.   

- **Bias of Models**: Since models are trained on an imbalanced dataset, they suffers from the problem of biased prediction. The problem of bias is visualized and code is shared under the folder of **vis-bias**. In order to solve the problem, [Bert+SOC](Contextualizing Hate Speech Classifiers with Post-hoc Explanation) has been used for debias in hate speech detection by adding the regularization term.

## Citation
If you use our codes in your project, please cite:
```
@misc{asonam2020hsp,
  title =        {Hate Speech Detection Tutorial, ASONAM 2020},
  howpublished = {\url{https://gitlab.com/bottle_shop/safe/hate-speech-model-zoo}},
  year =         {2020}
}
```