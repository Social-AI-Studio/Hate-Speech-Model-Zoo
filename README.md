# Hate Speech Model Zoo 

A public repository compilation of existing methods and commonly used datasets for hate speech detection. This repository is used for ASONAM20 Tutorial: "Perils and Promises of Automated Hate Speech Detection"

## Problem Overview  

Hate speech is defined as: public speech that expresses hate or en- courages violence towards a person or group based on something such as race, religion, sex, or sexual orientation. The goal of hate speech detection is to correctly identify hate speech among texts. It can be regarded as a classification task. 

## Dependencies:  

- Python 3.6.9
- Pytorch 1.6.0

## Data for Training HateGAN

| Dataset | Label (Count)                                     |
| :-----: | :-----------------------------------------------: | 
| WZ \cite{DBLP:conf/acl-alw/ParkF17}      | hate (3,435) non-hate (9,767)                     |
| DT      | hate (1,430) offensive (19,190) beither (4,163)   |
| FOUNTA  | hate (3,907) abusive (19,232) spam (13,840) normal (53,011)      |  

## Reference  
Referred paper:
```
@inproceedings{DBLP:conf/acl-alw/ParkF17,
  author    = {Ji Ho Park and
               Pascale Fung},
  title     = {One-step and Two-step Classification for Abusive Language Detection
               on Twitter},
  booktitle = {Proceedings of the First Workshop on Abusive Language Online, ALW@ACL
               2017},
  pages     = {41--45},
  publisher = {Association for Computational Linguistics},
  year      = {2017}
}
```