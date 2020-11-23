import torch
import torch.nn as nn
import numpy as np
import config
import os
import torch.nn.functional as F
from torch.autograd import Variable
import math

opt=config.parse_opt()
class Word_Embedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(Word_Embedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self,emb_dir):
        print ('Initializing glove Embedding...')
        glove_weight=torch.from_numpy(np.load(emb_dir))
        #glove_weight=torch.from_numpy(np.load('./glove_embedding.npy'))
        self.emb.weight.data[:self.ntoken]=glove_weight
  
    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(emb)
        return emb

        