import torch
import torch.nn as nn
import config
import numpy as np
import torch.nn.functional as F
from transformers import BertForSequenceClassification,BertConfig

from full_rnn import Part_RNN, CNN_Model, CNN_GRU, Hybrid_CNN
from language_model import Word_Embedding, Char_Embedding
from classifier import SimpleClassifier,SingleClassifier

class Deep_Basic(nn.Module):
    def __init__(self,w_emb,encoder,fc,opt):
        super(Deep_Basic,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.fc=fc
        self.encoder=encoder
        
        self.model=opt.MODEL
        
    def forward(self,basic,char):
        if self.model=='HYBRID':
            w_emb=self.w_emb(basic)
            repre=self.encoder(w_emb,char)
        else:
            w_emb=self.w_emb(basic)
            repre=self.encoder(w_emb)
        logits=self.fc(repre)
        
        return logits 
        
        
def build_baseline(ntokens,emb_dir,opt): 
    opt=config.parse_opt()
    
    if opt.DATASET=='dt' or opt.DATASET=='wz' :
        final_dim=3
    elif opt.DATASET=='founta':
        final_dim=4
    
    if opt.MODEL=='BERT':
        model=BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
             num_labels=final_dim,
             output_attentions=False,
             output_hidden_states=False
        )    
        return model
    
    w_emb=Word_Embedding(ntokens,opt.EMB_DIM,opt.EMB_DROPOUT) 
    print ('Initializing word embeddings...')
    w_emb.init_embedding(emb_dir)
    
    if opt.MODEL=='LSTM':
        encoder=Part_RNN(opt.EMB_DIM,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
        fc=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    elif opt.MODEL=='CNN':
        encoder=CNN_Model(opt.EMB_DIM,opt.FILTER_SIZE,opt.NUM_FILTER)
        fc=SimpleClassifier(opt.NUM_FILTER * len(opt.FILTER_SIZE.split(',')),
                            opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    elif opt.MODEL=='CNNGRU':
        gru=Part_RNN(opt.NUM_FILTER,opt.NUM_HIDDEN,opt.NUM_LAYER, 
                     opt.BIDIRECT,opt.L_RNN_DROPOUT,rnn_type='GRU')
        encoder=CNN_GRU(opt.EMB_DIM,opt.CG_FILTER_SIZE,opt.NUM_FILTER,opt.FC_DROPOUT,gru)
        fc=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    elif opt.MODEL=='HYBRID':
        c_emb=Char_Embedding(opt.EMB_DIM,opt.EMB_DROPOUT) 
        w_cnn=CNN_Model(opt.EMB_DIM,opt.W_FILTER_SIZE,opt.NUM_FILTER)
        c_cnn=CNN_Model(opt.EMB_DIM,opt.C_FILTER_SIZE,opt.NUM_FILTER)
        encoder=Hybrid_CNN(c_emb,w_cnn,c_cnn,opt.FC_DROPOUT)
        fc=SimpleClassifier(
            opt.NUM_FILTER * (len(opt.W_FILTER_SIZE.split(','))+len(opt.C_FILTER_SIZE.split(','))),
            opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    return Deep_Basic(w_emb,encoder,fc,opt)
    
    