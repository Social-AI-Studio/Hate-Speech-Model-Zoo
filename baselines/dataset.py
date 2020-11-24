import os
import pandas as pd
import json
import pickle as pkl
import numpy as np
import torch
from tqdm import tqdm
import config
import random

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
        
class Wraped_Data():
    def __init__(self,opt,word2idx,split_data,test_num,mode='training'):
        super(Wraped_Data,self).__init__()
        self.opt=config.parse_opt()
        self.split_data=split_data
        self.test_num=test_num
        self.mode=mode
        self.word2idx=word2idx
        self.ntokens=len(word2idx)
        
        if self.opt.DATASET=='dt' or self.opt.DATASET=='wz':
            self.classes=3
        elif self.opt.DATASET=='founta':
            self.classes=4
        
        if opt.DEBUG:
            self.entries=self.load_tr_val_entries()[:200]
        else:
            self.entries=self.load_tr_val_entries()
        
        self.tokenize()
        self.tensorize()
        
    def load_tr_val_entries(self):
        all_data=[]
        #loading dataset for training and testing
        if self.mode=='training':
            for i in range(self.opt.CROSS_VAL):
                if i==self.test_num:
                    continue
                all_data.extend(self.split_data[str(i)])
        else:
            all_data.extend(self.split_data[str(self.test_num)])
            
        entries=[]
        count=0
        for info in all_data:
            sent=info['bert_token']
            label=info['label']
            entry={
                'bert':sent,
                'text':info['sent'].lower(),
                'answer':label
            }
            entries.append(entry)
        return entries
    
    def padding_sent_bert(self,tokens,length):
        if len(tokens)<length:
            padding=[0]*(length-len(tokens))
            tokens=tokens+padding
        else:
            tokens=tokens[:length]
        return tokens
    
    def padding_sent_basic(self,tokens,length):
        if len(tokens)<length:
            padding=[self.ntokens]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
    
    def padding_char(self,tokens,length):
        if len(tokens)<length:
            padding=[26]*(length-len(tokens))
            tokens=padding+tokens
        else:
            tokens=tokens[:length]
        return tokens
   
    def matching(self,text):
        tokens=text.split(' ')
        token_num=[]
        for t in tokens:
            if t in self.word2idx:
                token_num.append(self.word2idx[t])
            else:
                token_num.append(self.word2idx['UNK'])
        return token_num
    
    def char_matching(self,chars):
        token_num=[]
        for c in chars:
            t=ord(c)-97
            if t>=26 or t<0:
                token_num.append(26)
            else:
                token_num.append(t)
        return token_num

    def tokenize(self):
        print('Tokenize Tweets...')
        length=self.opt.LENGTH
        char_length=self.opt.CHAR_LENGTH
        for entry in tqdm(self.entries):
            tokens=entry['bert']
            pad_tokens=self.padding_sent_bert(tokens,length)
            entry['bert_tokens']=np.array((pad_tokens),dtype=np.int64)
            mask=[int(num>0) for num in pad_tokens]
            entry['masks']=np.array((mask),dtype=np.int64)
            
            tokens=self.matching(entry['text'])
            pad_tokens=self.padding_sent_basic(tokens,length)
            entry['basic_tokens']=np.array((pad_tokens),dtype=np.int64)
            
            chars=list(''.join(entry['text'].split(' ')))
            char_tokens=self.char_matching(chars)
            pad_tokens=self.padding_char(char_tokens,char_length)
            entry['char']=np.array((pad_tokens),dtype=np.int64)
            
    def tensorize(self):
        print ('Tesnsorize all Information...')
        count=0
        for entry in tqdm(self.entries):
            entry['bert_tokens']=torch.from_numpy(entry['bert_tokens'])
            entry['basic_tokens']=torch.from_numpy(entry['basic_tokens'])
            entry['char']=torch.from_numpy(entry['char'])
            target=torch.from_numpy(np.zeros((self.classes),dtype=np.float32))
            target[entry['answer']]=1.0
            entry['label']=target
            entry['masks']=torch.from_numpy(entry['masks'])
                       
    def __getitem__(self,index):
        entry=self.entries[index]
        bert=entry['bert_tokens']
        basic=entry['basic_tokens']
        label=entry['label']
        masks=entry['masks']
        sent=entry['text']
        char=entry['char']
        return bert,basic,masks,label,sent,char
        
        
    def __len__(self):
        return len(self.entries)
    
