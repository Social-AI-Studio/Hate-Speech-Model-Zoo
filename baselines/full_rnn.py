import torch
import torch.nn as nn
import numpy as np
import config
from torch.autograd import Variable
import torch.nn.functional as F
    
class CNN_Model(nn.Module):
    def __init__(self,in_dim,filter_size,num_filter):
        super(CNN_Model,self).__init__()
        self.in_dim=in_dim
        filter_sizes=[int(fsz) for fsz in filter_size.split(',')]
        self.conv=nn.ModuleList([nn.Conv2d(1,num_filter,(fsz,in_dim)) for fsz in filter_sizes])
        self.pool=nn.MaxPool1d(kernel_size=4, stride=4)
        
    def forward(self,emb):
        emb=emb.unsqueeze(1)#B,1,L,D
        conv_result=[F.relu(conv(emb)) for conv in self.conv]
        pool_result=[F.max_pool2d(input=x_i,kernel_size=(x_i.shape[2],x_i.shape[3])) for x_i in conv_result]
        mid=[torch.squeeze(x_i) for x_i in pool_result]
        final=torch.cat(mid,1)
        return final

class Hybrid_CNN(nn.Module):
    def __init__(self,c_emb,w_cnn,c_cnn,dropout):
        super(Hybrid_CNN,self).__init__()
        self.c_emb=c_emb
        self.w_cnn=w_cnn
        self.c_cnn=c_cnn
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,emb,char):
        conv_w=self.w_cnn(emb)
        conv_w=self.dropout(conv_w)
        char_emb=self.c_emb(char)
        conv_c=self.c_cnn(char_emb)
        conv_c=self.dropout(conv_c)
        final=torch.cat((conv_w,conv_c),dim=1)
        return final
    
class CNN_GRU(nn.Module):
    def __init__(self,in_dim,filter_size,num_filter,dropout,gru):
        super(CNN_GRU,self).__init__()
        self.in_dim=in_dim
        filter_sizes=[int(fsz) for fsz in filter_size.split(',')]
        self.conv=nn.ModuleList([nn.Conv2d(1,num_filter,(fsz,in_dim)) for fsz in filter_sizes])
        self.gru=gru
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,emb):
        emb=emb.unsqueeze(1)
        conv_result=F.relu(self.conv[0](emb)) 
        mid=torch.squeeze(conv_result).transpose(1,2)
        result=self.dropout(self.gru(mid))
        return result
    
class Part_RNN(nn.Module):
    def __init__(self,in_dim,num_hidden,num_layer,bidirect,dropout,rnn_type='LSTM'):
        super(Part_RNN,self).__init__()
        rnn_cls=nn.LSTM if rnn_type=='LSTM' else nn.GRU
        self.rnn=rnn_cls(in_dim,num_hidden,num_layer,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.num_layer=num_layer
        self.rnn_type=rnn_type
        self.num_bidirect=1+int(bidirect)
        
    def init_hidden(self,batch):
        weight=next(self.parameters()).data
        hid_shape=(self.num_layer * self.num_bidirect,batch,self.num_hidden)
        if self.rnn_type =='LSTM':
            return (Variable(weight.new(*hid_shape).zero_().cuda()),
                    Variable(weight.new(*hid_shape).zero_().cuda()))
        else:
            return Variable(weight.new(*hid_shape).zero_()).cuda()
    
    def forward(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        return output[:,-1,:]