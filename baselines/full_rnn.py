import torch
import torch.nn as nn
import numpy as np
import config
from torch.autograd import Variable
import torch.nn.functional as F
    
class CNN_Model(nn.Module):
    def __init__(self,in_dim,num_hidden,opt):
        super(CNN_Model,self).__init__()
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.dropout=opt.CNN_DROPOUT
        filter_sizes=[int(fsz) for fsz in opt.FILTER_SIZE.split(',')]
        self.conv=nn.ModuleList([nn.Conv2d(1,opt.NUM_FILTER,(fsz,in_dim)) for fsz in filter_sizes])
        
    def forward(self,emb):
        emb=emb.unsqueeze(1)
        conv_result=[F.relu(conv(emb)) for conv in self.conv]
        mid=[torch.squeeze(x_i).transpose(1,2).contiguous() for x_i in conv_result]
        
        return mid
    
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