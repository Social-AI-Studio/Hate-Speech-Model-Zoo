import os
import time 
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import config
import numpy as np
import h5py
import pickle as pkl
import json
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report,precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup,AdamW
    
def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

def compute_score(logits,labels):
    logits=torch.max(logits,1)[1]
    labels=torch.max(labels,1)[1]
    score=logits.eq(labels)
    score=score.sum().float()
    return score

def compute_other(logits,labels):
    acc=compute_score(logits,labels)
    logits=np.argmax(logits.cpu().numpy(),axis=1)
    label=np.argmax(labels.cpu().numpy(),axis=1)
    length=logits.shape[0]
    
    f1=f1_score(label,logits,average='weighted',labels=np.unique(label))
    recall=recall_score(label,logits,average='weighted',labels=np.unique(label))
    precision=precision_score(label,logits,average='weighted',labels=np.unique(label))
  
    m_f1=f1_score(label,logits,average='macro',labels=np.unique(label))
    m_recall=recall_score(label,logits,average='macro',labels=np.unique(label))
    m_precision=precision_score(label,logits,average='macro',labels=np.unique(label))
    result=classification_report(label,logits)
    print(result)
    info=result.split('\n')[2].split('     ')
    #print (info)
    h_p=float(info[3].strip())
    h_r=float(info[4].strip())
    h_f=float(info[5].strip())
    #print (float(info[2].strip()),float(info[3].strip()),float(info[4].strip()))
    return f1,recall,precision,acc,m_f1,m_recall,m_precision,h_p,h_r,h_f

def train_for_deep(test_set,model,opt,train_set,folder):
    num_class={'wz':3,'dt':3,'founta':4}
    if opt.DATASET=='dt':
        logger=utils.Logger(os.path.join(opt.DT_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='wz':
        logger=utils.Logger(os.path.join(opt.WZ_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    elif opt.DATASET=='founta':
        logger=utils.Logger(os.path.join(opt.FOUNTA_RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    
    log_hyperpara(logger,opt)
    train_size=len(train_set)
    test_size=len(test_set)
    train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=False,num_workers=1)
    if opt.MODEL=='BERT':
        optimizer=AdamW(model.parameters(),
                        lr=2e-5,
                        eps=1e-8
                       )
        num_training_steps=len(train_loader) * opt.EPOCHS
        scheduler=get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=0,
                                                  num_training_steps=num_training_steps
                                                 )
    else:
        optimizer=torch.optim.Adamax(model.parameters())
        
    for epoch in range(opt.EPOCHS):
        total_loss=0
        train_score=0.0
        eval_loss=0
        eval_score=0.0
        t=time.time()
        for i,(bert,basic,masks,labels,sent,char) in enumerate(train_loader):
            if opt.MODEL=='BERT':
                bert=bert.cuda()
                masks=masks.cuda()
                pred=model(bert,token_type_ids=None,attention_mask=masks)[0]
            else:
                char=char.cuda()
                basic=basic.cuda()
                pred=model(basic,char)
                
            labels=labels.float().cuda()
            loss=bce_for_loss(pred,labels)
            total_loss+=loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if opt.MODEL=='BERT':
                scheduler.step()#updating the learning rate
            optimizer.zero_grad()
            batch_score=compute_score(pred,labels)
            train_score+=batch_score
            if i==0:
                t_labels=labels
                t_pred=pred
            else:
                t_labels=torch.cat((t_labels,labels),0)
                t_pred=torch.cat((t_pred,pred),0)
        print ('Epoch', epoch,'for training loss:',total_loss)
        f1,recall,precision,acc,m_f1,m_recall,m_precision,_,_,_=compute_other(t_pred.detach(),t_labels)
        model.train(False)
        evaluate_score,test_loss,e_f1,e_recall,e_precision,m_f1,m_recall,m_precision,h_p,h_r,h_f=evaluate_for_offensive(model,test_loader,opt,epoch,folder)
        eval_score=100 * evaluate_score /test_size
        total_loss = total_loss /train_size
        train_score=100 * train_score / train_size
        e_f1=100.0 * e_f1 
        e_recall=100.0 * e_recall 
        e_precision=100.0 * e_precision 
        h_p=100*h_p
        h_r=100*h_r
        h_f=100*h_f
        print ('Epoch:',epoch,'evaluation score:',eval_score,' loss:',eval_loss)
        print ('Epoch:',epoch,'evaluation f1:',e_f1,' recall:',e_recall)
        logger.write('epoch %d, time: %.2f' %(epoch, time.time() -t))
        logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, train_score))
        logger.write('\teval accuracy: %.2f ' % ( eval_score))
        logger.write('\teval f1: %.2f ' % ( e_f1))
        logger.write('\teval precision: %.2f ' % ( e_precision))
        logger.write('\teval recall: %.2f ' % (e_recall))
        logger.write('\teval macro f1: %.2f ' % ( h_f))
        logger.write('\teval macro precision: %.2f ' % (h_p))
        logger.write('\teval macro recall: %.2f ' % (h_r))
        model.train(True)
    return eval_score,e_f1,e_precision,e_recall,h_f,h_r,h_p
    
def evaluate_for_offensive(model,test_loader,opt,epoch,folder):
    score=0.0
    total_loss=0
    f1=0.0
    precision=0.0
    recall=0.0
    acc=0.0
    total_num=len(test_loader.dataset)
    print ('The length of the loader is:',len(test_loader.dataset))
    for i,(bert,basic,masks,labels,sent,char) in enumerate(test_loader):
        with torch.no_grad():
            if opt.MODEL=='BERT':
                bert=bert.cuda()
                masks=masks.cuda()
                pred=model(bert,token_type_ids=None,attention_mask=masks)[0]
            elif opt.MODEL=='HYBRID':
                basic=basic.cuda()
                char=char.cuda()
                pred=model(basic,char)
            labels=labels.float().cuda()
        batch_score=compute_score(pred,labels)
        batch_loss=bce_for_loss(pred,labels)
        total_loss+=batch_loss
        score+=batch_score
        _,prediction=torch.max(pred,dim=1)
        prediction=prediction.detach().cpu().numpy()
        if i==0:
            t_labels=labels
            t_pred=pred
        else:
            t_labels=torch.cat((t_labels,labels),0)
            t_pred=torch.cat((t_pred,pred),0)
    f1,recall,precision,acc,m_f1,m_recall,m_precision,h_p,h_r,h_f=compute_other(t_pred,t_labels)
    avg_loss=total_loss 
    return score,avg_loss,f1,recall,precision,m_f1,m_recall,m_precision,h_p,h_r,h_f    
            
            
            