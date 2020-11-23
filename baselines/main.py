import torch
import torch.nn as nn
from torch.utils.data import Subset,ConcatDataset

from dataset import Wraped_Data
from train import train_for_deep
import preprocessing
import baseline
import utils
import config

import os
import pickle as pkl

if __name__=='__main__':
    opt=config.parse_opt()
    constructor='build_baseline'
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    
    #result saving
    if opt.DATASET=='wz':
        logger=utils.Logger(os.path.join(opt.WZ_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
        dict_path=os.path.join(opt.WZ_DATA,'dictionary.pkl')
        glove_path=os.path.join(opt.WZ_DATA,'glove.npy')
    elif opt.DATASET=='dt':
        logger=utils.Logger(os.path.join(opt.DT_FULL_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
        dict_path=os.path.join(opt.DT_DATA,'dictionary.pkl')
        glove_path=os.path.join(opt.DT_DATA,'glove.npy')
    elif opt.DATASET=='founta':
        logger=utils.Logger(os.path.join(opt.FOUNTA_RESULT,'final_'+str(opt.SAVE_NUM)+'.txt'))
        dict_path=os.path.join(opt.FOUNTA_DATA,'dictionary.pkl')
        glove_path=os.path.join(opt.FOUNTA_DATA,'glove.npy')
    
    #definitions for criteria
    score=0.0
    f1=0.0
    recall=0.0
    precision=0.0
    m_f1=0.0
    m_recall=0.0
    m_precision=0.0
    
    split_dataset=pkl.load(open(os.path.join(opt.SPLIT_DATASET,opt.DATASET+'.pkl'),'rb'))
    
    if opt.CREATE_DICT==True:
        print ('Creating the dictionary...')
        word2idx,idx2word=preprocessing.create_dict(opt,split_dataset,dict_path)
    else:
        print ('Loading the dictionary...')
        word2idx,idx2word=utils.load_pkl(dict_path)
        
    ntokens=len(idx2word)
    if opt.CREATE_EMB==True:
        print ('Creating the embeddings...')
        preprocessing.create_emb(opt,glove_path,ntokens,idx2word)
        
    for i in range(opt.CROSS_VAL):
        train_set=Wraped_Data(opt,word2idx,split_dataset,i)
        test_set=Wraped_Data(opt,word2idx,split_dataset,i,'test')
        model=getattr(baseline,constructor)(ntokens,glove_path,opt).cuda()
        s,f,p,r,m_f,m_r,m_p=train_for_deep(test_set,model,opt,train_set,i)
        score+=s
        f1+=f
        precision+=p
        recall+=r
        m_f1+=m_f
        m_precision+=m_p
        m_recall+=m_r
        logger.write('validation folder %d' %(i+1))
        logger.write('\teval score: %.2f ' % (s))
        logger.write('\teval precision: %.2f ' % (p))
        logger.write('\teval recall: %.2f ' % (r))
        logger.write('\teval f1: %.2f ' % (f))
        logger.write('\teval macro precision: %.2f ' % (m_p))
        logger.write('\teval macro recall: %.2f ' % (m_r))
        logger.write('\teval macro f1: %.2f ' % (m_f))
        
    score/=opt.CROSS_VAL
    f1/=opt.CROSS_VAL
    precision/=opt.CROSS_VAL
    recall/=opt.CROSS_VAL
    m_f1/=opt.CROSS_VAL
    m_precision/=opt.CROSS_VAL
    m_recall/=opt.CROSS_VAL
    logger.write('\n final result')
    logger.write('\teval score: %.2f ' % (score))
    logger.write('\teval precision: %.2f ' % (precision))
    logger.write('\teval recall: %.2f ' % (recall))
    logger.write('\teval f1: %.2f ' % (f1))
    logger.write('\teval hate precision: %.2f ' % (m_precision))
    logger.write('\teval hate recall: %.2f ' % (m_recall))
    logger.write('\teval hate f1: %.2f ' % (m_f1))
    exit(0)
    