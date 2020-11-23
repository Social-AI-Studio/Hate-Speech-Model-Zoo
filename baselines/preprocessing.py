from collections import defaultdict
import os
import pickle as pkl
import numpy as np

def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb')) 
    
def create_dict(opt,split_data,save_path):
    word2idx={}
    idx2word=[]
    word_count=defaultdict(int)
    
    for i in range(opt.CROSS_VAL):
        info=split_data[str(i)]
        for row in info:
            text=row['sent']
            tokens=text.split(' ')
            for t in tokens:
                word_count[t]+=1
                
    cur=0
    for word in word_count.keys():
        if word_count[word]>opt.MIN_OCC:
            word2idx[word]=cur
            cur+=1
            idx2word.append(word)
    if 'UNK' not in word2idx:
        word2idx['UNK']=cur
        idx2word.append('UNK')
    dump_pkl(save_path,[word2idx,idx2word])
    return word2idx,idx2word

def create_emb(opt,emb_dir,ntokens,idx2word):
    word2emb={}
    with open(opt.GLOVE_PATH,'r') as f:
        entries=f.readlines()
    emb_dim=len(entries[0].split(' '))-1
    weights=np.zeros((len(idx2word),emb_dim),dtype=np.float32)
    for entry in entries:
        word=entry.split(' ')[0]
        word2emb[word]=np.array(list(map(float,entry.split(' ')[1:])))
    for idx,word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx]=word2emb[word]
            
    np.save(os.path.join(emb_dir),weights)