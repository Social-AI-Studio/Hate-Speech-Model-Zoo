import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    
    #all models are named using capitals 
    parser.add_argument('--MODEL',type=str,default='LSTM')
    
    '''path configuration'''
    #path for pre-precessing and result saving
    parser.add_argument('--DT_RESULT',type=str,default='./dt/result')
    parser.add_argument('--DT_DATA',type=str,default='./dt/dictionary')
    parser.add_argument('--WZ_RESULT',type=str,default='./wz/result')
    parser.add_argument('--WZ_DATA',type=str,default='./wz/dictionary')
    parser.add_argument('--FOUNTA_RESULT',type=str,default='./founta/result')
    parser.add_argument('--FOUNTA_DATA',type=str,default='./founta/dictionary')
    
    #path for the split dataset
    parser.add_argument('--SPLIT_DATASET',type=str,default='../dataset/split_data')
    parser.add_argument('--GLOVE_PATH',type=str,default='/home/ruicao/trained/embeddings/glove.6B.300d.txt')
    
    
    #basic configurations
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--CUDA_DEVICE', type=int, default=0)
    parser.add_argument('--TEST_NUM',type=int,default=0)
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    parser.add_argument('--CROSS_VAL',type=int,default=5)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    #dt for DT, fouta for FOUNTA, wz for WZ
    parser.add_argument('--DATASET',type=str,default='wz')
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    
    #hyper parameters
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    #for bert we set it as 64, for other baselines, we set it as 30
    parser.add_argument('--LENGTH',type=int,default=64)
    parser.add_argument('--EPOCHS',type=int,default=6)
    parser.add_argument('--MIN_OCC',type=int,default=3)
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.5) 
    parser.add_argument('--MID_DIM',type=int,default=512)
    parser.add_argument('--NUM_HIDDEN',type=int,default=1024)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    
    args=parser.parse_args()
    return args
