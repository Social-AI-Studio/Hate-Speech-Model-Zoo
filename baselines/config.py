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
    parser.add_argument('--DEBUG',type=bool,default=False)
    
    #hyper parameters
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    #for bert we set it as 64, for other baselines, we set it as 30
    parser.add_argument('--LENGTH',type=int,default=64)
    parser.add_argument('--EPOCHS',type=int,default=6)
    parser.add_argument('--MIN_OCC',type=int,default=3)
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.5) 
    parser.add_argument('--MID_DIM',type=int,default=512)
    #LSTM
    parser.add_argument('--NUM_HIDDEN',type=int,default=1024)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    #CNN
    parser.add_argument('--NUM_FILTER',type=int,default=150)
    parser.add_argument('--FILTER_SIZE',type=str,default="2,3,4")
    #CNN-GRU
    parser.add_argument('--CG_FILTER_SIZE',type=str,default="4")
    parser.add_argument('--GRU_HIDDEN',type=int,default=128)
    #Hybrid CNN
    parser.add_argument('--W_FILTER_SIZE',type=str,default="1,2,3")
    parser.add_argument('--C_FILTER_SIZE',type=str,default="3,4,5")
    parser.add_argument('--CHAR_LENGTH',type=int,default=64)
    args=parser.parse_args()
    return args
