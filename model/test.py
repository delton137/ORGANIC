from organic import *

params = {
#    'MAX_LENGTH': 16,
    'GEN_ITERATIONS': 4,
    'DIS_EPOCHS': 8,
    'DIS_BATCH_SIZE': 30,
    'GEN_BATCH_SIZE': 30,
    'GEN_EMB_DIM': 32,
    'DIS_EMB_DIM': 32,
    'DIS_FILTER_SIZES': [5, 10, 15],
    'DIS_NUM_FILTERS': [100, 100, 100],
    'DIS_DROPOUT'              :   0.85,
    'DIS_L2REG' :   0.3,
  # 'CHK_PATH' : '/home/delton/Dropbox/AAA_UMD_RESEARCH/ORGANIC/model/test_ckpt'
}

model = ORGANIC('energetics_test2', params=params)
#model.load_training_set('../data/trainingsets/small_mols.csv')
model.load_training_set('../data/datasets/energetics_list.csv')
model.set_training_program(['diversity',  'variety','soft_novelty'], [50,50,50] )
model.load_metrics()
model.train(ckpt_dir='test_ckpt')
