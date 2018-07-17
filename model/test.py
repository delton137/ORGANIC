from organic import *

params = {
#    'MAX_LENGTH': 217,
    'GEN_ITERATIONS': 3,  #default 4
    'DIS_EPOCHS': 1,      #default 8
    'DIS_BATCH_SIZE': 20, #default 30
    'GEN_BATCH_SIZE': 20, #default 30
    'GEN_EMB_DIM': 32,
    'SAMPLE_NUM' : 3200, #default 6400
    'BIG_SAMPLE_NUM' :  16000, #default 32000
    'LAMBDA'      :   0.5,
    'DIS_EMB_DIM': 32,
    'DIS_FILTER_SIZES': [5, 10, 15],
    'DIS_NUM_FILTERS': [100, 100, 100],
    'DIS_DROPOUT' :   0.85,
    'DIS_L2REG' :   0.3,
    #'CHK_PATH' : '/home/delton/Dropbox/AAA_UMD_RESEARCH/ORGANIC/model/checkpoints/energetics_test2/checkpoints/energetics_test2/'
}

model = ORGANIC('energetics_test10', params=params)
#model.load_training_set('../data/trainingsets/small_mols.csv')
#model.load_prev_training("energetics_test2")
model.load_training_set('../data/datasets/energetics_list_cleaned.csv')
model.load_prev_pretraining(ckpt='test_ckpt/energetics_test9_pretrain_ckpt')

#energetics test 9
metrics = ['det_vel','original_synthesizability','diversity','soft_novelty']
ratios = [3,2,2,1]


#metrics = ['det_vel']
#metrics = ['original_synthesizability','diversity']
#ratios = [2,1,1]

total_epochs = 200
education_metrics = []
education_epochs = []
for i in range(total_epochs//len(metrics)):
    for j in range(len(metrics)):
        education_metrics += [metrics[j]]
        education_epochs += [ratios[j]]
print("total epochs in education = ", sum(education_epochs))

model.set_training_program(education_metrics, education_epochs)
model.load_metrics()
model.train(ckpt_dir='test_ckpt')
