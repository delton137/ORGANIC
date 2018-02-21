from organic import * 

params = {
    'MAX_LENGTH': 16,
    'GEN_ITERATIONS': 1,
    'DIS_EPOCHS': 1,
    'DIS_BATCH_SIZE': 30,
    'GEN_BATCH_SIZE': 30,
    'GEN_EMB_DIM': 32,
    'DIS_EMB_DIM': 32,
    'DIS_FILTER_SIZES': [5, 10, 15],
    'DIS_NUM_FILTERS': [100, 100, 100]
}

model = ORGANIC('tutorial1', params=params)                            
model.load_training_set('../data/trainingsets/toy.csv') 
model.set_training_program(['logP'], [5])               
model.load_metrics()                         
model.train() 


