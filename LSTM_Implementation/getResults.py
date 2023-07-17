#Uses the trained models to get the predictions for the test dataset which are used to create the submission.csv file

import pandas as pd
import torch
import datasetHandler
import VS_LSTM
from torch.utils.data import Dataset, DataLoader

label_type_dict = {
    0 : 'harsh',
    1 : 'extremely_harsh',
    2 : 'vulgar',
    3 : 'threatening',
    4 : 'disrespect',
    5 : 'targeted_hate'
}

def appendToDF(df, label_type, dataset):
    batchSize = 1
    
    inputSize = len(dataset[0][0][0])
    num_layers = 3
    learning_rate = 0.0005
    
    model = VS_LSTM.LSTM(num_layers, inputSize*2, inputSize)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    
    FILE = label_type+"_checkpoint.pth"
    checkpoint = torch.load(FILE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    batchNum = checkpoint['batch_number']
    
    test_loader = DataLoader(dataset=dataset,
                      batch_size=batchSize,
                      shuffle=False,
                      num_workers=0)      
    
    class_probs = []
    
    ##########################################################
    for i, (inputs, labels, seqLens) in enumerate(test_loader):
        seqLens = seqLens.view(seqLens.size(0))
        seqLens = [int(x) for x in seqLens]
        # Forward pass and loss
        y_pred = model(inputs, seqLens)
        class_probs.append(y_pred[0][1].item())
    
    print("Class_Probs_Length: "+str(len(class_probs)))
    df[label_type] = class_probs
    return df


dataset = datasetHandler.LSTMdataset("Datasets", "TestDataset.txt")
test_df = pd.read_csv("harsh-comment-classification/test.csv")
df = pd.DataFrame([])
df['id'] = test_df['id']
for i in range(6):
    df = appendToDF(df, label_type_dict[i], dataset)
    
df.to_csv("submission.csv", index=False)
