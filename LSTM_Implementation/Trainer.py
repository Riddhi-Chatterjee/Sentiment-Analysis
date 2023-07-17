#This is the training script which is used to train the LSTM model on the training dataset

import datasetHandler
import VS_LSTM
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import exists
import math

def signal_handler(sig, frame):
    checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "batch_number" : batchNum,
    }
    FILE = label_type+"_checkpoint.pth"
    torch.save(checkpoint, FILE)
    
    with open(label_type+"_log.out", 'a') as lg:
        lg.write('\nExiting...\n')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

label_type_dict = {
    0 : 'harsh',
    1 : 'extremely_harsh',
    2 : 'vulgar',
    3 : 'threatening',
    4 : 'disrespect',
    5 : 'targeted_hate'
}
label_type_index = int(sys.argv[1]) #INPUT - 1
label_type = label_type_dict[label_type_index]

learning_rate = 0.0005
        
num_epochs = 10000 
num_layers = 3
batchSize = 1
printingBatch = 0

#Settings:
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
criterion = nn.MSELoss().to(device)
dataset = datasetHandler.LSTMdataset("Datasets", label_type+"_TrainDataset.txt")
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batchSize)
inputSize = len(dataset[0][0][0])
model = VS_LSTM.LSTM(num_layers, inputSize*2, inputSize)
#model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

epoch = 0
batchNum = 0
loss = "Dummy Initialisation"

ch1 = sys.argv[2] #INPUT - 2
if ch1.upper() == "Y":
    with open(label_type+"_checkpoint.pth", "w") as c:
        pass
else:
    FILE = label_type+"_checkpoint.pth"
    checkpoint = torch.load(FILE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    batchNum = checkpoint['batch_number']
    with open(label_type+"_checkpoint.pth", "w") as c:
        pass

with open(label_type+"_log.out", 'a') as lg:
    lg.write("\nStarting from:\n")
    lg.write("epoch = "+str(epoch)+"\n")
    lg.write("batchNum = "+str(batchNum)+"\n")
    lg.write("batchSize = "+str(batchSize)+"\n\n")

train_loader = DataLoader(dataset=dataset,
                      batch_size=batchSize,
                      shuffle=False,
                      num_workers=0)      

while(epoch < num_epochs):    
    ##########################################################
    for i, (inputs, labels, seqLens) in enumerate(train_loader):
        if i == batchNum:
            seqLens = seqLens.view(seqLens.size(0))
            seqLens = [int(x) for x in seqLens]
            # Forward pass and loss
            y_pred = model(inputs, seqLens)
            #y_pred = model(inputs)
            #y_pred = y_pred.view(y_pred.size(0))
            labels = labels.view(labels.size(0))
            labels = torch.tensor([[1-labels.item(), labels.item()]], dtype=torch.float32).to(device)
            #labels = labels.long()
            loss = criterion(y_pred, labels)
            if batchNum == printingBatch:
                with open(label_type+"_log.out", 'a') as lg:
                    lg.write("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item())+"\n")
                    lg.write("\n")
                    lg.write("y_pred:\n")
                    lg.write(str(y_pred)+"\n")
                    lg.write("\n")
                    lg.write("labels:\n")
                    lg.write(str(labels)+"\n")
                    lg.write("\n\n")
            # Backward pass and update
            loss.backward()
            optimizer.step()  
            # zero grad before new step
            optimizer.zero_grad()
            batchNum += 1
    ##########################################################
    batchNum = 0
    epoch += 1
signal_handler(0, 0)
