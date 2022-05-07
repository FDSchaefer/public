"""
Rights: Franz D Schaefer
https://github.com/FDSchaefer/public
Please Give Credit if used


"""

import os
import numpy as np
from medpy.io import load
import scipy.ndimage as img
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


## DIRECTORIES ###
root = os.getcwd()
processedDir = root+'/processed'
SaveDir = root+'/models/'


### CUDA ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: ' + str(device))

## PREP IMAGE DATA ##
imageList = np.load(processedDir + '/standardData.npy', mmap_mode=None, allow_pickle=True)

dataSet  = torch.from_numpy(imageList)
dataSet  = dataSet.float().to(device)
validation_split = .15
random_seed= np.random.randint(0,100)

from sklearn.model_selection import train_test_split
X_train, X_vali, Y_train, Y_vali = train_test_split(
    dataSet, dataSet, test_size=validation_split, random_state=random_seed)

# Network
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.conv1 = nn.Conv3d(1,20,(7,7,7),1,3)
        self.conv2 = nn.Conv3d(20,40,(5,5,5),1,2)
        self.conv3 = nn.Conv3d(40,60,(3,3,3),1,1)
        
        self.maxP = nn.MaxPool3d((3, 3, 3), stride=3,padding = 0, return_indices=True)

        self.maxUP = nn.MaxUnpool3d((3, 3, 3), stride=3,padding = 0)

        self.conv1N = nn.ConvTranspose3d(20,1,(7,7,7),1,3)
        self.conv2N = nn.ConvTranspose3d(40,20,(5,5,5),1,2)
        self.conv3N = nn.ConvTranspose3d(60,40,(3,3,3),1,1)

        #self.Sig = nn.Sigmoid()
        
    def encoder(self, x):

        x = F.relu(self.conv1(x))
        x,self.in1  = self.maxP(x)
        x = F.relu(self.conv2(x))
        x,self.in2 = self.maxP(x)
        x = F.relu(self.conv3(x))

        return x
    
    def decoder(self, x):

        x = F.relu(self.conv3N(x))
        x = self.maxUP(x,self.in2)
        x = F.relu(self.conv2N(x))
        x = self.maxUP(x,self.in1)
        x = F.relu(self.conv1N(x))
        #x = self.Sig(x)

        return x
        
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x

model = autoencoder().to(device)
model.share_memory()

### TRAINING OPTIONS ###

learning_rate = 1.5e-4
num_epochs = 100

batch_size = 3
kn = 5

criterion1  = nn.MSELoss(reduction='sum')
criterion2  = nn.L1Loss(reduction='sum')

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-6)

### SAVEMODEL OPTION ###
SAVE = 'N'
SAVE = input("Save Model? (Y/N) ") 
if SAVE == 'Y' or SAVE == 'y':
    NAME = 'DEFAULT'
    NAME = input('Name Model: ')
    os.makedirs(SaveDir+NAME)

    Increment = num_epochs* int(input('Save Increment (%): '))/100
    

### INIT HISTORY LISTS ###

LTmean1, LTstd1 = [],[]
LTmean2, LTstd2 = [],[]

LVmean1, LVstd1 = [],[]
LVmean2, LVstd2 = [],[]

BESTtrain, BESTvali = float("inf"),float("inf")

### START TRAINING ###
from sklearn.model_selection import ShuffleSplit
import time

for epoch in range(num_epochs):
    #Start Epoch Timer
    t = time.perf_counter()

    ### Kfold Splitting ###   
    random_seed= np.random.randint(0,num_epochs)
    folder = ShuffleSplit(n_splits = kn, test_size = validation_split, random_state = random_seed)
    DataFolds = folder.split(X_train)    
    
    for train_index, test_index in DataFolds:

        ### Training Fold ###
        dataloader = DataLoader(X_train, batch_size=batch_size, sampler= SubsetRandomSampler(train_index))
        for data in dataloader:
            # ===================forward=====================
            out = model(data)
            Tl1 = criterion1(out, data)  #Use criterion 1 as the training method
            Tl2 = criterion2(out, data)

            # ===================backward====================
            optimizer.zero_grad()
            Tl1.backward()
            optimizer.step()

        ### Testing Fold ###
        dataloader = DataLoader(X_train, batch_size=batch_size, sampler= SubsetRandomSampler(test_index))
        T1, T2 = [], []
        for data in dataloader:
            # ===================forward=====================
            out = model(data)
            Tl1 = criterion1(out, data)
            Tl2 = criterion2(out, data)

            # ===================log========================
            T1.append(Tl1.to('cpu').data)
            T2.append(Tl2.to('cpu').data)

    LTmean1 .append(np.mean(T1))
    LTstd1  .append(np.std(T1))

    LTmean2 .append(np.mean(T2))
    LTstd2  .append(np.std(T2))


    ### Validation Check ###   
    validation_loader = DataLoader(X_vali, batch_size=batch_size)
    V1, V2 = [], []
    for data in validation_loader:
        Vout    = model(data)
        Vl1     = criterion1(Vout, data)
        Vl2     = criterion2(Vout, data)

        V1.append(Vl1.to('cpu').data)
        V2.append(Vl2.to('cpu').data)

    LVmean1 .append(np.mean(V1))
    LVstd1  .append(np.std(V1))

    LVmean2 .append(np.mean(V2))
    LVstd2  .append(np.std(V2))
    
    ### PRINT LOGGING ###  
    print('epoch[{}/{}], Time:{:.1f}s | Tloss1: {:.2f}, Vloss1: {:.2f} | Tloss2: {:.2f}, Vloss2: {:.2f} |'
        .format(epoch + 1, num_epochs, time.perf_counter() - t, LTmean1[epoch], LVmean1[epoch], LTmean2[epoch], LVmean2[epoch]))

    ### Save Model Checkpoints ###
    if SAVE == 'Y' or SAVE == 'y':

        #IF CHECKPOINT
        if epoch % Increment == 0:
            torch.save(model.state_dict(), SaveDir + NAME +'//' + str(epoch)+ '.pth')

        #IF FINAL EPOCH
        if epoch == num_epochs-1:
            torch.save(model.state_dict(), SaveDir + NAME +'//Final.pth')

        # CHECK FOR BEST VERSION
        if LTmean1[epoch] <= BESTtrain and LVmean1[epoch] <= BESTvali:
            BESTtrain,BESTvali = LTmean1[epoch],LVmean1[epoch]

            print('New Best at: ' + str(round(BESTtrain)) + ' and ValBest: '+ str(round(BESTvali)))
            torch.save(model.state_dict(), SaveDir + NAME +'//Best.pth')

if SAVE == 'Y' or SAVE == 'y':
    #SAVE ALL TRAINING DATA/SETS for Future Display
    torch.save(model, SaveDir + NAME +'\ModelSave.npy')
    np.save(SaveDir + NAME +'/TrainingHistory', [LTmean1, LTstd1, LTmean2, LTstd2, LVmean1, LVstd1, LVmean2, LVstd2])



### SAVE MODEL OUTPUTS FOR LATER ANALYIS ###
model = model.load_state_dict(torch.load(SaveDir + NAME +'//Best.pth'))

Original, Encoded, Decoded = [], [], []
for i in range (0,len(dataSet)):
    input = dataSet[i:i+1]

    Original    .append(                    input   .cpu().detach().numpy()[0,0])
    Encoded     .append(    model.encoder(  input   ).cpu().detach().numpy()[0])
    Decoded     .append(    model(          input   ).cpu().detach().numpy()[0,0])

##CLEAR UP MEMORY FOR SAVING 
del model
del imageList
del dataSet

np.save(SaveDir + NAME +'/PostModelImages', [Original, Encoded, Decoded])

from ScrollView import Scroller

Scroller(Original[10])
Scroller(Encoded[10][0])
Scroller(Decoded[10])

#np.load('PATH',mmap_mode=None, allow_pickle = True)
print('debug')