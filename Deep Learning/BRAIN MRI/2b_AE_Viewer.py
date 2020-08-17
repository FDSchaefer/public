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
device = "cpu"


# Network
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
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


### LOAD MODEL Base ###
NAME = input('Load Model: ')
torch.nn.Module.dump_patches = True
model = torch.load(SaveDir + NAME +'\\ModelSave.npy')
model = model.to(device)
model.eval()

#Load MODEL State
State = str(input('Load State: (with BEST as Default) '))
if State == '': #Load Default
    model.load_state_dict(torch.load(SaveDir + NAME +'//Best.pth'))
elif State+'.pth' in os.listdir(SaveDir + NAME):   #Check If State exists
    model.load_state_dict(torch.load(SaveDir + NAME +'//'+ State + '.pth'))
else:
    model.load_state_dict(torch.load(SaveDir + NAME +'//Best.pth')) #If State does not load Default 
    print('State Does Not Exist, Loading Default')


### Load Data ###
Data = np.load(SaveDir + NAME +'\\TrainingHistory.npy', mmap_mode=None, allow_pickle=True)
LTmean1, LTstd1, LTmean2, LTstd2, LVmean1, LVstd1, LVmean2, LVstd2 = Data[0],Data[1],Data[2],Data[3],Data[4],Data[5],Data[6],Data[7]

epochList = range(len(LTmean1))

#Loss over Time
plt.figure()
###
plt.subplot(1,2,1)
plt.errorbar(epochList,LTmean1,LTstd1)
plt.text(epochList[-1],LTmean1[-1],str("{:.2e}".format(LTmean1[-1])), ha = 'right')

plt.errorbar(epochList,LVmean1,LVstd1,alpha = 0.5)
plt.text(epochList[-1],LVmean1[-1],str("{:.2e}".format(LVmean1[-1])), ha = 'right')

plt.title('MSE Loss Vs Epochs')
plt.legend(('Training','Validation'))
plt.grid(which='both')
plt.xlabel('Epoch')
plt.ylabel('log(Loss) MSELoss')
plt.yscale('log')
###
plt.subplot(1,2,2)
plt.errorbar(epochList,LTmean2,LTstd2)
plt.text(epochList[-1],LTmean2[-1],str("{:.2e}".format(LTmean2[-1])), ha = 'right')

plt.errorbar(epochList,LVmean2,LVstd2,alpha = 0.5)
plt.text(epochList[-1],LVmean2[-1],str("{:.2e}".format(LVmean2[-1])), ha = 'right')

plt.title('L1 Loss Vs Epochs')
plt.legend(('Training','Validation'))
plt.grid(which='both')
plt.xlabel('Epoch')
plt.ylabel('log(Loss) L1Loss')
plt.yscale('log')

plt.show()

### Load Image Data ###
Data = np.load(SaveDir + NAME +'\\PostModelImages.npy', mmap_mode=None, allow_pickle=True)
Orig, Encode, Decode = Data[0],Data[1],Data[2]


# IDENTIFY Problem Samples

def MSECal(X,Y):
    return(((X - Y)**2).mean(axis=None))

MSElist = []
for i in range(len(Orig)):
    MSElist.append(MSECal(Orig[i],Decode[i]))

MSElist = np.array(MSElist)

plt.figure()
plt.bar(range(len(Orig)),MSElist)
plt.plot(range(len(Orig)),np.ones(len(Orig))*np.mean(MSElist),color = 'red')

plt.title('MSE per Image')
plt.legend(('Mean MSE','Image MSE'))
plt.grid(which='both')
plt.xlabel('Image')
plt.ylabel('MSE Value')

plt.show()


from ScrollView import ScrollerMulti
check = 'Y'
while check == 'Y' or check == 'y':

    imgNumber = int(input('Which Scan to View?: '))
    ScrollerMulti([Orig[imgNumber],Decode[imgNumber]],2,['Original Image','Decoded Image','Test'])

    check = input('View Another?: ')




print('Debug')

