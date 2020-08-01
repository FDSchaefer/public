

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as fil


## Function Def
def textureMake(DensityList,LayersList,Dim):
    #Density is scalar 0 - 1, length of number of layers
    #Layers is scalar no limit, length of number of layers
    #Dim is image input 
    
    def textureLayerMake(Density,Dim):
        if Density > 1 or Density <0:
            print('Density Out of Bounds, Capping')
            if Density > 1:
                Density = 1
            else:
                Density = 0  

        num = int(Dim * (1-(Density-0.5))**3 )
        
        Dm = Dim-1 
        C   = np.ones((Dim,Dim))

        ## Option 1, Evenly Distributed
        c  = (np.random.rand(num, 2)*Dm).astype(int) 

        ## Option 2, Radialy Distributed
        #c   = np.random.randn(num, 2)
        #c   /=np.max(c)
        #c   = (Dm/2+(c*Dm/2)).astype(int)



        for i in range(len(c)):
            C[c[i,0],c[i,1]] = 0
    
        texMap = fil.distance_transform_edt(C,sampling = [Dim,Dim])
        M = np.max(texMap)
        outMap = texMap/M       #Basic Option

        ## Option 2 Inverted Map
        outMap = -outMap +1

        return(outMap)


    ##Layering
    Empty    = np.zeros((Dim,Dim))
    EmptyL   = [np.zeros((Dim,Dim))]*len(LayersList)
    for i in range(len(LayersList)):
        L =  textureLayerMake(DensityList[i],Dim) * LayersList[i]
        EmptyL[i] = L
        Empty += L
    
    M = np.max(Empty)
    Output = Empty/M
    
    return(Output,EmptyL)



Dim         = 1000
DensityList = [0,0.7,1,0.3,1]
LayersList  = [0.4,0.7,0.9,0.3,1]

Final,Mix = textureMake(DensityList,LayersList,Dim)

plt.figure()
plt.imshow(Final)

plt.figure()
for i in range(len(DensityList)):
    plt.subplot(1,len(DensityList),i+1)
    plt.imshow(Mix[i])

plt.show()

#Spooky
#Changing the lowest value of noise, animated

Dim         = 1000
DensityList = [0.7,1,0.3,1,0]
LayersList  = [0.7,0.4,0.3,1,0.4]

#Base Noise
Base,_ = textureMake(DensityList,LayersList,Dim)

Upd = plt.figure()
plt.title('Dynamic Noise')

#First Update
Dlist = [0.6,   0.3,  0]
Llist = [1,     0.6, 0.3]
DnoiseWeight = 0.7
ChangeOld,_ = textureMake(Dlist,Llist,Dim)
Disp = np.add(Base,ChangeOld*DnoiseWeight)
IMG = plt.imshow(Disp,cmap = 'RdPu')
#Hot is nice, like sun
# RdPu like onion cells under microscope
TransStep = 20

#Create Dynamic
while True:
    #New Noise
    ChangeNew,_ = textureMake(Dlist,Llist,Dim)

    for j in range(TransStep):
        #Dyn Change by changing weights
        TransWeight = (j+1)/TransStep
        NewW =  TransWeight         #Increasing Weight Per Trans update
        OldW =  (1-TransWeight)     #Decreasing towards 0

        Disp = np.add(Base,np.add(ChangeOld*OldW,ChangeNew*NewW))
        IMG.set_data(Disp)
        plt.draw(), plt.pause(1e-3)

    #Inherit
    ChangeOld = ChangeNew
