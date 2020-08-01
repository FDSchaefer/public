"""
Rights: Franz D Schaefer
https://github.com/FDSchaefer/public
Please Give Credit if used

.
.
.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as fil

import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

## Function Def


class DynNoise:
    def __init__(self):

        #Universal 
        self.DIM        = 1000
        self.color      = 'RuPu'
        self.running    = True
        self.TransStep       = 20
        self.FigSize    = 5

        #Base
        self.DensityList = [0.7,1,0.3,1,0]
        self.LayersList  = [0.7,0.4,0.3,1,0.4]


        #Dynamic
        self.Dlist = [0.6,   0.3,  0]
        self.Llist = [1,     0.6, 0.3]
        self.DnoiseWeight = 0.7

        #Define Main
        self.DynNoise()

    def DynNoise(self):
        self.createUI()                 #Create The UI
        self.root.update()
        self.initialize()               #Create The Animation
        while (True):                   #Loop Forever 
            if self.running:            #If Active run The simulation Tick               
                self.noiseMovement()
                self.root.update_idletasks()
                self.root.update()

    def textureMake(self,DensityList,LayersList,Dim):
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

    def createUI(self):  
        self.root = tkinter.Tk()
        self.root.wm_title("Dynamic Noise Application")

        plt.ion()
        self.fig = Figure(figsize=(self.FigSize, self.FigSize), dpi=100)
        self.fig.add_subplot(111).imshow(np.zeros((self.DIM,self.DIM)), vmin=0, vmax=1)


        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        def _quit():
            self.root.quit()     # stops mainloop
            self.root.destroy()  # this is necessary on Windows to prevent

        button = tkinter.Button(master=self.root, text="Quit", command=_quit)
        button.pack(side=tkinter.BOTTOM)

        def _color(*args):
            self.color = varCol.get()
            self.fig.axes[0].images[0].set_cmap(self.color)

        varCol = tkinter.StringVar(self.root)
        varCol.set("RdPu") # initial value

        colOption = tkinter.OptionMenu(self.root, varCol, "viridis", "RdPu", "hot", "Greys",command=_color)
        colOption.pack()

        varCol.trace("w", _color)

    def initialize(self):
        

        #Base Noise
        self.Base,_ = self.textureMake(self.DensityList,self.LayersList,self.DIM)

        self.ChangeOld,_ = self.textureMake(self.Dlist,self.Llist,self.DIM)
        Disp = np.add(self.Base,self.ChangeOld*self.DnoiseWeight)
        
        #self.IMG = plt.imshow(Disp,cmap = 'RdPu')
        #Hot is nice, like sun
        # RdPu like onion cells under microscope
        

    def noiseMovement(self):
        #New Noise
        ChangeNew,_ = self.textureMake(self.Dlist,self.Llist,self.DIM)

        for j in range(self.TransStep):
            #Dyn Change by changing weights
            TransWeight = (j+1)/self.TransStep
            NewW =  TransWeight         #Increasing Weight Per Trans update
            OldW =  (1-TransWeight)     #Decreasing towards 0

            Disp = np.add(self.Base,np.add(self.ChangeOld*OldW,ChangeNew*NewW)*self.DnoiseWeight)
            M = np.max(Disp)
            Disp = Disp/M

            self.fig.axes[0].images[0].set_data(Disp)
            self.fig.axes[0].relim()
            self.fig.axes[0].autoscale_view()

            self.fig.canvas.draw_idle()
            self.root.update()
            self.root.update_idletasks()
            plt.pause(1e-3)

        #Inherit
        self.ChangeOld = ChangeNew

if __name__ == "__main__":
    b = DynNoise()     #Init the class