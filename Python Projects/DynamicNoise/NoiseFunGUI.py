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

import time

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
        self.randSet    = True
        self.inverted   = False

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
            self.root.update_idletasks()
            self.root.update()
            if self.running:            #If Active run The simulation Tick               
                self.noiseMovement()


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
            if self.randSet == True:
                c  = (np.random.rand(num, 2)*Dm).astype(int) 

            ## Option 2, Radialy Distributed
            if self.randSet == False:
                c   = np.random.randn(num, 2)
                c   /=np.max(c)
                c   = (Dm/2+(c*Dm/2)).astype(int)



            for i in range(len(c)):
                C[c[i,0],c[i,1]] = 0
    
            texMap = fil.distance_transform_edt(C,sampling = [Dim,Dim])
            M = np.max(texMap)
            outMap = texMap/M       #Basic Option

            ## Option 2 Inverted Map
            if self.inverted == True:
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
        ## INIT
        self.root = tkinter.Tk()
        self.root.wm_title("Dynamic Noise Application")

        #CREATE FIGURE
        plt.ion()
        self.fig = Figure(figsize=(self.FigSize, self.FigSize), dpi=100)
        self.fig.add_subplot(111).imshow(np.zeros((self.DIM,self.DIM)), vmin=0, vmax=1)

        ## INIT CANVAS
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        ## PAUSE ANIMATION
        def _pause():
            self.running = not self.running

        button = tkinter.Button(master=self.root, text="Pause", command=_pause)
        button.pack(padx=5, pady=10, side=tkinter.LEFT)

        ## NEW SEED
        def _seed():
            self.initialize()

        button = tkinter.Button(master=self.root, text="New Seed", command=_seed)
        button.pack(padx=5, pady=10, side=tkinter.LEFT)

        ## RADIAL OR EVEN DISTRIBUTION
        def _noise():
            self.running = False
            time.sleep(.5)
            self.randSet = not self.randSet
            self.initialize()               #Create The Animation
            self.running = True

        button2 = tkinter.Button(master=self.root, text="Radial/Even", command=_noise)
        button2.pack(padx=5, pady=10, side=tkinter.LEFT)

        ## INVERT NOISE PATTERN
        def _invert():
            self.running = False
            time.sleep(.5)
            self.inverted = not self.inverted
            self.initialize()               #Create The Animation
            self.running = True

        button3 = tkinter.Button(master=self.root, text="Invert Noise", command=_invert)
        button3.pack(padx=5, pady=10, side=tkinter.RIGHT)

        ## COLOUR OPTIONS
        def _color(*args):
            self.color = varCol.get()
            self.fig.axes[0].images[0].set_cmap(self.color)

        varCol = tkinter.StringVar(self.root)
        varCol.set("RdPu") # initial value
        colOption = tkinter.OptionMenu(self.root, varCol, "viridis", "RdPu", "hot", "Greys",command=_color)
        colOption.pack(padx=5, pady=10, side=tkinter.RIGHT)
        varCol.trace("w", _color)

        ## INPUT LIST FOR BASE DENSITY
        def BaseCommand():
            self.running = False
            time.sleep(.5)
            self.DensityList = np.array(entryBD.get().split(" "),dtype=float)
            self.LayersList = np.array(entryBL.get().split(" "),dtype=float)
            self.initialize()
            self.running = True

        entryBL = tkinter.Entry(self.root)
        entryBL.insert(tkinter.END, self.LayersList)

        entryBD = tkinter.Entry(self.root)
        entryBD.insert(tkinter.END, self.DensityList)
        buttonBD = tkinter.Button(self.root, text="Input Base", command=BaseCommand)

        entryBL.pack(padx=5, pady=5, side=tkinter.BOTTOM)
        entryBD.pack(padx=5, pady=5, side=tkinter.BOTTOM)
        buttonBD.pack(side=tkinter.BOTTOM)

        ## INPUT LIST FOR DYNAMIC DENSITY
        def DynCommand():
            self.running = False
            time.sleep(.5)
            self.Dlist = np.array(entryDD.get().split(" "),dtype=float)
            self.Llist = np.array(entryDL.get().split(" "),dtype=float)
            self.initialize()
            self.running = True

        entryDD = tkinter.Entry(self.root)
        entryDD.insert(tkinter.END, self.Dlist)

        entryDL = tkinter.Entry(self.root)
        entryDL.insert(tkinter.END, self.Llist)
        buttonDL = tkinter.Button(self.root, text="Input Dynamic", command=DynCommand)

        entryDL.pack(padx=5, pady=5, side=tkinter.BOTTOM)
        entryDD.pack(padx=5, pady=5, side=tkinter.BOTTOM)
        buttonDL.pack(side=tkinter.BOTTOM)
        
    def initialize(self):
        #Base Noise
        self.Base,_ = self.textureMake(self.DensityList,self.LayersList,self.DIM)
        self.ChangeOld = np.zeros((self.DIM,self.DIM))

        self.fig.axes[0].images[0].set_data(self.Base)
        self.fig.axes[0].relim()
        self.fig.axes[0].autoscale_view()

        self.root.update()
        self.root.update_idletasks()
        plt.pause(1e-3)


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

            if self.running:
                self.fig.canvas.draw_idle()

            self.root.update()
            self.root.update_idletasks()
            plt.pause(1e-3)

        #Inherit
        self.ChangeOld = ChangeNew

if __name__ == "__main__":
    b = DynNoise()     #Init the class