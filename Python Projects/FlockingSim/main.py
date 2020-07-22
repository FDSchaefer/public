"""
Rights: Franz D Schaefer
https://github.com/FDSchaefer/public
Please Give Credit if used

Boid Flocking Sim, based on Vpython & WxPython. 
Includes UI for pausing, number of Boids and movement factors. 
"""

#Import Librarys
import wx
from vpython import *
import numpy as np
from random import randrange


class Boids:
    def __init__(self, numboids = 100, sidesize = 50.0):
        
        #class constants 
        self.SIDE               = sidesize
        self.MIN                = self.SIDE * -1.0
        self.MAX                = self.SIDE
        self.NUMBOIDS           = numboids

        #UI Controls
        self.running            = True
        self.FOC                = True
        self.AVD                = True
        self.MVE                = True

        #Boid Features 
        self.RADIUS             = 1
        self.length             = 3
        self.BoidSpeed          = 0.5

        #Factors For Calculation
        self.NEARBY             = self.RADIUS * 10
        self.viewAngle          = 0.6
        self.FlockFACTOR        = 0.04
        self.velMatchFactor     = 0.2
        self.avoidanceFactor    = 0.5
        
        #Define Main
        self.boids()

    def boids(self):
        self.createUI()                 #Create The UI
        self.initializePositions()      #Create The Animation
        while (True):                   #Loop Forever 
            rate(100)                   #Rate Of Updates
            if self.running:            #If Active run The simulation Tick               
                self.moveAllBoidsToNewPositions()
    def createUI(self):
        
        ## INIT GUI
        scene.width = 500
        scene.height = 500
        scene.range = self.MAX*2        #Set the Zoom of the Scene
        scene.title = "Boid Flocking Behaviour\n"
        
        ## Pause And Number UI
        def Run(b): 
            self.running = not self.running
            if self.running: b.text = "Pause"
            else: b.text = "Run"
        button(text="Pause", pos=scene.title_anchor, bind=Run)

        def RunNUM(m):
            self.running = False
            import time
            time.sleep(0.5) #Allow current Tick to finish

            for b in range(self.NUMBOIDS):
                self.boidflock[b].visible = False   #Remove the Boids from simulation
            self.NUMBOIDS = int(m.selected)
            self.initializePositions()              #Run Creation function again with new Number

            self.running = True
        wtext(text='Number Of Boids: ')
        menu(choices=['50', '100', '150', '200', '250', '300'], index=1, bind=RunNUM)
        wtext(text='\n')

        ## Create Factor UI
        def FOC():
            self.FOC = not self.FOC
        def AVD():
            self.AVD = not self.AVD
        def MVE():
            self.MVE = not self.MVE
        def setFOC(s):
            wt1.text = '{:1.2f}  '.format(s.value)
            self.FlockFACTOR = 0.04*s.value**2
        def setAVD(s):
            wt2.text = '{:1.2f}  '.format(s.value)
            self.avoidanceFactor = 0.2*s.value**2
        def setMVE(s):
            wt3.text = '{:1.2f}  '.format(s.value)
            self.velMatchFactor = 0.5*s.value**2
        
        s1 = slider(min=0.1, max=3, value=1, length=150, bind=setFOC, right=15,style=wx.SL_HORIZONTAL)
        wtext(text='x')
        wt1 = wtext(text='{:1.2f}  '.format(s1.value))
        r1 = radio(bind=FOC, checked=True, text='Flock To Centre\n\n')
        
        s2 = slider(min=0.1, max=3, value=1, length=150, bind=setAVD, right=15,style=wx.SL_HORIZONTAL)
        wtext(text='x')
        wt2 = wtext(text='{:1.2f}  '.format(s2.value))
        r2 = radio(bind=AVD, checked=True, text='Avoidance\n\n')
        
        s3 = slider(min=0.1, max=3, value=1, length=150, bind=setMVE, right=15,style=wx.SL_HORIZONTAL)
        wtext(text='x')
        wt3 = wtext(text='{:1.2f}  '.format(s3.value))
        r3 = radio(bind=MVE, checked=True, text='Match Velocitys\n\n')

        #Create Default Button
        def Default():
            s1.value= 1
            s1.bind(s1)
            s2.value= 1
            s2.bind(s2)
            s3.value= 1
            s3.bind(s3)
        button(text="Reset Defaults", bind=Default)
        
    def initializePositions(self):

        #wire frame of space (Not Worth Looping for ease of reading)
        curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MAX, self.MIN, self.MIN)], color=color.white)
        curve(pos=[(self.MIN, self.MAX, self.MIN), (self.MAX, self.MAX, self.MIN)], color=color.white)
        curve(pos=[(self.MIN, self.MIN, self.MAX), (self.MAX, self.MIN, self.MAX)], color=color.white)
        curve(pos=[(self.MIN, self.MAX, self.MAX), (self.MAX, self.MAX, self.MAX)], color=color.white)
        curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MIN, self.MIN, self.MAX)], color=color.white)
        curve(pos=[(self.MIN, self.MAX, self.MIN), (self.MIN, self.MAX, self.MAX)], color=color.white)
        curve(pos=[(self.MAX, self.MIN, self.MIN), (self.MAX, self.MIN, self.MAX)], color=color.white)
        curve(pos=[(self.MAX, self.MAX, self.MIN), (self.MAX, self.MAX, self.MAX)], color=color.white)
        curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MIN, self.MAX, self.MIN)], color=color.white)
        curve(pos=[(self.MAX, self.MIN, self.MIN), (self.MAX, self.MAX, self.MIN)], color=color.white)
        curve(pos=[(self.MIN, self.MIN, self.MAX), (self.MIN, self.MAX, self.MAX)], color=color.white)
        curve(pos=[(self.MAX, self.MIN, self.MAX), (self.MAX, self.MAX, self.MAX)], color=color.white)
        
        self.boidflock = []             #empty list of boids
        self.boidvelo = []              #empty list of boids Velocity

        #Place Boids Randomly and add velocity
        for b in range(self.NUMBOIDS):       
            #Create coordinates within range
            x = randrange(self.MIN, self.MAX) 
            y = randrange(self.MIN, self.MAX) 
            z = randrange(self.MIN, self.MAX) 

            #Create velocity at max 1 magnitude
            self.boidvelo.append(vector.random())

            self.boidflock.append(cone(pos = vector(x,y,z), axis=self.boidvelo[b]*self.length, radius=self.RADIUS,  color=color.yellow))
            
    def moveAllBoidsToNewPositions(self):

        for b in range(self.NUMBOIDS):
            #Check the Boundary Condition
            self.boundRule(b)                          

            #Aquire the boids which are within the 'visual' range.
            Vlist = self.actionRadius(b)

            #Aquire Vector Factors          
            v1 = vector(0.0,0.0,0.0)        #initialize vector for rule 1
            v2 = vector(0.0,0.0,0.0)        #initialize vector for rule 2
            v3 = vector(0.0,0.0,0.0)        #initialize vector for rule 3
            
            v1 = self.rule1(b,Vlist)*self.FOC       #get the vector for rule 1
            v2 = self.rule2(b,Vlist)*self.AVD       #get the vector for rule 2
            v3 = self.rule3(b,Vlist)*self.MVE       #get the vector for rule 3

            #Apply Changes
            boidvelocity            = self.boidvelo[b]                                          #Get the previous Velocity
            boidvelocity            = hat(boidvelocity + v1 + v2 + v3)                          #accumulate the rules vector results
            self.boidflock[b].pos   = self.boidflock[b].pos + (boidvelocity*self.BoidSpeed)     #move the boid
            self.boidvelo[b]        = boidvelocity                                              #Update Vel list
            self.boidflock[b].axis  = self.boidvelo[b]*self.length                              #Update Boid Orientation

    def actionRadius(self,bSel):
        ViewList = []
        for b in range(self.NUMBOIDS):
            if b != bSel:
                DifVec     = self.boidflock[bSel].pos - self.boidflock[b].pos
                #Check if Close enough and within the angle of view
                if mag(DifVec) <= self.NEARBY:
                    #Split them to avoid extra calculations if possible
                    AngleOfBoid     = diff_angle(self.boidvelo[bSel],DifVec)
                    if AngleOfBoid <= 3.14*self.viewAngle:
                        ViewList.append(b)
        return ViewList

    def boundRule(self,b):
        #manage boids hitting the boundaries
        #Not Very efficent but handles it ok.
        if self.boidflock[b].pos.x < self.MIN:
            self.boidflock[b].pos.x = self.MAX
                
        if self.boidflock[b].pos.x > self.MAX:
            self.boidflock[b].pos.x = self.MIN
                
        if self.boidflock[b].pos.y < self.MIN:
            self.boidflock[b].pos.y = self.MAX
                
        if self.boidflock[b].pos.y > self.MAX:
            self.boidflock[b].pos.y = self.MIN
                
        if self.boidflock[b].pos.z < self.MIN:
            self.boidflock[b].pos.z = self.MAX
                
        if self.boidflock[b].pos.z > self.MAX:
            self.boidflock[b].pos.z = self.MIN
        return(self)
                
    def rule1(self, aboid, Vlist):    #Rule 1:  boids fly to perceived flock center
        v = vector(0.0,0.0,0.0)
        Vlist.append(aboid)
        for b in Vlist:
            v = v + self.boidflock[b].pos                               #calculate the total vector
        v = v/(len(Vlist)+1)                                            #average the vector

        vOut = (hat(v - self.boidflock[aboid].pos))*self.FlockFACTOR    #Clac Final Flock Vector
        return vOut 

    def rule2(self, aboid,Vlist):    #Rule 2: boids avoid other boids
        v = vector(0,0,0)
        c = 0
        for b in Vlist:
            if mag(self.boidflock[b].pos-self.boidflock[aboid].pos) < self.NEARBY/2:    #If too Close
                v = v + self.boidflock[b].pos
                c += 1
        if c != 0:
            v = v/c

        vout = hat(v - self.boidflock[aboid].pos)*-self.avoidanceFactor                 #Create Avoidance Vector
        return vout
        
    def rule3(self, aboid, Vlist):    #Rule 3: boids try to match speed of flock
        v = vector(0.0,0.0,0.0)
        Vlist.append(aboid)
        for b in Vlist:
            v = v + self.boidvelo[b]
        v = v/(len(Vlist)+1)
        vOut = hat(v) * self.velMatchFactor     #Create Flock Vector

        return vOut


if __name__ == "__main__":
    b = Boids()     #Init the Boids class




