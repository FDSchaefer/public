
from vpython import *
import wx
import numpy as np
from random import randrange


class Boids:
    def __init__(self, numboids = 100, sidesize = 50.0):     #class constructor with default parameters filled
        #class constants 
        #display(title = "Boids v1.0")   #put a title in the display window

        self.SIDE = sidesize            #unit for a side of the flight space

        #the next six lines define the boundaries of the torus
        """
        torus:  donut shaped space, i.e. infinite
        effect: boids flying out of bounds appear at the opposite side
        """

        #Controls
        self.running    = True
        self.FOC        = True
        self.AVD        = True
        self.MVE        = True

        self.MIN = self.SIDE * -1.0    #left
        self.MAX = self.SIDE           #right
        
        self.RADIUS = 1                 #radius of a boid.  I wimped and used spheres.
        self.length = 3


        self.NEARBY = self.RADIUS * 10  #the view of space around each boid
        self.viewAngle = 0.6
        self.FlockFACTOR = 0.04
        self.velMatchFactor = 0.2
        self.avoidanceFactor = 0.5
        

        self.NUMBOIDS = numboids        #the number of boids in the flock
        self.View     = 5               #View Range of Boids
        self.boidflock = []             #empty list of boids
        self.boidvelo = []              #empty list of boids Velocity
        self.BoidSpeed = 0.5


        self.boids()                    #okay, now that all the constants have initialized, let's fly!

    def boids(self):
        self.createUI()
        self.initializePositions()      #create a space with boids
        while (True):                   #loop forever
            rate(100)                    #controls the animation speed, bigger = faster
            if self.running:
                self.moveAllBoidsToNewPositions()   #um ... what it says
         
    def createUI(self):
        
        scene.width = 500
        scene.height = 500
        scene.range = self.MAX*2
        scene.title = "Boid Flocking Behaviour\n"
        

        def Run(b): 
            self.running = not self.running
            if self.running: b.text = "Pause"
            else: b.text = "Run"
        button(text="Pause", pos=scene.title_anchor, bind=Run)

        def RunNUM(m):
            self.running = False
            import time
            time.sleep(0.5)

            for b in range(self.NUMBOIDS):
                self.boidflock[b].visible = False
            self.NUMBOIDS = int(m.selected)
            self.initializePositions()

            self.running = True

        wtext(text='Number Of Boids: ')
        menu(choices=['50', '100', '150', '200', '250', '300'], index=1, bind=RunNUM)
        wtext(text='\n')

        def FOC():
            self.FOC = not self.FOC
            
        def AVD():
            self.AVD = not self.AVD

        def MVE():
            self.MVE = not self.MVE

        def setFOC(s):
            wt1.text = '{:1.2f}  '.format(s.value)
            self.FlockFACTOR = 0.04*s.value

        def setAVD(s):
            wt2.text = '{:1.2f}  '.format(s.value)
            self.avoidanceFactor = 0.2*s.value

        def setMVE(s):
            wt3.text = '{:1.2f}  '.format(s.value)
            self.velMatchFactor = 0.5*s.value
        
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
        backBottom      = curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MAX, self.MIN, self.MIN)], color=color.white)
        backTop         = curve(pos=[(self.MIN, self.MAX, self.MIN), (self.MAX, self.MAX, self.MIN)], color=color.white)
        frontBottom     = curve(pos=[(self.MIN, self.MIN, self.MAX), (self.MAX, self.MIN, self.MAX)], color=color.white)
        frontTop        = curve(pos=[(self.MIN, self.MAX, self.MAX), (self.MAX, self.MAX, self.MAX)], color=color.white)
        leftBottom      = curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MIN, self.MIN, self.MAX)], color=color.white)
        leftTop         = curve(pos=[(self.MIN, self.MAX, self.MIN), (self.MIN, self.MAX, self.MAX)], color=color.white)
        rightBottom     = curve(pos=[(self.MAX, self.MIN, self.MIN), (self.MAX, self.MIN, self.MAX)], color=color.white)
        rightTop        = curve(pos=[(self.MAX, self.MAX, self.MIN), (self.MAX, self.MAX, self.MAX)], color=color.white)
        backLeft        = curve(pos=[(self.MIN, self.MIN, self.MIN), (self.MIN, self.MAX, self.MIN)], color=color.white)
        backRight       = curve(pos=[(self.MAX, self.MIN, self.MIN), (self.MAX, self.MAX, self.MIN)], color=color.white)
        frontLeft       = curve(pos=[(self.MIN, self.MIN, self.MAX), (self.MIN, self.MAX, self.MAX)], color=color.white)
        frontRight      = curve(pos=[(self.MAX, self.MIN, self.MAX), (self.MAX, self.MAX, self.MAX)], color=color.white)
        
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

            self.boundRule(b)                           #Check the Boundary Condition

            #Aquire the boids which are within the 'visual' range. 
            Vlist = self.actionRadius(b)
            #print(Vlist)

            v1 = vector(0.0,0.0,0.0)        #initialize vector for rule 1
            v2 = vector(0.0,0.0,0.0)        #initialize vector for rule 2
            v3 = vector(0.0,0.0,0.0)        #initialize vector for rule 3

            v1 = self.rule1(b,Vlist)*self.FOC              #get the vector for rule 1
            v2 = self.rule2(b,Vlist)*self.AVD                #get the vector for rule 2
            v3 = self.rule3(b,Vlist)*self.MVE              #get the vector for rule 3

            boidvelocity = self.boidvelo[b]
            boidvelocity = hat(boidvelocity + v1 + v2 + v3)  #accumulate the rules vector results
            self.boidflock[b].pos = self.boidflock[b].pos + (boidvelocity*self.BoidSpeed) #move the boid
            self.boidvelo[b]    = boidvelocity #Update Vel list
            self.boidflock[b].axis = self.boidvelo[b]*self.length

    def actionRadius(self,bSel):
        ViewList = []
        for b in range(self.NUMBOIDS):              #for all the boids
            if b != bSel:
                differenceVec = self.boidflock[bSel].pos - self.boidflock[b].pos
                if mag(differenceVec) <= self.NEARBY: #Check if close enough
                    #Now check if angle is in direction of flight/view
                    if diff_angle(self.boidvelo[bSel],differenceVec) <= 3.142*self.viewAngle:
                        ViewList.append(b)
        return ViewList

    def boundRule(self,b):
        #manage boids hitting the boundaries
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
        pfc = vector(0.0,0.0,0.0)                   #pfc: perceived flock center
        Vlist.append(aboid)
        for b in Vlist:              #for all the boids
            pfc = pfc + self.boidflock[b].pos  #calculate the total pfc
        pfc = pfc/(len(Vlist)+1)           #average the pfc


        pfcOut = (hat(pfc - self.boidflock[aboid].pos))*self.FlockFACTOR


        return pfcOut 

    def rule2(self, aboid,Vlist):    #Rule 2: boids avoid other boids
        MeanClose = vector(0,0,0)
        c = 0

        #Find closet boid
        for b in Vlist:              #for all the boids
            if mag(self.boidflock[b].pos-self.boidflock[aboid].pos) < self.NEARBY/2:
                MeanClose = MeanClose + self.boidflock[b].pos
                c += 1
        if c != 0:
            MeanClose = MeanClose/c

        vout = hat(MeanClose - self.boidflock[aboid].pos)*-self.avoidanceFactor

        return vout
        
    def rule3(self, aboid, Vlist):    #Rule 3: boids try to match speed of flock
        pfv = vector(0.0,0.0,0.0)                   #pfc: perceived flock velocity
        Vlist.append(aboid)
        for b in Vlist:              #for all the boids
            pfv = pfv + self.boidvelo[b]  #calculate the total pfc
        pfv = pfv/(len(Vlist)+1)           #average the pfc
        pfv = hat(pfv) * self.velMatchFactor

        return pfv


if __name__ == "__main__":
    b = Boids()     #instantiate the Boids class, the class constructor takes care of the rest.




