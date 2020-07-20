
#from vpython import curve
#from vpython import vector
#from vpython import color
#from vpython import curve
from vpython import *
import numpy as np
from random import randrange

class Boids:
    def __init__(self, numboids = 150, sidesize = 50.0):     #class constructor with default parameters filled
        #class constants 
        #display(title = "Boids v1.0")   #put a title in the display window

        self.SIDE = sidesize            #unit for a side of the flight space

        #the next six lines define the boundaries of the torus
        """
        torus:  donut shaped space, i.e. infinite
        effect: boids flying out of bounds appear at the opposite side
        note:   cartesian matrices don't handle toruses very well, but I couldn't
                figure out a better way to keep the flock in view.
        """
        self.MIN = self.SIDE * -1.0    #left
        self.MAX = self.SIDE           #right
        
        self.RADIUS = 1                 #radius of a boid.  I wimped and used spheres.
        self.NEARBY = self.RADIUS * 10  #the view of space around each boid
        self.viewAngle = 0.6
        self.FlockFACTOR = 0.01
        self.velMatchFactor = 0.2
        self.avoidanceFactor = 0.5
        #self.NEGFACTOR = self.FACTOR * -1.0 #same thing, only negative
        
        self.NUMBOIDS = numboids        #the number of boids in the flock
        self.View     = 5               #View Range of Boids
        self.boidflock = []             #empty list of boids
        self.boidvelo = []              #empty list of boids Velocity
        self.BoidSpeed = 0.5

        self.boids()                    #okay, now that all the constants have initialized, let's fly!

    def boids(self):
        self.initializePositions()      #create a space with boids
        while (1==1):                   #loop forever
            rate(100)                    #controls the animation speed, bigger = faster
            self.moveAllBoidsToNewPositions()   #um ... what it says
            
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

        #Place Boids Randomly and add velocity
        for b in range(self.NUMBOIDS):       
            #Create coordinates within range
            x = randrange(self.MIN, self.MAX) 
            y = randrange(self.MIN, self.MAX) 
            z = randrange(self.MIN, self.MAX) 

            self.boidflock.append(sphere(pos = vector(x,y,z), radius=self.RADIUS, color=color.yellow))

            #Create velocity at max 1 magnitude
            self.boidvelo.append(vector.random())

    def moveAllBoidsToNewPositions(self):

        for b in range(self.NUMBOIDS):

            self.boundRule(b)                           #Check the Boundary Condition

            #Aquire the boids which are within the 'visual' range. 
            Vlist = self.actionRadius(b)
            #print(Vlist)

            v1 = vector(0.0,0.0,0.0)        #initialize vector for rule 1
            v2 = vector(0.0,0.0,0.0)        #initialize vector for rule 2
            v3 = vector(0.0,0.0,0.0)        #initialize vector for rule 3

            v1 = self.rule1(b,Vlist)              #get the vector for rule 1
            v2 = self.rule2(b,Vlist)              #get the vector for rule 2
            v3 = self.rule3(b,Vlist)              #get the vector for rule 3

            boidvelocity = self.boidvelo[b]
            boidvelocity = hat(boidvelocity + v1 + v2 + v3)  #accumulate the rules vector results
            self.boidflock[b].pos = self.boidflock[b].pos + (boidvelocity*self.BoidSpeed) #move the boid
            self.boidvelo[b]    = boidvelocity #Update Vel list


            

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




