# FDS Public Project Repository
Welcome, I hope to always add new projects, all work is done myself unless specificaly credited otherwise.
All projects are organised by language, then in individual project folders (If Uploaded).
Sneak Peeks will be shown here, with relevent information/visuals.

(Please note this repo is a recent creation & I still need to clear publication/NDA agreements for larger projects)

## Deep Learning Projects  
1. **MetaData Based Treatment Planning Thesis (Fully Connected Neural Network & Autoencoders)** (100%)
<details>
  <summary>Details</summary>
  This Projects purpose was to create a fully connected neural network to predict treatment sucess based on patient metadata and the respecitive treatment plan that was implemented. Additionaly this model was used to optimise treatments by implementing the network into the existing planning software *MatRad*. The project involved full data aquisition, extraction, standardisation, autoencoding and final model creation and subsequent integration.  
  
  **NOTE: Currently being refined for publication, therefore will not be uploaded until then**
  
  [Link 2 Abstract](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/Treatment%20Planning/Abstract.pdf)
</details>
 
2. [**Brain MRI: 3D Convolutional AutoEncoder & Patient Age Classifier**](https://github.com/FDSchaefer/public/tree/master/Deep%20Learning/BRAIN%20MRI) (75%)
<details>
  <summary>Details</summary>
  This project involves the collection of healthy brain MRI images with various patient ages. The autoencoder compresses the 3D MRI data to a more manageable form for the Age classifier network. (This is due ot the memory limitations of my GPU). The Convolutional classifier then reads the encoded data, to predict the age of the patient, of whom the MRI was taken. The network was written in Pytorch, with CUDA compatability.
  
  The data aquired from: https://www.insight-journal.org/midas/community/view/21
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/BRAIN%20MRI/README/gif2.gif)  
  
</details>   
 
3. **Receipt Scanner and Expense Organiser(OCR)** *IN PROGRESS*
<details>
  <summary>Details</summary>
  The concept behind this project is to implement a OCR for the purposes of reading pictures of reciepts, and extracting the name and price of each item. This would then be tabulated and the items placed into catagories. Giving the user an overview of where and how money is being spent, which items are candidates for bulk buying, etc. Idealy this project would be integrated into an app in a later development stage. 
  
</details>  


## Python Mini Projects
1. [**Boid Flocking Sim**](https://github.com/FDSchaefer/public/tree/master/Python%20Projects/FlockingSim)  (100%)
<details>
  <summary>Details</summary>
  This project involved the implementation of a simple Boid Flocking simulation, using the 3 laws. Additional GUI additions were added to allow the user to play around with the simulation, including sliders, buttons and menus for all relevent options. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/BoidGif.gif)  
  
</details>

2. [**Dynamic Textures, Via Random Noise**](https://github.com/FDSchaefer/public/tree/master/Python%20Projects/DynamicNoise)  (100%)
<details>
  <summary>Details</summary>
  This project worked on creating dynamic textures via random noise, by layering differnet noise dencities in differnt ways, to allow for a fast yet always unique experience. Works well as a screensaver/background or animated poster. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/Noise.gif)  
  
</details>
  
## MatLab Mini Projects
1. [**Collision Simulation**](https://github.com/FDSchaefer/public/tree/master/MatLab%20Projects/TriangleCollision)  (100%)
<details>
  <summary>Details</summary>
  This project involved the implementation of 2D collision mechanics for randomly placed moving ships. Using the main script one would be able to add or remove the number of ships, and take manual control over the frame updates. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/ColliderGif.gif?raw=true)
  
</details>


## Unity (C#)
Too Be added Soon




## C++ 
Too Be added Soon
