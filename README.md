# FDS Public Project Repository
Welcome, I hope to always add new projects, all work is done myself unless specifically credited otherwise.
All projects are organised by language, then in individual project folders (If uploaded).
Sneak Peeks will be shown here, with relevant information/visuals.

## Deep Learning Projects  
1. [**Brain MRI: 3D Convolutional AutoEncoder**](https://github.com/FDSchaefer/public/tree/master/Deep%20Learning/BRAIN%20MRI) (100%)
<details>
  <summary>Details</summary>
  This project involves the creation of an autoencoder to process 3D brain MRI data via 3 3D convolutions, for a total compression of circa 60%.
  
  It is not uncommon that one of the main limitations of machine learning models is the inability of the host system to handle the large datasets, or the network getting overwhelmed by the number of features. An autoencoder, allows us to use machine learning to compress the image down to a manageable size while maintaining the core feature information which would be needed in future modelling steps.
  
  The network was written in Pytorch, with CUDA compatibility using raw python scripts.
  
  The data acquired from: https://www.insight-journal.org/midas/community/view/21
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/BRAIN%20MRI/README/gif2.gif)  
  
</details>   
 
2. [**Satellite Image, location prediction via 2D classification**](https://github.com/FDSchaefer/public/tree/master/Deep%20Learning/ClassSat) *IN PROGRESS*
<details>
  <summary>Details</summary>
  With the USA being a large and diverse country in terms of landscapes and environments, it is not unreasonable to assume one could identify states from satellite photography. Inspired by games like Geo-Guesser, where the classification is done by humans, i wanted to experiment to see if this would be possible via a neural network. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/ClassSat/images/RandomSelection.jpg)  
  
  I acquired data from 4 US States, (California-CA, Maine-ME, New Mexico-NM and Florida-FL) via the [USGS Earth Explorer](https://earthexplorer.usgs.gov). By collecting a sample 50 images from each state we ensured a general overview with some variation in landscape and even cityscapes. As the files were encoded in the .jp2 format there was a significant effort to parse the information into more accessible forms, additionally to ensure enough data for training i decided that each HD satellite image (13200x12000x4) would be sampled 20 times (128x128x4) using a random non-repeating sampling algorithm. We additionally set aside 15% of the full size images to keep as final unseen testing data, which were then sampled and stored separately. 
  
  ![Training](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/ClassSat/images/TrainingData.jpg)
  
  The training and validation sets were split via stratified random sampling at 30% used for validation. We also implemented a mild dropout and some basic data augmentation within the network to avoid overfitting. After 200 epochs of training we found a very acceptable training and validation accuracy of: 98.7% and 98.3%, with the unseen testing set being predicted with 96% accuracy. Showing that our network was generally applicable for these states.
  
  ![Testing](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/ClassSat/images/Testing.jpg)
  
  Some next steps would be to use the pretrained model to introduce a 5th class, and observe if it would be able to distinguish and maintain its previous training. 
  
</details>  

3. **Masters Thesis - Metadata Based Treatment Planning via Fully Connected Neural Network** (100%)
<details>
  <summary>Details</summary>
  This Projects purpose was to create a fully connected neural network to predict treatment success based on patient metadata and the respective treatment plan that was implemented. Additionally this model was used to optimise treatments by implementing the network into the existing planning software *MatRad*. The project involved full data acquisition, extraction, standardisation, autoencoding and final model creation and subsequent integration.  
  
  **NOTE: Currently being refined for publication, therefore code will not be uploaded until then**
  
  [Link 2 Abstract](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/Treatment%20Planning/Abstract.pdf)
</details>

## Python Mini Projects
1. [**Boid Flocking Sim**](https://github.com/FDSchaefer/public/tree/master/Python%20Projects/FlockingSim)  (100%)
<details>
  <summary>Details</summary>
  This project involved the implementation of a simple Boid Flocking simulation, using the 3 laws. Additional GUI additions were added to allow the user to play around with the simulation, including sliders, buttons and menus for all relevant options. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/BoidGif.gif)  
  
</details>

2. [**Dynamic Textures, Via Random Noise**](https://github.com/FDSchaefer/public/tree/master/Python%20Projects/DynamicNoise)  (100%)
<details>
  <summary>Details</summary>
  This project worked on creating dynamic textures via random noise, by layering different noise densities in different ways, to allow for a fast yet always unique experience. Works well as a screensaver/background or animated poster. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/Noise.gif)  
  
</details>
  
## MatLab Mini Projects
1. [**Collision Simulation**](https://github.com/FDSchaefer/public/tree/master/MatLab%20Projects/TriangleCollision)  (100%)
<details>
  <summary>Details</summary>
  This project involved the implementation of 2D collision mechanics for randomly placed moving ships. Using the main script one would be able to add or remove the number of ships, and take manual control over the frame updates. 
  
  ![Preview](https://github.com/FDSchaefer/public/blob/master/README/ColliderGif.gif?raw=true)
  
</details>



