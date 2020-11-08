**Brain MRI: 3D Convolutional AutoEncoder & Patient Age 3D Classifier** 

This project involves the collection of healthy brain MRI images with various patient ages. The autoencoder compresses the 3D MRI data to a more manageable form for the Age classifier network. (This is due ot the memory limitations of my GPU). The Convolutional classifier then reads the encoded data, to predict the age of the patient, of whom the MRI was taken. The network was written in Pytorch, with CUDA compatability. The data aquired from: https://www.insight-journal.org/midas/community/view/21

<details>
  <summary>AutoEncoder Images</summary>
  
  ![AutoEncoder Training Details](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/BRAIN%20MRI/README/LossCurves.jpeg)
  
  ![Decoded Images MSE](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/BRAIN%20MRI/README/MSEperImage.jpeg)
  
  ![Image Comparison](https://github.com/FDSchaefer/public/blob/master/Deep%20Learning/BRAIN%20MRI/README/gif2.gif)
  
</details>  
