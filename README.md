<p align="center"><img width=100% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/logo.jpg"></p>


![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# TangrIAm Project
#### Real-time tangram form detection from a live video stream.

The project is a partnership between Exploradôme museum, OCTO Technology and Microsoft and it aims to introduce the concept and application of artificial intelligence to young children. The specific application developed for the project is to apply object detection to live tangram solving.

A tangram is a dissection puzzle consisting of seven flat polygons (5 triangles, 1 square and 1 parallelogram) which are combined to obtain a specific shape. The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap.

The goal of the game of tangram is to replicate the outline shown on the target card using a set of seven individual polygons (triangles, square or parallelogram). 
Tangram can be played either by multiple teams at the same time, racing to be the first to finish the outline. 

This project's objective is to detect the similarity between the shape drawn by each player at any point in time and the twelve possible target classes. 
The predictions are to be in real-time from a live video feed of the game board.

Within the framework of the project, 12 tangram selected shapes act as classes for the classifier:

- Boat (Bateau)
- Bow (Bol)
- Bridge (Pont)
- Cat (Chat)
- Fox (Renard)
- Hammer (Marteau)
- Heart (Coeur)
- House (Maison)
- Mountain (Montagne)
- Rabbit (Lapin)
- Swan (Cygne)
- Turtle (Tortue)

<p align="center"><img width=100% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/Montages.jpg"></p>



## Approach & Objective

Our approach has been to avoid the use of Transfer learning techniques and train a custom Convolutional Neural Network (CNN) model to perform real-time recognition of tangram shapes.

The model is built using Keras API on a TensorFlow backend. 

## Data

### Data collection

- Training data was collected by taking a sample video with the webcam (to be used for live recording) and breaking it down into frames (images). 

  The breakdown into frames was made with VLC Media Player.

  Link to video [here](https://drive.google.com/file/d/1bX_x2rNIOm3q86X5xBEyLZxVzltYR2bD/view?usp=sharing)

  Each resulting image was cut in half to obtain two images with a tangram shape on both sides of the board, using :

  

  ```python
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  
  import os
  import imageio
  count = 0
  classe = "cygnes"
  for root, dirs, files in os.walk(f"C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/{classe}", topdown = False):
      for name in files:
          os.path.join(root, name)
          image = imageio.imread(os.path.join(root, name))
          height, width = image.shape[:2]
          width_cutoff = width // 2
          s1 = image[:, :width_cutoff]
          s2 = image[:, width_cutoff : ]
          status = imageio.imwrite(f'C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/image_coupe/{classe}/{classe}_left_{count}.jpg', s1)
          print("Image written to file-system : ",status)
          status = imageio.imwrite(f'C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/image_coupe/{classe}/{classe}_right_{count}.jpg', s2)
          print("Image written to file-system : ",status)
          count += 1
  ```

  

 

  The resulting images were saved in 12 separate folders (by class). 

  Only images with no foreign object (e.g. hands) obstructing the tangram shape were retained. 

  The resulting dataset aimed to relatively balance the available training images by class. The following is the number of initial images per class. 


| Label           |  Total images | 
|-----------------|------|
|boat(bateau)     | 246  | 
| bowl(bol)       | 148  |  
| cat(chat)       | 100  | 
| heart(coeur)    | 216  |  
| swan(cygne)     | 248  |  
| rabbit(lapin)   | 247  |  
| house(maison)   | 136  |  
| hammer(marteau) | 245  |  
| mountain(montagne)  |  313 |  
| bridge(pont)    | 431  |  
| fox(renard)     | 502  |  
| turtle(tortue)  | 164  |  
| TOTAL           | 2996 | 



The dataset is available [here](https://drive.google.com/drive/folders/1CK7x1mHU27PEGIR34WgxyCYxj0yGd9lz?usp=sharing)



# Image processing steps

- Split the video feed into two halves (left and right side)
- Data augmentation on the data to add some noise to built a robust model 
- Resize image to expected size for the model and expaned its dimension


  ## Data augmentation

  The dataset was further augmented and split into training (70% of data), validation (20% of data) and test (10% of data) using [Roboflow](https://roboflow.ai/).

  After data augmentation, each class had 1140 images in the training dataset.

  The filter used to augment the data were:

 * rotation 15° of each side
 * shear
 * brightness
 * blur
 * noise


  
  | Label           |  Before Data Augmentation  |   After Data Augmentation* | 
|-----------------|---------------|--------------|
| boat(bateau)    | 716           |        2148  | 
| bowl(bol)       | 248           |        744   | 
| cat(chat)       | 266           |        800   | 
| heart(coeur)    | 273           |        820   | 
| swan(cygne)     | 321           |        964   | 
| rabbit(lapin)   | 257           |        772   | 
| house(maison)   | 456           |        1368  | 
| hammer(marteau) | 403           |        1209  | 
| mountain(montagne)  |  573      |        1720  | 
| bridge(pont)    | 709           |        2128  | 
| fox(renard)     | 768           |        2304  |  
| turtle(tortue)  | 314           |        942   |  
| **TOTAL**           | **5304**          |   ****        | 



**The final dataset has the following directory structure:**
```
├──  Train  
│    └── bateau: [bateau1.jpg, bateau2.jpg, bateau3.jpg ....]  
│    └── bol: [bol1.jpg, bol.jpg, bol.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├──  Test  
│    └── bateau: [bateau_left1.jpg, bateau_left2.jpg, bateau_left3.jpg ....]  
│    └── bol: [bol_left1.jpg, bol_left2.jpg, bol_left3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│   
└── Valid  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
```


The dataset is available [here](https://drive.google.com/drive/folders/1VSARFx8Y8r9yEGKA9lutmm-34AHzeS51?usp=sharing)
 

# Model

The model was first trained on the initial dataset and despite a good performance on the training and validation set, the model failed to generalize well when tested on a live video stream.  

The following table records the model performance on the initial data. 

Further on, the model was trained on the augment data. 

The model performance on the augmented dataset is presented [here](https://simplonformations-my.sharepoint.com/:x:/g/personal/fmujani_simplonformations_onmicrosoft_com/EYL8EaznSh5LvV0Jm_7D3ekB7MfpqFQv99vXPj7SP2V8Jw?e=mJAaT4)

Link to models [here](https://drive.google.com/drive/u/1/folders/1GpLE5O6VSEYY6Wemhw5pcsaNKVeQSVCq)


# Getting Started

## Project Structure

* Notebooks folder: includes Jupyter notebooks with a detailed explanation on how models are trained
* Models folder: includes trained models
* Tangram_app folder: includes python scripts of preprocessing applicated to our data 
* Tangram_detection.py the main file used for inference


## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)
- Python 3.7.x, 64bit

```bash
pip install opencv-python tensorflow
```

## Get more models

The trained models are available in the `models folder:

```
cd Models/
```
## Inference Execution

All model files can be found in the models folder. To use a model for inference, either connect the camera to your device or select a video file and write the following command line:

```
python Tangram_detection -c [camera] -s [side : left | right] -o [output_folder] -m [model] -i [input folder (OPTIONAL)]
```

**Example:**

```
python Tangram_detection.py -c 1 -s left -o result_pics -m Models/model_gridsearch.h5
```



<p align="center"><img width=100% src="https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS"></p>
