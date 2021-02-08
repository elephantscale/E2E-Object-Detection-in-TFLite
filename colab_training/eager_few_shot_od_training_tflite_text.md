# Object Detection with TFLite

## Introduction

Welcome to the colab for Object Detection in TFLite. Here, we will be examining how we can perform object detection on 
a new dataset with new classes 

Object detection is a class of problems which involve identifying objects within images.    


## Imports

This notebook will currently only run on TF v2.5+  THat means that at the moment that we need to install tf-nightly.


### Cloning Tensorflow Models

We will need to install the tensorflow models from github.

### Install Object Detection API

We will be installing the Object Detection API for Tensorflow 2.x.  You can find some documents on it [here.](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).

### Initialize All Random Possible seeds

This is a must for having reproducible results.


### Download Images

We will be using the salad dataset.  This dataset has 225 unique images with a total of 1757 unique objects identified within those images. Clearly,
most images have more than one object identified.  

Each identified object is called an annotation.  It consists of a box drawn around the object (specified by coordinates), with a label that 
indicates what the object is.

We will be downloading the .csv file that has the dataset, and then we will be downloading each of the images in that CSV file and storing them
into a folder.



### Data Cleaning

We need to do a little massaging of the .csv file while we load it into a pandas dataframe.  There are some extraneous columns which we'v called e1,e2,e3,e4 that we are going to drop.  We will also use a regex to adjust the paths to the local path directory structure we made.

### Extract the number of images that should be recognized.

There a total of 5 labels of the objects in these images:

1. Baked Goods
2. Tomato
3. Cheese
4. Salad
5. Seafood

As you can see it is not entirely balanced, although the imbalance is not too severe.  Unbalanced datasets tend to perform more poorly on the minority classes, such as "Baked Goods" in this case.

### Utilities

These are some utility functions to do the following:

1. Load the data into a a numpy array, while at the same time shrinking it into the requried 320x320 pixel size required by the model.
2. Plot the detections

### Salad Data

Let us now look at the salad dataset.  The original dataset used to train the SSD MobileNet V2 model used in this example is called the COCO dataset, which has nearly 100 possible classes.  However, none of the classes identified in the COCO dataset really belong to salads, so this dataset needs to be adapted through transfer learning to be able to identify objects commonly found in salads.

#### Load Images

Loading the images into memory will take some time.  We will load all 255 unique images into memory along with all the boxes that represent different objects in the dataset.

#### Visualize A Few Images

Let us look at what a few of the salad images look like.  Note that the salads are often one of a number of things one might find in an image.


## Transfer Learning

Transfer learning means that we are taking a model that is trained to recognize one kind of data, such as the COCO dataset with 90+ varied classes, and transfer it to a more specific problem, in this case, identifying objects related to salads.


### Building Category Index

We need to build an index of the five classes that we want to detect. For that end, we will have a dictionary we can use exchange the indexed numeric class label with the actual Text.



