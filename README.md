# Udacity-CVND-Project1-Facial-Keypoints-Detection
Applying knowledge of image processing and deep learning to create a convolutional neural network (CNN) for facial keypoints (eyes, mouth, nose, etc.) detection.


## About the Project

This project will be all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces.  The first step in any challenge like this is to load and visualize the data we work with. Let's take a look at some examples of images and corresponding facial keypoints.<br>

<img src='images/key_pts_example.png' width=60% height=60%/>

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y), for that face**.  These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.<br>

<img src='images/landmarks_numbered.jpg' width=40% height=40%/>

The model is tested on a real-life family photo to check if it works. Please find the example below. (Refer to 3.1 Jupyter notebook for the code.) 

## Project Instructions

The project is broken up into a few main parts in four Python notebooks:

**Notebook 1** : Loading and Visualizing the Facial Keypoint Data

**Notebook 2** : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

**Notebook 3** : Facial Keypoint Detection Using Haar Cascades and the Trained CNN

**Notebook 4** : Fun Filters and Keypoint Uses


## Implementation

The complete computer vision pipeline consists of:

1.  Detecting faces on the image with [OpenCV](https://opencv.org/) [Haar Cascades](https://en.wikipedia.org/wiki/Haar-like_feature).
2.  Detecting 68 facial keypoints with CNN with architecture based on [this paper](https://arxiv.org/pdf/1710.00977.pdf).

## Example: Testing the model on a real-life photo.

[![image](https://github.com/ChaitanyaC22/Udacity-CVND-Project1-Facial-Keypoints-Detection/blob/chai_main/images/family_example_facial_keypoints_detection.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project1-Facial-Keypoints-Detection/blob/chai_main/images/family_example_facial_keypoints_detection.png)
