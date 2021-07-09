# Udacity-CVND-Project1-Facial-Keypoints-Detection
Applying knowledge of image processing and deep learning to create a convolutional neural network (CNN) for facial keypoints (eyes, mouth, nose, etc.) detection.


## About the Project

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The resultant CNN architecture is able to look at any image, detect faces, and predict the locations of facial keypoints on each face.<br>
The model is tested on a real-life family photo to check if it works. Please find the example below. (Refer to 3.1 Jupyter notebook for the code.) 

## Implementation

The complete computer vision pipeline consists of:

1.  Detecting faces on the image with [OpenCV](https://opencv.org/) [Haar Cascades](https://en.wikipedia.org/wiki/Haar-like_feature).
2.  Detecting 68 facial keypoints with CNN with architecture based on [this paper](https://arxiv.org/pdf/1710.00977.pdf).

## Example: Testing the model on a real-life photo.

[![image](https://github.com/ChaitanyaC22/Udacity-CVND-Project1-Facial-Keypoints-Detection/blob/chai_main/images/family_example_facial_keypoints_detection.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project1-Facial-Keypoints-Detection/blob/chai_main/images/family_example_facial_keypoints_detection.png)
