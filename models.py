## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F   # Mainly for activation functions
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # As we will be feeding in the image of size 224 x 224px (Input size, W=224)
        # filter size, F
        # padding, P=0
        # stride, S
        # Formula: (W-F+2P)/S+ 1. In this case, P=0, therefore the understanding of number of filters/kernels and 
        # the reduced formula, (W-F)/S +1 can be used to compute the dimensions of the output 
        
        # Conv2D layer 1
        self.conv1 = nn.Conv2d(1, 32, 5)    ## output tensor shape: (For W:224, F:5, S:1) ... (32, 220, 220)
        self.pool1 = nn.MaxPool2d(2, 2)     # maxpool that uses a square window of kernel_size=2, stride=2
        # after maxpooling, output tensor shape: (For W: 220, F:2, S:2, Using formula (W-F)/S +1) ... (32, 110, 110)

        
        # Conv2D layer 2
        self.conv2 = nn.Conv2d(32, 32, 3)   ## output tensor shape: (For W: 110, F: 3, S:1) ... (32, 108, 108)
        self.pool2 = nn.MaxPool2d(2, 2)     # maxpool that uses a square window of kernel_size=2, stride=2
        # after maxpooling, output tensor shape: (For W: 108, F:2, S:2) ... (32, 54, 54)

        
        # Conv2D layer 3
        self.conv3 = nn.Conv2d(32, 64, 3)   ## output tensor shape: (For W: 54, F: 3, S:1) ... (64, 52, 52)
        self.pool3 = nn.MaxPool2d(2, 2)     # maxpool that uses a square window of kernel_size=2, stride=2
        # after maxpooling, output tensor shape: (For W: 52, F:2, S:2) ... (64, 26, 26) 

        
        # Conv2D layer 4
        self.conv4 = nn.Conv2d(64, 64, 1)   ## output tensor shape: (For W: 26, F: 1, S:1) ... (64, 26, 26)
        self.pool4 = nn.MaxPool2d(2, 2)     # maxpool that uses a square window of kernel_size=2, stride=2
        # after maxpooling, output tensor shape: (For W: 26, F:2, S:2) ... (64, 13, 13)   
        
        
        # Fully-Connected layer 1, fc1
        self.fc1 = nn.Linear(64*13*13, 4096)
        # fc1_dropout with p=0.5
        self.fc1_dropout = nn.Dropout(p=0.5)
        
        # Fully-Connected layer 2, fc2
        self.fc2 = nn.Linear(4096, 1024)
        # fc2_dropout with p=0.5
        self.fc2_dropout = nn.Dropout(p=0.5)

        # Fully-Connected layer 3, fc3
        self.fc3 = nn.Linear(1024, 136)    # Since, 68*2= 136 values, 2 for each of the 68 keypoint (x, y) pairs

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # All conv2D+ ReLU + Maxpooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # Use to Flatten in PyTorch
        
        # Fully-Connected Layer 1
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        
        # Fully-Connected Layer 2
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        
        # Fully-Connected Layer 3
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
