import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size: int, pool_size: int):
        """
        This class represents a CNN block which projects
        `in_channels` to a usually higher amount of channels
        `out_channels`. It runs the image through `depth` layers
        with a kernel of size `kernel_size`. In order to keep
        the resolution of the image the same, it must be padded
        with `kernel_size//2` zeroes around the image. 

        The image is then pooled with a MaxPool2d operation of size
        `pool_size`, the resulting is a model that has input of shape
        (batch, in_channels, width, height), and outputs an image of
        shape (batch, out_channels, width//pool_size, height//pool_size).
        
        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolutional layers in encoder block
            kernel_size: size of the kernel of the convolutional layers
            pool_size: size of the kernel of the pooling layer
        """
        super(Encoder, self).__init__()

        # create a list of convolutions
        layers = []
        # append the first layer, with in_channels as the in_channels and out_channels as the out_channels
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        # append a ReLU layer
        layers.append(nn.ReLU(inplace=True))
        # for each value in range(depth-1)
        for _ in range(depth-1):
            # create a conv2d layer with inputs out_channels and outputs out_channels
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            # create a relu
            layers.append(nn.ReLU(inplace=True))
        # extract the list into an nn.Sequential as self.convs
        self.convs = nn.Sequential(*layers)
        # create a maxpool2d layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, img):
        """
        runs the partial prediction in the encoder block

        Inputs:
            img: input image of shape
            (batch, in_channels, width, height)

        Outputs:
            img: output image of shape
            (batch, out_channels, width//pool_size, height//pool_size)
        """
        img = torch.nan_to_num(img)
        img = self.convs(img)
        img = self.pool(img)
        return img


class SegmentationCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, embedding_size: int = 64,
                 pool_sizes: List[int] = [5, 5, 2], kernel_size: int = 3, **kwargs):
        """
        Basic CNN that performs segmentation. This model takes
        an image of shape (batch, in_channels, width, height)
        and outputs an image of shape 
        (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes)),
        where prod() is the product of all the values in pool_sizes.

        This is done by using len(pool_sizes) Encoder layers, each of which
        pools the resolution down by a factor of pool_sizes[i].

        The first encoder must project the in_channels to embedding_size.
        Each subsequent layer must double the number of channels in its input,
        for example, the second layer must go from embedding_size to 2*embedding_size,
        the third layer from 2*embedding_size to 4*embedding_size and so on, until
        the pool_sizes list has been depleted. 

        The final layer (decoder) must project the (2**len(pool_sizes))*embedding_size 
        channels to output_channels channels. In order to keep the resolution, you 
        may use a 1x1 kernel.

        Hint: If you use a regular list to save your Encoders, these
        will not register with the pytorch module and will not
        have their parameters added to the optimizer. Use nn.ModuleList to avoid this.
        """
        super(SegmentationCNN, self).__init__()

        # save in_channels, out_channels, depth and embedding_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.embedding_size = embedding_size

        # create a list of encoders
        self.encoders = nn.ModuleList()
        # append the first encoder layer, with in_channels as the in_channels and embedding_size as the out_channels
        self.encoders.append(Encoder(in_channels=in_channels, out_channels=embedding_size, depth=depth, kernel_size=kernel_size, pool_size=pool_sizes[0]))
        # for each value of pool_sizes after the first one
        for pool_size in pool_sizes[1:]:
            # append an Encoder layer with inputs embedding_size and outputs 2*embedding_size
            self.encoders.append(Encoder(in_channels=embedding_size, out_channels=2*embedding_size, depth=depth, kernel_size=kernel_size, pool_size=pool_size))
            # double the embedding_size
            embedding_size *= 2

        # create a decoder (conv2d layer) from embedding_size to out_channels
        self.decoder = nn.Conv2d(in_channels=embedding_size, out_channels=out_channels, kernel_size=1)

    def forward(self, X):
        """
        Runs the input X through the encoders and decoders.
        Inputs:
            X: image of shape 
            (batch, in_channels, width, height)
        Outputs:
            y_pred: image of shape
            (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes))
        """
        X = torch.nan_to_num(X)
        # for each encoder, run X through the encoder
        for encoder in self.encoders:
            X = encoder(X)
        # for the decoder, run X through that layer
        X = self.decoder(X)
        
        return X
