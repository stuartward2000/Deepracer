import torch
import numpy as np
from math import prod
import torch.nn as nn
from deepracer_gym.envs.utils import (
    # un-flattened observation shapes
    LIDAR_SHAPE,
    STEREO_CAMERA_SHAPE,
    FRONT_FACING_CAMERA_SHAPE
)

# pre-processing parameters
LIDAR_RANGE_MAX: float = 1.00
LIDAR_RANGE_MIN: float = 0.15
CAMERA_MAX_MEASUREMENT: int = 255

# encoder params
HIDDEN_CHANNELS: int = 16
LIDAR_LATENT_DIMENSION: int = 32
CAMERA_LATENT_DIMENSION: int = 96

# flattened observation shapes
LIDAR_FLATTEN_SHAPE: int = prod(LIDAR_SHAPE)
STEREO_FLATTEN_CAMERA_SHAPE: int = prod(STEREO_CAMERA_SHAPE)
FRONT_FLATTEN_FACING_CAMERA_SHAPE: int = prod(FRONT_FACING_CAMERA_SHAPE)


def initialize(layer, bias_const=0.01, std=np.sqrt(2)):
    '''Example NN weight initializaiton strategy.'''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def activation():
    '''Example NN activation function.'''
    return nn.LeakyReLU()


class CNN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels: int = HIDDEN_CHANNELS,
            one_dimensional: bool = False
    ):
        super().__init__()
        conv = nn.Conv1d if one_dimensional else nn.Conv2d
        self.network = nn.Sequential(
            initialize(conv(
                in_channels, hidden_channels, kernel_size=4, stride=2,
            )),
            activation(),
            initialize(conv(
                hidden_channels, out_channels, kernel_size=4, stride=2
            )),
            activation(),
        )

    def forward(self, x):
        return self.network(x)


class PreprocessLiDAR(nn.Module):
    '''
    boiler-plate pre-processor for camera observations
    TODO: modify this class as required in case you want to pre-process LiDAR observations.
    '''

    def forward(self, x):
        return x


class PreprocessCamera(nn.Module):
    '''
    boiler-plate pre-processor for camera observations
    TODO: modify this class as required in case you want to pre-process camera observations.
    '''

    def forward(self, x):
        # Convert to float and scale to [0, 1]
        return x.float() / 255.0


class PreprocessStereoCameras(PreprocessCamera):
    '''boiler-plate, in case customisation is required'''


class PreprocessFrontFacingCamera(PreprocessCamera):
    '''boiler-plate, in case customisation is required'''


class EncodeStereoCameras(nn.Module):
    '''
    example CNN encoder for Stereo Camera observations.
    '''

    def __init__(self, latent_dim=CAMERA_LATENT_DIMENSION):
        super().__init__()
        self.image_encoder = nn.Sequential(
            PreprocessStereoCameras(),
            CNN(
                in_channels=1,
                out_channels=HIDDEN_CHANNELS // 2
            )
        )
        self.stereo_encoder = nn.Sequential(
            CNN(
                in_channels=HIDDEN_CHANNELS,
                out_channels=4 * HIDDEN_CHANNELS  # 64
            ),
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            initialize(
                nn.Linear(
                    10 * 16 * HIDDEN_CHANNELS,  # 2560
                    latent_dim,
                )
            ),
            activation(),
        )

    def forward(self, x):
        left = x[:, 0:1, :, :]
        right = x[:, 1:2, :, :]
        left_encoded = self.image_encoder(left)
        right_encoded = self.image_encoder(right)

        stereo = torch.cat(
            (left_encoded, right_encoded), 1
        )
        stereo_encoded = self.stereo_encoder(
            stereo
        )

        return self.output_layer(
            stereo_encoded
        )


class EncodeFrontFacingCamera(nn.Module):
    '''
    example CNN encoder for Front Facing Camera observations.
    '''

    def __init__(self, latent_dim=CAMERA_LATENT_DIMENSION):
        super().__init__()
        self.image_encoder = nn.Sequential(
            PreprocessFrontFacingCamera(),
            CNN(
                in_channels=3,
                out_channels=HIDDEN_CHANNELS
            ),
            CNN(
                in_channels=HIDDEN_CHANNELS,
                out_channels=HIDDEN_CHANNELS
            )
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            initialize(
                nn.Linear(
                    10 * 16 * HIDDEN_CHANNELS,  # 2560
                    latent_dim,
                )
            ),
            activation(),
        )

    def forward(self, x):
        image_encoded = self.image_encoder(x)

        return self.output_layer(
            image_encoded
        )


class EncodeLiDAR(nn.Module):
    '''
    example CNN encoder for LiDAR observations.
    '''

    def __init__(
            self,
            latent_dim=LIDAR_LATENT_DIMENSION,
    ):
        super().__init__()
        self.lidar_encoder = nn.Sequential(
            PreprocessLiDAR(),
            CNN(
                in_channels=1,
                out_channels=HIDDEN_CHANNELS,
                one_dimensional=True,
            ),
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            initialize(
                nn.Linear(
                    14 * HIDDEN_CHANNELS,  # 2560
                    latent_dim,
                )
            ),
            activation(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dim if missing

        lidar_encoded = self.lidar_encoder(x)
        return self.output_layer(
            lidar_encoded
        )


class UnflattenObservation(nn.Module):
    def forward(self, x):
        # # trailing values are LiDAR
        # return (
        #     x[..., :-LIDAR_FLATTEN_SHAPE].view(-1, *STEREO_CAMERA_SHAPE),
        #     x[..., -LIDAR_FLATTEN_SHAPE:].view(-1, *LIDAR_SHAPE)
        # )
        # leading values are LiDAR
        return (
            x[..., LIDAR_FLATTEN_SHAPE:].view(-1, *STEREO_CAMERA_SHAPE),
            x[..., :LIDAR_FLATTEN_SHAPE].view(-1, *LIDAR_SHAPE)
        )


class EncodeObservation(nn.Module):
    '''
    use this class to aggregate all of your observation
    pre-processing and encoding functions if required.
    feel free to modify/change as required!
    '''

    def __init__(self):
        super().__init__()
        self.unflatten = UnflattenObservation()
        self.camera_encoder = EncodeStereoCameras()
        self.lidar_encoder = EncodeLiDAR()

    def forward(self, x):
        camera, lidar = self.unflatten(x)
        camera_encoded = self.camera_encoder(camera)
        lidar_encoded = self.lidar_encoder(lidar)

        return torch.cat(
            (camera_encoded, lidar_encoded), -1
        )
