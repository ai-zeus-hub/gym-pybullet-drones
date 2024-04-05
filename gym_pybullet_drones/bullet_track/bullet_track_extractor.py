from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch as th
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Identity
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

from gymnasium import spaces
import gymnasium as gym

from typing import Dict

from pathlib import Path

from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict


def scale_point(point_0_1):
    return (point_0_1 * 2) - 1


def normalize_depth(depth: float, max_distance: float = 2.) -> float:
    return depth / max_distance

    
def get_depth(xywh: th.Tensor, depth: th.Tensor) -> float:
    as_int = xywh.int()
    point = depth[as_int[0], as_int[1]].item()
    return normalize_depth(point)


def result_to_output(results: list[Results], depths: th.Tensor) -> th.Tensor:
    outputs = th.zeros((len(results), 3), device=depths.device)
    for result_index, result in enumerate(results):
        if len(result.boxes) == 0:
            # Return a tensor with zeros if no detections are found
            outputs[result_index] = th.tensor([-1.0, -1.0, -1.0])
            continue

        best_box_index = 0
        best_box_confidence = 0.
        for box_index, conf in enumerate(result.boxes.conf):
            if conf > best_box_confidence:
                best_box_index = box_index

        best_box_xywh = result.boxes.xywh[best_box_index]
        best_box_xywhn = result.boxes.xywhn[best_box_index]
        x: float = best_box_xywh[0]
        y: float = best_box_xywh[1]
        z: float = get_depth(best_box_xywh, depths[result_index])
        outputs[result_index][0] = scale_point(best_box_xywhn[0])
        outputs[result_index][1] = scale_point(best_box_xywhn[1])
        outputs[result_index][2] = scale_point(z)
    return outputs


class UntrainableYOLO(YOLO):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

        for param in self.parameters():
            param.requires_grad = False

    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        pass

model = None

def isolated_forward(x):
    global model
    if model is None:
        model = UntrainableYOLO("bullet-track-yolov8-s-drone-detector-single-cls/train/weights/best.pt")
    bgr_tensor = x[:, [2, 1, 0], :, :]
    return model.predict(bgr_tensor, conf=0.35, imgsz=64, verbose=False)


# class BulletTrackCNN(BaseFeaturesExtractor):
#     def __init__(self,
#                  observation_space: gym.Space,
#                  features_dim: int = 3,
#                  normalized_image: bool = False,
#                  pretrained_weights: Path = Path.cwd() /
#                                             "bullet-track-yolov8-s-drone-detector-single-cls" /
#                                             "train" /
#                                             "weights" /
#                                             "best.pt"):
#         super().__init__(observation_space, features_dim)
#         self.observation_space = observation_space
#         self.input_size = observation_space.shape  # Extracting the input size from the observation space
#         self.normalized_image = normalized_image
#         self.model = UntrainableYOLO(pretrained_weights)

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         images = x[:, 0:3, :, :]
#         depths = x[:, 3, :, :]
#         with th.no_grad():
#             # result = self.model.predict(images, imgsz=self.input_size[1:])
#             result = isolated_forward(images)  # todo: ajr -- ok? , device=0 -- verify using cuda:0 later
#         return result_to_output(result, depths)

class BulletTrackCNN(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_dim: int = 3,
                 normalized_image: bool = False,
                 pretrained_weights: Path | None = None): 
                                            # Path.cwd() /
                                            # "bullet-track-yolov8-s-drone-detector-single-cls" /
                                            # "train" /
                                            # "weights" /
                                            # "best.pt"):
        super().__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.input_size = observation_space.shape  # Extracting the input size from the observation space
        self.normalized_image = normalized_image
        # self.model = UntrainableYOLO(pretrained_weights)
        if pretrained_weights is None:
            self.model = models.efficientnet_b0(models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        else:
            raise RuntimeError("Add support: todo")
        
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.model(x)
        return out
        # images = x[:, 0:3, :, :]
        # depths = x[:, 3, :, :]
        # with th.no_grad():
        #     # result = self.model.predict(images, imgsz=self.input_size[1:])
        #     result = isolated_forward(images)  # todo: ajr -- ok? , device=0 -- verify using cuda:0 later
        # return result_to_output(result, depths)



class BulletTrackCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        image_feature_extractor = BulletTrackCNN,
        feature_dims: int = 0
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = image_feature_extractor(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        if feature_dims > 0:
            self.out = nn.Linear(in_features=total_concat_size, out_features=feature_dims)
        else:
            self.out = nn.Flatten()
            feature_dims = total_concat_size

        # Update the features dim manually
        self._features_dim = feature_dims

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            observation = observations[key]
            features = extractor(observation)
            encoded_tensor_list.append(features)
        t = th.cat(encoded_tensor_list, dim=1)
        return self.out(t)

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = BulletTrackCNN(subspace, features_dim=3, normalized_image=normalized_image)
                total_concat_size += 3
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        embedding_dims = 128
        num_heads = 4
        num_layers = 3

        self.embedding = nn.Linear(total_concat_size, embedding_dims)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dims,
                                                nhead=num_heads,
                                                dim_feedforward=embedding_dims * 4)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Update the features dim manually
        self._features_dim = embedding_dims

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            observation = observations[key]
            features = extractor(observation)
            encoded_tensor_list.append(features)
        t = th.cat(encoded_tensor_list, dim=1)
        t = self.embedding(t)
        t = self.transformer_encoder(t)
        return t

# class BulletTrackCNN(BaseFeaturesExtractor):
#     """
#      A neural network for identifying the center point of a drone in an image, based on MobileNetV3 Small architecture.

#      Attributes:
#          observation_space (gym.Space): The space object representing the input dimensions.
#          features_dim (int): The dimensionality of the output layer, defaults to 2 for x and y coordinates.
#          normalized_image (bool): Flag indicating whether input images are normalized.
#      """

#     def __init__(self,
#                  observation_space: gym.Space,
#                  features_dim: int = 3,
#                  normalized_image: bool = False,
#                  intermediate_dims: int = 64):
#         super().__init__(observation_space, features_dim)
#         self.observation_space = observation_space
#         self.input_size = observation_space.shape  # Extracting the input size from the observation space
#         self.normalized_image = normalized_image

#         # Load the pre-trained MobileNetV3 Small model
#         from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
#         self.mobilenet_v3_small = mobilenet_v3_small(pretrained=True)

#         # Check the number of input channels (3 for RGB, 4 for RGB-D)
#         num_input_channels = observation_space.shape[0]
#         if num_input_channels not in [3, 4]:
#             raise ValueError("Unsupported number of input channels. Expected 3 (RGB) or 4 (RGB-D).")

#         # If the input has 4 channels, adapt the first convolutional layer to accept RGB-D input
#         if num_input_channels == 4:
#             first_conv_layer = self.mobilenet_v3_small.features[0][0]
#             new_first_conv_layer = nn.Conv2d(
#                 in_channels=4,  # Change from 3 to 4 to accept RGB-D input
#                 out_channels=first_conv_layer.out_channels,
#                 kernel_size=first_conv_layer.kernel_size,
#                 stride=first_conv_layer.stride,
#                 padding=first_conv_layer.padding,
#                 bias=first_conv_layer.bias
#             )
#             # Copy the weights from the first 3 channels of the pre-trained model
#             with th.no_grad():
#                 new_first_conv_layer.weight[:, :3] = first_conv_layer.weight
#             # Initialize the weights for the new depth channel
#             nn.init.kaiming_normal_(new_first_conv_layer.weight[:, 3:], mode='fan_out', nonlinearity='relu')
#             # Replace the first convolutional layer in the model with the new one
#             self.mobilenet_v3_small.features[0][0] = new_first_conv_layer

#         # Identify the correct number of output features from the last convolutional layer
#         # This is usually the number of channels in the output of the last block before the classifier
#         last_conv_output_channels = self.mobilenet_v3_small.features[-1].out_channels

#         # Replace the classifier of the MobileNetV3 Small to adapt to the task
#         self.mobilenet_v3_small.classifier = nn.Sequential(
#             nn.Linear(last_conv_output_channels, intermediate_dims),
#             nn.Tanh(),
#             nn.Linear(intermediate_dims, features_dim),
#             nn.Tanh(),
#         )

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         """
#         Forward pass through the network.

#         Args:
#             x (torch.Tensor): The input tensor representing a batch of images.

#         Returns:
#             torch.Tensor: The output tensor representing the x and y coordinates of the drone's center point in each image.
#         """
#         return self.mobilenet_v3_small(x)

# class MobileNet(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space: gym.Space,
#         features_dim: int = 512,
#         normalized_image: bool = False,
#     ) -> None:
#         assert isinstance(observation_space, spaces.Box), (
#             "CNN must be used with a gym.spaces.Box ",
#             f"observation space, not {observation_space}",
#         )
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
#             "You should use CNN "
#             f"only with images not with {observation_space}\n"
#             "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
#             "If you are using a custom environment,\n"
#             "please check it using our env checker:\n"
#             "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
#             "If you are using `VecNormalize` or already normalized channel-first images "
#             "you should pass `normalize_images=False`: \n"
#             "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
#         )

#         self.normalized_image = normalized_image
#         self._setup_backbone()

#         self.input_size = observation_space.shape
#         dummy_input = th.autograd.Variable(th.rand(1, *self.input_size))
#         self.output_shape = self._get_output_shape(dummy_input)
#         n_flatten = self.output_shape[1]
#         # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())  # ReLU instead of tanh?

#     def _backbone_fn(self, x):
#         x = self._backbone_transform(x)
#         x = self._backbone(x)
#         return x

#     def _setup_backbone(self):
#         from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
#         _transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms

#         full_backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")
#         for module in full_backbone.modules():
#             if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
#                 module.track_running_stats = False

#         self._backbone = nn.Sequential(
#             full_backbone.features,
#             full_backbone.avgpool
#         )
#         self._backbone_transform = _transform()

#     def _get_output_shape(self, dummy_input):
#         """Function to dynamically compute the output shape"""
#         with th.no_grad():
#             output = self._backbone(dummy_input)
#             return output.shape

#     def forward(self, x: th.Tensor):
#         assert isinstance(x, th.Tensor)
#         # Assuming x is in the format (N, C, H, W)
#         if not self.normalized_image:
#             x = x / 255.
#         x = self._backbone_fn(x)
#         x = x.flatten(1)  # Flatten starting from the second dimension
#         return x

# class FeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super().__init__(observation_space, features_dim=1)
#
#         extractors = {}
#
#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
#                 total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
#             elif key == "vector":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Linear(subspace.shape[0], 16)
#                 total_concat_size += 16
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []
#
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)
