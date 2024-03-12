from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import torch as th
from torch import nn
from gymnasium import spaces
import gymnasium as gym
from typing import Dict

from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import NatureCNN


class MobileNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "CNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use CNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        self.normalized_image = normalized_image
        self._setup_backbone()

        self.input_size = observation_space.shape
        dummy_input = th.autograd.Variable(th.rand(1, *self.input_size))
        self.output_shape = self._get_output_shape(dummy_input)
        n_flatten = self.output_shape[1]
        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())  # ReLU instead of tanh?

    def _backbone_fn(self, x):
        x = self._backbone_transform(x)
        x = self._backbone(x)
        return x

    def _setup_backbone(self):
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        _transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms

        full_backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")
        for module in full_backbone.modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.track_running_stats = False

        self._backbone = nn.Sequential(
            full_backbone.features,
            full_backbone.avgpool
        )
        self._backbone_transform = _transform()

    def _get_output_shape(self, dummy_input):
        """Function to dynamically compute the output shape"""
        with th.no_grad():
            output = self._backbone(dummy_input)
            return output.shape

    def forward(self, x: th.Tensor):
        assert isinstance(x, th.Tensor)
        # Assuming x is in the format (N, C, H, W)
        if not self.normalized_image:
            x = x / 255.
        x = self._backbone_fn(x)
        x = x.flatten(1)  # Flatten starting from the second dimension
        return x

    # def forward(self, observations: th.Tensor) -> th.Tensor:
    #     return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
                # total_concat_size += extractors[key].output_shape[1]
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

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
