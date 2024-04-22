from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ultralytics import YOLO
from ultralytics.engine.results import Results

import torch as th
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Identity
from torchvision import models, transforms
from typing import List
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

from gymnasium import spaces
import gymnasium as gym

from typing import Dict

from pathlib import Path

from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict


def assign_clusters(data: List[int], eps: float, min_pts: int) -> List[int]:
    cluster_labels: List[int] = [-1] * len(data)  # All points initially marked as unprocessed/noise
    cluster_id: int = 0

    def neighbors(point_idx: int) -> List[int]:
        """ Return the indices of all points within 'eps' of 'point_idx' """
        return [i for i in range(len(data)) if abs(data[point_idx] - data[i]) <= eps]

    for point_idx in range(len(data)):
        if cluster_labels[point_idx] != -1:
            continue  # Already processed or assigned to a cluster

        nearby_points: List[int] = neighbors(point_idx)
        if len(nearby_points) < min_pts:
            cluster_labels[point_idx] = -1  # Mark as noise if not enough points in neighborhood
        else:
            # Start a new cluster
            cluster_labels[point_idx] = cluster_id
            i: int = 0
            while i < len(nearby_points):
                point: int = nearby_points[i]
                if cluster_labels[point] == -1:
                    cluster_labels[point] = cluster_id  # Change from noise to part of the cluster
                    point_neighbors: List[int] = neighbors(point)
                    if len(point_neighbors) >= min_pts:
                        # Merge density-reachable points into the seed set
                        nearby_points = list(set(nearby_points + point_neighbors))
                i += 1
            cluster_id += 1  # Prepare next cluster id

    return cluster_labels


def cluster_then_mean(data: List[int], eps: float = 0.1, min_pts: int = 5) -> float:
    cluster_labels = assign_clusters(data, eps, min_pts)

    # Count the frequency of each cluster label (excluding noise)
    cluster_sizes = {}
    for label in cluster_labels:
        if label != -1:
            if label in cluster_sizes:
                cluster_sizes[label] += 1
            else:
                cluster_sizes[label] = 1

    if len(cluster_sizes) == 0:
        return 0.

    # Find the cluster ID with the maximum size
    largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

    # Collect all points that belong to the largest cluster
    largest_cluster = [data[i] for i in range(len(data)) if cluster_labels[i] == largest_cluster_id]

    # Calculate the mean of the largest cluster
    return np.mean(largest_cluster)

def scale_point(point_0_1):
    return (point_0_1 * 2) - 1


def normalize_depth(depth: float, max_distance: float = 2.) -> float:
    return depth / max_distance

    
def get_depth(xywh: th.Tensor, depth: th.Tensor) -> float:
    as_int = xywh.int()
    point = depth[as_int[0], as_int[1]].item()
    return normalize_depth(point)


def result_to_output(results: list[Results], depths: th.Tensor) -> th.Tensor:
    outputs = th.zeros((len(results), 3))
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


class BulletTrackYOLO(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_dim: int = 3,
                 normalized_image: bool = False,
                 pretrained_weights: Path | None = Path.cwd() /
                                            "bullet-track-yolov8-s-drone-detector-single-cls" /
                                            "train" /
                                            "weights" /
                                            "best.pt"):
        super().__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.input_size = observation_space.shape  # Extracting the input size from the observation space
        self.normalized_image = normalized_image
        self.model = UntrainableYOLO(pretrained_weights)
        
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        images = x[:, 0:3, :, :]
        depths = x[:, 3, :, :]
        with th.no_grad():
            result = isolated_forward(images)
        return result_to_output(result, depths)


class BulletTrackEfficientNet(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.Space,
                 features_dim: int = 12,
                 normalized_image: bool = False,
                 pretrained_weights: "Path | None" = None):
        super().__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.input_size = observation_space.shape

        # Initialize EfficientNet with the desired pretrained weights
        if pretrained_weights is None:
            self.backbone = models.efficientnet_b0(models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # self.model = models.efficientnet_b0(models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        else:
            raise RuntimeError("Add support: todo")
        # Adjust first convolutional layer to accept 4-channel input
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(self.input_size[0],
                                                 original_conv.out_channels,
                                                 kernel_size=original_conv.kernel_size,
                                                 stride=original_conv.stride,
                                                 padding=original_conv.padding,
                                                 bias=original_conv.bias)
        if pretrained_weights is not None:
            with th.no_grad():
                self.backbone.features[0][0].weight[:, :3] = original_conv.weight
                # Initialize the new channel to 0
                self.backbone.features[0][0].weight[:, 3] = 0
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, features_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.backbone(x)


class BulletTrackCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        image_feature_extractor = BulletTrackYOLO,
        feature_dims: int = 0
    ) -> None:
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

class TransformerExtractor(BulletTrackCombinedExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        image_feature_extractor = BulletTrackYOLO,
        feature_dims: int = 0,
        embedding_dims = 128,
        num_heads = 4,
        num_layers = 3
    ) -> None:
        super().__init__(observation_space, cnn_output_dim, normalized_image, image_feature_extractor, feature_dims)

        self.embedding = nn.Linear(self._features_dim, embedding_dims)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dims,
                                                nhead=num_heads,
                                                dim_feedforward=embedding_dims * 4)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Update the features dim manually
        self._features_dim = embedding_dims

    def forward(self, observations: TensorDict) -> th.Tensor:
        extracted_features = super().forward(observations)
        t = self.embedding(extracted_features)
        t = self.transformer_encoder(t)
        return t

class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
