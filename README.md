# Autonomous UAV Navigation using Reinforcement Learning with Proximal Policy Optimization


## Setup

```shell
conda create --name drones python=3.10
conda activate drones
pip install poetry
poetry install
```

## Run

First, train yolov8 in to bullet-track-yolov8-s-drone-detector-single-cls with the `train_yolov8.ipynb` to enable YOLO training.
```shell
python train.py --image-extractor nature \
  --controls mlp \
  --action rpm  \
  --observation multi \
  --include-rpos
```