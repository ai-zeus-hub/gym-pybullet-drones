# Autonomous UAV Navigation using Reinforcement Learning with Proximal Policy Optimization


## Setup

```shell
conda create --name drones python=3.10
conda activate drones
pip install poetry
poetry install
```

## Run

```shell
python train.py --image-extractor nature \
  --controls mlp \
  --action rpm  \
  --observation multi \
  --include-rpos
```