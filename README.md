# Computer Vision Final Project

In this project, we re-implement [this paper](https://arxiv.org/abs/1904.09569) based on author's original repo.

-----
### 1. [Install](#install)
### 2. [Usage](#usage)
#### 2.a [Inference](#infer)
#### 2.b [Train](#train)
#### 3. [Maintainer](#maintainer)
-----

## <div id="#install"> Install </div>
First, follow the tutorial in [link](https://pytorch.org/) to install PyTorch.

For inference only install:
``` shell
# Install (Need ssh key)
pip install git+ssh://git@github.com/xing1999/PoolNet@v.0.1.0

# Install (no ssh key)
pip install git+https://github.com/xing1999/PoolNet@v.0.1.0
```

For modify code, please first clone this repo, then in this folder, run:
```
pip install -e .[dev]
```

## <div id="#infer"> Inference </div>

First, choose model weight to download (Or train your own model).

[Link](https://drive.google.com/drive/folders/1SEfT66id2yIPFhqN-1d7KFvD7mg517Fc?usp=sharing)

Example code:
```
from pool_net import PoolNetInterface

weight_path = "path/to/model/weight.cptk"

# Load model
model = PoolNetInterface(weight_path, device="gpu")

# Call function
mask = model.process("path/to/image/file.png")
```

## <div id="#train"> Training </div>
To train in normal mode, please download data from original repos.

The csv file should have format:
```
path/to/img0.jpg path/to/gt0.png
path/to/img1.jpg path/to/gt1.png
path/to/img2.jpg path/to/gt2.png
```

To train, you need to cd to `pool_net` folder, and call `main.py`.

Example:
```
python main.py --train_root /path/to/train/folder/root \
    --train_csv_file /path/to/train.csv \
    --batch_size 8 \
    --val_csv_file /path/to/val.csv \
    --epochs 40 \
    --n_gpus 1 
```


To train in edge mode, your csv should have format like below:
```
path/to/img0.jpg path/to/gt0.png path/to/edge_gt0.png
path/to/img1.jpg path/to/gt1.png path/to/edge_gt1.png
path/to/img2.jpg path/to/gt2.png path/to/edge_gt2.png
```

Then call the `edge_main.py` in `pool_net` like this example:
```
python edge_main.py --train_root /path/to/train/folder/root \
    --train_csv_file /path/to/train.csv \
    --batch_size 8 \
    --val_csv_file /path/to/val.csv \
    --epochs 40 \
    --n_gpus 1 
```

To view tensorboard, call:
```
tensorboard --logdir lightning_logs/
```

Your model weight will be saved under: `lightning_logs/version_<your version>/checkpoints/`
