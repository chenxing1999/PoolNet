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
# Install master (unstable)
pip install lib-poolnet@git+ssh://git@github.com:xing1999/PoolNet@master
```

For modify code, please first clone this repo, then in this folder, run:
```
pip install -e .[dev]
```
