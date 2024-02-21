# Visual Attention Network for Classification
- forked from `https://github.com/Visual-Attention-Network/VAN-Classification`
- crated new conda environment `van` with python pytorch
- activate conda using:- 

  ```conda activate van```
- install timm and yaml using:- 

  ```pip install timm==0.4.12 pyyaml```

- downloaded tiny-imagenet dataset from `https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download` for testing
- extracted zip file and placed it in `tiny-imagenet-200`
---
<!-- - rename `tiny-imagenet-200` to `imagenet` -->
- make train.sh executable using:- 

  ```chmod +x train.sh```
- run train.sh using:- 

  ```./train.sh```



- delete van conda environment using:-   

  ```conda env remove -n van```
- create van environment with torchvison using:-
  ```conda create -n van python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia```

- run for single core using

  ```python3 train.py imagenet --model van_b0 -b 128 --lr 1e-3 --drop-path 0.1```