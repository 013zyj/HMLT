# Hierarchical Multi-scale Learning Transformer for Video-based Person Re-identification

Implementation of the proposed HMLT.


## Getting Started
### Requirements
Here is a brief instruction for installing the experimental environment.
```
# install virtual envs
$ conda create -n HMLT python=3.6 -y
$ conda activate HMLT
# install pytorch 1.8.1/1.6.0 (other versions may also work)
$ pip install timm scipy einops yacs opencv-python tensorboard pandas
```

### Download pre-trained model
The pre-trained vit model can be downloaded in this [link](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and should be put in the `/home/[USER]/.cache/torch/checkpoints/` directory.

### Dataset Preparation
For iLIDS-VID, please refer to this [issue](https://github.com/deropty/PiT/issues/2).

## Training and Testing
```
# This command below includes the training and testing processes.
$ python train.py --config_file configs/MARS/hmlt.yml MODEL.DEVICE_ID "('0')" 
# For testing only, the parameter TEST.WEIGHT in yml file should be the directory of model weights. Otherwise, it should be None.
```


## Results in the Paper
The results of MARS and iLIDS-VID are trained using one 24G NVIDIA GPU and provided below. You can change the parameter `DATALOADER.P` in yml file to decrease the GPU memory cost.

| Model | Rank-1@MARS | Rank-1@iLIDS-VID |
| --- | --- | --- |
| HMLT |  [91.45] |  [96.00] |


 ```
$ python test.py --config_file configs/MARS/hmlt.yml MODEL.DEVICE_ID "('0')" 
```


## Acknowledgement

This repository is built upon the repository [TranReID](https://github.com/damo-cv/TransReID).

## Citation
If you find this project useful for your research, please kindly cite:


## License
This repository is released under the GPL-2.0 License as found in the [LICENSE](LICENSE) file.
