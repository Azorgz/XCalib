# Unsupervised Cameras Calibration for Thermal-Visible Image Registration
Pytorch implementation of the paper "Unsupervised Cameras Calibration for Thermal-Visible Image Registration".

### [Paper]()

## Abstract
>
## Prerequisites
* Python 3.11 
* See environment file
* Download the weights for your chosen monocularDepth Core [depth_pro](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt), [zoeDepth](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth)
* CUDA 11.6.55, CuDNN 8.4, and Ubuntu 20.04.

## Data Preparation 
Edit the file "config/dataset/MyDataset.yaml" to put your paths toward RGB and IR source folder and your Dataset Name. The frames have to be synchronized as much as possible to get a better pose estimation
Adjust the Image Shape according your data.
In the file "config/XCalib.yaml" you can adjust the 'max_steps' value to fit your need. - 400 should works for 80% of the datasets -
You can also connect to your Weight&Bias account if you got one. If you prefer to get your result locally, put 'mode' of the wandb section to "disabled".
```
mkdir FLIR_datasets
# The directory structure should be this:
Mydataset
   ├── Cam1_name
       ├── IM_01002.png 
       └── ...
   ├── Cam2_name
       ├── IM_02135.png
       └── ...
   ├── Cam3_name
       ├── IM_03175.png
       └── ...
```
## Inference Using Pretrained Model

<details>
  <summary>
    <b>1) Run</b>
  </summary>
  
un the command 
```bash
python /XCalib/xcalib.py
```