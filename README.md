# Unsupervised Cameras Calibration for Thermal-Visible Image Registration (X_Calib)
Pytorch implementation of the paper "Unsupervised Cameras Calibration for Thermal-Visible Image Registration".

![tease](https://github.com/FuyaLuo/FoalGAN/blob/main/docs/Model.PNG)

### [Paper](https://ieeexplore.ieee.org/abstract/document/10313314)

## Abstract
>Stable imaging in adverse environments (e.g., total darkness) makes thermal infrared (TIR) cameras a prevalent option for night scene perception. However, the low contrast and lack of chromaticity of TIR images are detrimental to human interpretation and subsequent deployment of RGB-based vision algorithms. Therefore, it makes sense to colorize the nighttime TIR images by translating them into the corresponding daytime color images (NTIR2DC). Despite the impressive progress made in the NTIR2DC task, how to improve the translation performance of small object classes is under-explored. To address this problem, we propose a generative adversarial network incorporating feedback-based object appearance learning (FoalGAN). Specifically, an occlusion-aware mixup module and corresponding appearance consistency loss are proposed to reduce the context dependence of object translation. As a representative example of small objects in nighttime street scenes, we illustrate how to enhance the realism of traffic light by designing a traffic light appearance loss. To further improve the appearance learning of small objects, we devise a dual feedback learning strategy to selectively adjust the learning frequency of different samples. In addition, we provide pixel-level annotation for a subset of the Brno dataset, which can facilitate the research of NTIR image understanding under multiple weather conditions. Extensive experiments illustrate that the proposed FoalGAN is not only effective for appearance learning of small objects, but also outperforms other image translation methods in terms of semantic preservation and edge consistency for the NTIR2DC task. Compared with the state-of-the-art NTIR2DC approach, FoalGAN achieves at least 5.4% improvement in semantic consistency and at least 2% lead in edge consistency. 

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