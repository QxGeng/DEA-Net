
# Anchor Retouching via Model Interaction forRobust Object Detection in Aerial Images
In this paper, we present an effective Dynamic Enhancement Anchor (DEA) network to construct a novel training sample generator. Different from other state-of-the-art techniques, the proposed network leverages a sample discriminator to realize interactive sample screening between an anchor-based unit and an anchor-free unit to generate eligible samples. Besides, multi-task joint training with a conservative anchor-based inference scheme enhances the performance of the proposed model while suppressing computational complexity. The proposed scheme supports both oriented and horizontal object detection tasks. Extensive experiments on two challenging aerial benchmarks (i.e., DOTA and HRSC2016) indicate that our method achieves state-of-the-art performance in accuracy with moderate inference speeds and computational overhead for training.

## Introduction
This codebase is created to build benchmarks for object detection in aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

## Results
Visualization results for oriented object detection on the test set of DOTA.
![Different class results](/show/show_all.png)

Comparison to the baseline on DOTA for oriented object detection. The figures with blue boxes are the results of the baseline and pink boxes are the results of our proposed DEA-Net.
![Baseline and DEA-Net results](/show/show_compare.png)

## Benchmark and model zoo
ImageNet Pretrained Model from Pytorch
- [ResNet50](https://drive.google.com/file/d/1mQ9S0FzFpPHnocktH0DGVysufGt4tH0M/view?usp=sharing)
- [ResNet101](https://drive.google.com/file/d/1qlVf58T0fY4dddKst5i7-CL3DXhBi3Mp/view?usp=sharing)
- [ResNet152](https://drive.google.com/file/d/1y08s30DdWUyaFU89vEpospMi8TjqrJIz/view?usp=sharing)  
- You can find the detailed configs in configs/DOTA.
- The trained models are available at [Google Drive](https://drive.google.com/file/d/1_Vz59vWp0YE36ashdMTWTn3KNZgtz6Ur/view?usp=sharing)

## Installation
 Please refer to [INSTALL.md](INSTALL.md) for installation.    
 
## Get Started
Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing
We appreciate all contributions to improve benchmarks for object detection in aerial images. 

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
