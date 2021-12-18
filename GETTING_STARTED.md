# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).



## Prepare DOTA dataset.
It is recommended to symlink the dataset root to `AerialDetection/data`.

Here, we give an example for single scale data preparation of DOTA-v1.0.

First, make sure your initial data are in the following structure.
```
data/dota
├── train
│   ├──images
│   └── labelTxt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```
Split the original images and create COCO format json. 
```
python DOTA_devkit/prepare_dota1.py --srcpath path_to_dota --dstpath path_to_split_1024
```
Then you will get data in the following structure
```
dota1_1024
├── test1024
│   ├── DOTA_test1024.json
│   └── images
└── trainval1024
     ├── DOTA_trainval1024.json
     └── images
```
For data preparation with data augmentation, refer to "DOTA_devkit/prepare_dota1_aug.py"

For data preparation of dota1.5, refer to "DOTA_devkit/prepare_dota1_5.py" and "DOTA_devkit/prepare_dota1_5_aug.py"


## Inference with pretrained models


### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/`.

1. Test DEA.
2. Due to the optimization and supplementary experiments during the review process, we are sorting and optimizing the code and will update it as soon as possible.

```shell
python tools/test.py configs/DOTA/faster_rcnn_RoITrans_r101_fpn_dea.py \
    work_dirs/faster_rcnn_RoITrans_r101_fpn_dea/epoch_12.pth \ 
    --out work_dirs/faster_rcnn_RoITrans_r101_fpn_dea/results.pkl
```


2. Parse the results.pkl to the format needed for [DOTA evaluation](http://117.78.28.204:8001/)

For methods with only OBB Head, set the type OBB.
```
python tools/parse_results.py --config configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --type OBB
```

### Demo of inference in a large size image.


```python
python demo_large_image.py
```


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.


### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (recommended): Perform evaluation at every k (default=1) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.





