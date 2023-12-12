# STL-RANet: Residual Attentive Network and Self-selective Learning for Unsupervised Video Anomaly Detection

## Introduction
This is an official pytorch implementation of <STL-RANet: Residual Attentive Network and Self-selective Learning for
Unsupervised Video Anomaly Detection>. 

## Environments

- Linux
- Python 3.7
- PyTorch 1.7.1
- torchvision 0.8.2
- scipy 1.7.1
- opencv-python 4.5.4.58
- pillow 8.2.0

## Data Preparation

Taking the ShanghaiTech Dataset as an example, we recommend you to extract the STCs as below. Please make sure that you have sufficient storage.

```shell
python gen_patches.py --dataset shanghaitech --phase test --filter_ratio 0.8 --sample_num 9
```
| Dataset         | # Patch (train) | # Patch (test)	 | filter ratio	 | sample num	 | storage |
|:----------------|:---------------:|:---------------:|:-------------:|:-----------:|:-------:|
| Ped2            |      27660      |      31925      |      0.5      |      7      |   20G   |
| Avenue          |      96000      |      79988      |      0.8      |      7      |   58G   |
| ShanghaiTech    |     145766      |     130361      |      0.8      |      9      |  119G   |


## Training
The best selection factos as below.

| Selection factors | S-factor | T-factor |
|:------------------|:--------:|:---------:|
| Ped2              |   0.9    |    0.9    |
| Avenue            |   0.9    |    0.9    |
| ShanghaiTech      |   0.9    |    0.7    |

```shell
python UVAD_main.py --dataset shanghaitech --val_step 2400 --print_interval 100 --batch_size 64 --sample_num 9 --epochs 100 --static_threshold 0.2
```

## Testing

```shell
python UVAD_main.py --dataset shanghaitech/avenue/ped2 --sample_num 9/7/7 --checkpoint xxx.pth
```
