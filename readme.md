# Fractional Skipping: Towards Finer-Grained Dynamic CNN inference [[PDF]](https://arxiv.org/abs/2001.00705)

Jianghao Shen, Yonggan Fu, Yue Wang, Pengfei Xu, Zhangyang Wang, Yingyan Lin

In AAAI 2020.

## Overview
We present DFS (Dynamic Fractional Skipping),  a dynamic inference framework that extends binary layer skipping options with "fractional skipping" ability - by quantizing the layer weights and activations into different bitwidths.

Highlights:

- **Novel integration** of two CNN inference mindsets: _dynamic_ _layer_ _skipping_ and _static_ _quantization_
-  Introduced _input_-_adaptive_ _quantization_ at inference for the **first time**
-  **Better performance and computational cost tradeoff** than SkipNet and other relevant competitors 

![performance_skipnet](https://i.ibb.co/kH5cghN/CIFAR10-DFS-Res-Net74-vs-Skip-Net74-1.png )
Figure 6: Comparing the accuracy vs. computation percentage of DFS-ResNet74 and SkipNet74 on CIFAR10.

![performance_static](https://i.ibb.co/WHXb9pz/CIFAR10-DFS-Res-Net38-vs-Scalable-HAQ.png)




## Method
![DFS](https://i.ibb.co/yRdw0mL/ezgif-5-ebd7e26308-pdf-1.png)
Figure1. An illustration of the DFS framework, where C1, C2, C3 denote three consecutive convolution layers, each of which consists of a column of filters as represented using cuboids.  For each layer, the decision is computed by the corresponding gating network denoted with "Gx".  In this example, the first conv layer is executed fractionally with a low bitwidth, the second layer is fully executed, while the third one is skipped.



![Gating](https://i.ibb.co/qkbv66X/ezgif-5-f5d1a89614-pdf-1.png)
Figure 2. An illustration of the RNN gate used in DFS. The output is a skipping probability vector, where the green arrows denote the layer skip options (skip/keep), and the blue arrows represent the quantization options. During inference, the skip/keep/quantization options corresponding to the largest vector element will be selected and to be executed.

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/Torment123/DFS.git
cd DFS
```
- Install dependencies
```bash
pip install requirements.txt
```
## Usage
- **Work flow:** pretrain the ResNet backbone  &rarr;  train gate &rarr; train DFS

**0. Data Preparation**
- `data.py` includes the data preparation for the CIFAR-10 and CIFAR-100 datasets.

**1. Pretrain the ResNet backbone**
We first train a base ResNet model in preparation for further DFS training stage.
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_base.py train cifar10_resnet_38 --dataset cifar10 --save-folder save_checkpoints/backbone
```

**2. Train gate**
We then add RNN gate to the pretrained ResNet. Fix the parameters of ResNet, only train the RNN gate to reach zero skip ratio. set minimum = 100, lr = 0.01, iters=2000

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_sp_integrate_dynamic_quantization_initial.py train cifar10_rnn_gate_38 --minimum 100 --lr 0.01  --resume save_checkpoints/backbone/model_best.pth.tar --iters 2000--save-folder save_checkpoints/full_execution
```

**3. Train DFS**
After the gate is trained to reach full execution, we then unfreeze the backbone's parameters and jointly train it with the gate for our specified skip ratio. Set minimum = _specified_ _computation_ _percentage_, lr = 0.01.
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_sp_integrate_dynamic_quantization.py train cifar10_rnn_gate_38 --minimum _specified_ _computation_ _percentage_ --lr 0.01 --resume save_checkpoints/full_execution/checkpoint_latest.pth.tar --save-folder save_checkpoints/DFS
```

## Acknowledgement
- The sequential formulation of dynamic inference problem from [SkipNet](https://github.com/ucbdrive/skipnet)
- The quantization function from [Scalable Methods](https://github.com/eladhoffer/quantized.pytorch)

## License
[MIT](https://choosealicense.com/licenses/mit/)






