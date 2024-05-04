# SPMFormer
A novel physical model-based transformer framework for underwater image enhancement.

This repository includes code for the following paper:

SPMFormer: Simplified Physical Model-based Transformer for Underwater Image Enhancement

# Training Environment
We train and test the code on PyTorch 1.10.2 + CUDA 11.3 + cuDNN 8.2.0.

1. Create a new conda environment
```
conda create -n SPMFormer python=3.7
conda activate SPMFormer
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Insturcions
Please execute the following instructions to configure the parameters for running the program:

1. Model Training
python train.py --model (model name) --dataset (dataset name) --exp (exp name)

2. Model Testing
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
