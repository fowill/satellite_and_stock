# Publicly available satellite imagery of container ports reveals the global stock markets predictability

This codebase accompanies the paper ***Publicly available satellite imagery of container ports reveals the global stock markets predictability***. It includes code to fully reproduce the core findings of the study. It also includes code to reproduce all the paper's figures.  

![未找到图片：https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure1.png](https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure1.png "未找到图片：https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure1.png")
![未找到图片：https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure2.png](https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure2.png "未找到图片：https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure2.png")
![figure3](https://github.com/satellite-and-stock-return/satellite_and_stock/blob/master/imgs/figure3.png "figure3")

## Requirements

- PyTorch == 1.10.0

- torchvision == 0.11.1

- albumentations == 1.1.0

- tqdm == 4.62.3

- opencv-python == 4.5.4.60



## Steps to plot paper figures using the paper's results & forecasts

Not finished yet



## Steps to training & using the paper's models

### 1) preliminary setup

- We use Python for deep-learning-related works.
- For Python package management, we use conda. If you don't yet have a conda, you can download it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Use conda to install the required packages.
- Training our deep learning models requires Nvidia GPUs and CUDA ([CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)) support. 

### 2) Train U-Net with training set provided by us

- Check all the parameters in `configs.py` 
- Check or modify the U-Net architecture in `U_Net_model.py`
- Check the training set in `data/training`, you can adjust training/testing/validation sets whatever you want.
- Run `train.py` to train U-Net models, the script will train in k-fold (k defaults to 10, modifiable in `configs.py`) cross-validation for a set of models with different architecture, in order to find the best-fitted model. The models & evaluations  would be saved in `data/training/epoch_models `and `data/training/training_results`

### 3) Use U-Net to identify the number of containers from satellite imagery of container ports.

- Check all the parameters in `configs.py`, parameter`CHECKPOINT_PATH`conducts the path of the model you use. `PREDICT_DATA_PATH` define the satellites image directory, the naming rules of satellites image must follow that in `data/predicting/predict_data`.
- Run `predict.py` to identify container numbers from new satellite imagery of 48 main harbors.
- Predicted results can be found at `PREDICTED_IMG_PATH` and`PREDICTED_PREDS_PATH`.
- Note that you can simply conduct training and predicting with one click by running `main.py`.



