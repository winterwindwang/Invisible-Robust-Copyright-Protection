# Invisible-Robust-Copyright-Protection
The official implementation of [An invisible, robust copyright protection method for DNN-generated content](https://doi.org/10.1016/j.neunet.2024.106391), which is accpeted by Neural Network.

# Dataset
In this work, we use two datasets: "summer2winter" released by the [cycleGAN](https://arxiv.org/pdf/1703.10593) and a subset of ImageNet that random sampled from ImageNet training set, and the sample index can be found in [here](https://github.com/winterwindwang/Data-efficient-UAP/blob/main/dataset/imagenet10k.txt). 

# Style Transfer
In this work, we adopt the style transfer technique proposed by [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), and we adopt the implementation provided by the pytorch [tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).

To perform style transfer, please config the data path and style image in `style_transfer/generate_transfered_images.py`, then run the following command:

```cmd
python style_transfer/generate_transfered_images.py
```

# Training
To train the network, you should first configure the data path in 'config.yaml', and then run the following command:

```cmd
python train_res256.py
```

Later, we will provide some pretrained models.

# Testing
To test the performance of the trained network, you should configure the related setting according to `checkpoint_dict` in `test_res256.py`, for example

```python
    "summer2winter": {  # the mode checkpoint for what (e.g., trained on which dataset) 
        "ckpt_path": "checkpoints/Res256_copyright_image_07-04-13-30/copyright_image_140000.pth",  # the checkpoint path
        "data_path": "F:/DataSource/StyleTransfer/summer2winter_yosemite/testA/",                   # the test image path
        "copyright_path": [  # the copyright image used during training
            'copyright_image/peking_university.png',
            'copyright_image/stanford_university.png',
            'copyright_image/Tsinghua.jpg',
            'copyright_image/ucla_university.png',
            'copyright_image/zhejiang_university.png',
            'copyright_image/UN.png',
        ]
    }
```
After that, you can run the following command to generate corresponding images
```cmd
python test_res256.py
```
Run the above command will spawn the following folder:
```text
Results
     --summer2winter
        --Test_clean      # the test image
        --Test_encoded    # the encoded image with copyright image
        --Test_decoded    # the decoded copyright image
        --Test_cover_residual
        --Test_secret_residual
        --Test_copyright_image  # the ground truth copyright image
```

# Metric Calculating
To calculate the metric, you should configure the `evaluation_directories` in `evaluate_metrics_ignite.py`, then run the following command
```cmd
python evaluate_metrics_ignite.py
```

# Citation information
```text
@article{WANG2024106391,
title = {An invisible, robust copyright protection method for DNN-generated content},
journal = {Neural Networks},
volume = {177},
pages = {106391},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106391},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024003150},
author = {Donghua Wang and Wen Yao and Tingsong Jiang and Weien Zhou and Lang Lin and Xiaoqian Chen},
}
```