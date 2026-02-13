# Wavelet–Frequency Fusion Based Diffusion Model for Underwater Image Enhancement and Restoration

Yun Zhang,Yushen Zheng,Maowei Zeng,Ping Li,Huanwen Wang*

<hr />

> **Abstract:** *During underwater image enhancement and restoration, images often suffer from a series of degradation-induced technical challenges. However, most existing methods, when faced with these challenges, only focus on exploring the spatial domain of the original image, while neglecting to incorporate image frequencies into the features, which leads to the unnecessary loss of many image characteristics. To address this limitation, we propose a novel underwater image restoration framework termed Wavelet–Frequency Fusion Based Diffusion Model(WFBDM). WFBDM consists of two components: the Dual-domain Wavelet-Fourier linked interaction network (DWFnet) and the Domain Residual Frequency Diffusion Calibration Module (DRFDC).To achieve effective early-stage enhancement of wavelet-domain frequency features, DWFnet combines the efficient feature extraction capability of depthwise separable convolutions with dual-domain modeling, enabling in-depth exploration of underwater image frequency characteristics and laying a solid foundation for subsequent image optimization. DRFDC focuses on the frequency-domain residuals and employs a diffusion-based mechanism to precisely calibrate high-frequency texture details and low-frequency color information, thereby further improving visual quality and detail clarity in underwater images.Through the synergy of these two components, WFBDM demonstrates competitive performance in both quantitative metrics and visual quality on multiple underwater image enhancement and restoration experiments. The related code is publicly available at:https://github.com/zhengyushen233/WFBDM.* 
<hr />

## Note
The experimental configuration of this code is defined in the YAML files within the `options` directory, which include basic general settings, dataset configuration for training and validation, network architecture configuration, path configuration, training strategy settings (such as the optimizer, scheduler, and loss functions), validation configuration, logging and model checkpoint saving configuration, and distributed training configuration.In addition, DWFnet is implemented in the `dft.arch` file.

## Environment
The environment configuration file can be found in the `requirements.txt` file in the code repository.


## Datasets
We adopt the  defined training and testing splits of the UIEB and LSUI datasets, and additionally utilize the no-reference datasets U45 and C60. These datasets are widely used benchmarks in the field of underwater image restoration, facilitating a comprehensive evaluation of the model’s restoration capability, generalization performance, and visual quality under diverse degradation types and complex underwater scenarios.

## Pretrained Models
The trained model should be placed in the models directory, and the path in pretrain_network_g within test.wfbdm.yml should be updated accordingly, for example: ./basicsr/models/net_g_xxxxxx.pth.

## Training

After preparing the training data, use 
```
Change the directory to WFBDM_final, then enter the following command in the terminal:

python basicsr/train.py -opt options/train/train_wfbdm.yml

If a background script tool is used, enter:

nohup python basicsr/train.py -opt options/train/train_wfbdm.yml > train.log 2>&1 &
```


## Testing

After preparing the testing data, use 
```
Change the directory to WFBDM_final, then enter the following command in the terminal:

python basicsr/test.py -opt options/test/test_wfbdm.yml

If a background script tool is used, enter:

nohup python basicsr/test.py -opt options/test/test_wfbdm.yml > test.log 2>&1 &
```


## Contact
Should you have any questions, please contact 1017063545@qq.com
 

