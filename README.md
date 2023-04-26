# Fully Contextual Network for Hyperspectral Scene Parsing (TGRS 2021)

## Di Wang, Bo Du, and Liangpei Zhang

## Update 2023.04: FullyContNet won the Highly Cited Paper.

![](https://github.com/DotWang/FullyContNet/blob/main/highly_cited.png)

### Pytorch implementation of our [paper](https://ieeexplore.ieee.org/document/9347487) for image-level hyperspectral image classification.

<figure>
<img src=Figs/network.png>
<figcaption align = "center"><b>Fig.1 - The proposed FullyContNet. </b></figcaption>
</figure>

&emsp;

<figure>
<img src=Figs/module.png>
<figcaption align = "center"><b>Fig.2 - The FCMs. </b></figcaption>
</figure>

&emsp;

<figure>
<img src=Figs/detailed_module.png>
<figcaption align = "center"><b>Fig.3 - Different schemes of the FCM. </b></figcaption>
</figure>


## Usage
1. Install Pytorch 1.x (>1.0) with Python 3.5.
2. Clone this repo.
```
git clone https://github.com/DotWang/FullyContNet.git
```
3. Training, evaluation and prediction with ***trainval.py*** :

For example, if the users use Pyramid-FCM with P-C-S scheme and training on [Indian Pines dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

```
CUDA_VISIBLE_DEVICES=0 python -u trainval.py \
    --dataset 'indian' --network 'FContNet' \
    --norm 'std' \
    --input_mode 'whole' \
    --experiment-num 1 --lr 1e-2 \
    --epochs 1000 --batch-size 1 \
    --val-batch-size 1 \
    --head 'psp' --mode 'p_c_s' \
    --use_apex 'True'
```
Then the evalution accuracies, the trained models and the classification map are separately saved.
## Note
- Supporting fine-tune, where the users should specify the path of resume.
- Supporting mixed-precision training with the help of APEX. However, if you use **Salinas dataset**, please set `use_apex=False`, or it will cause the error.
- In our experiments, we directly adopt the whole image and training on the 16G NVIDIA Tesla V100 GPU. However, it is difficulty on the GPU that with smaller memory, especially for the **Houston dataset**. Thus, the sliding window training using *partial image* is also realized in the codes, where the users can freely configure the size of input patches and overlapping areas. However, the accuracies may be affected.

## Paper and Citation

If this repo is useful for your research, please cite our [paper](https://ieeexplore.ieee.org/document/9347487).

```
@ARTICLE{2021FullyContNet,
  author={Wang, Di and Du, Bo and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Fully Contextual Network for Hyperspectral Scene Parsing}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2021.3050491}}
```

## Thanks
[PSPNet](https://github.com/hszhao/semseg) &ensp; [Deeplab](https://github.com/jfzhang95/pytorch-deeplab-xception) &ensp; [DANet](https://github.com/junfu1115/DANet) &ensp;[CCNet](https://github.com/speedinghzl/CCNet) &ensp; [CCNet-Pure-Pytorch](https://github.com/Serge-weihao/CCNet-Pure-Pytorch) &ensp; [OCNet](https://github.com/openseg-group/OCNet.pytorch)


## Relevant Projects
[1] <strong> Pixel and Patch-level Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification, IEEE TGRS, 2020 | [Paper](https://ieeexplore.ieee.org/document/9121743/) | [Github](https://github.com/DotWang/ASSMN)
<br> <em> &ensp; &ensp;  Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang and Yonghao Xu</em>

[2] <strong> Graph Convolution based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification, IEEE TNNLS, 2023 | [Paper](https://arxiv.org/abs/2106.13952) | [Github](https://github.com/DotWang/SSGRN)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>

[3] <strong> Neural Architecture Search for Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search, IEEE TNNLS, 2023 | Paper | [Github](https://github.com/DotWang/HKNAS)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang, and Dacheng Tao</em>

[4] <strong> ImageNet Pretraining and Transformer based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification, IEEE TIP, 2023 | [Paper](https://arxiv.org/abs/2304.09915) | [Github](https://github.com/DotWang/DCN-T)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Jing Zhang, Bo Du, Liangpei Zhang, and Dacheng Tao</em>


