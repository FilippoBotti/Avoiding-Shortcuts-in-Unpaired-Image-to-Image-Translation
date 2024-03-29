# Avoiding Shortcuts in Unpaired Image-to-Image Translation

This repository contains the PyTorch code for our ICIAP 2022 paper [“Avoiding Shortcuts in Unpaired Image-to-Image
Translation”](https://link.springer.com/chapter/10.1007/978-3-031-06427-2_39). 
<br>This code is based on the PyTorch implementation of CycleGAN provided by [Jun-Yan Zhu](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and on the PyTorch implementation of GradCam provided by [Jacob Gildenblat](https://github.com/jacobgil/pytorch-grad-cam).

## Brief intro

Our architecture introduces an additional constraint during the training phase of an unpaired image-to-image
translation network; this forces the model to have the same attention
both when applying the target domains and when reversing the translation. This attention is calculated with GradCam on the last residual block.

Our model architecture is defined as depicted below, please refer to the paper for more details: 
<img src='imgs/image8.png' width="900px"/>

## Mapping results

### Horse-to-Zebra image translation results: 
<img src='imgs/horse2zebra-compared.png' width="900px"/>


### Apple-to-Orange image translation results: 
<img src='imgs/apple2orange-compared.png' width="900px"/>


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/FilippoBotti/Avoiding-Shortcuts-in-Unpaired-Image-to-Image-Translation
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies.
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. horse2zebra):
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```
- Train a model:
```
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan
```
- Test the model:
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra --model test --no_dropout 
```

### Generate the results
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra --model test --no_dropout 
```








## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{10.1007/978-3-031-06427-2_39,
author="Fontanini, Tomaso
and Botti, Filippo
and Bertozzi, Massimo
and Prati, Andrea",
editor="Sclaroff, Stan
and Distante, Cosimo
and Leo, Marco
and Farinella, Giovanni M.
and Tombari, Federico",
title="Avoiding Shortcuts in Unpaired Image-to-Image Translation",
booktitle="Image Analysis and Processing -- ICIAP 2022",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="463--475",
abstract="Image-to-image translation is a very popular task in deep learning. In particular, one of the most effective and popular approach to solve it, when a paired dataset of examples is not available, is to use a cycle consistency loss. This means forcing an inverse mapping in order to reverse the output of the network back to the source domain and reduce the space of all the possible mappings. Nevertheless, the network could learn to take shortcuts and softly apply the target domain in order to make the reverse translation easier therefore producing unsatisfactory results. For this reason, in this paper an additional constraint is introduced during the training phase of an unpaired image-to-image translation network; this forces the model to have the same attention both when applying the target domains and when reversing the translation. This approach has been tested on different datasets showing a consistent improvement over the generated results.",
isbn="978-3-031-06427-2"
}
```


## Related Projects
**[Jun-Yan Zhu](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)**<br>
**[Jacob Gildenblat](https://github.com/jacobgil/pytorch-grad-cam)**



## Acknowledgments
Our code is inspired by [Jun-Yan Zhu](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and by [Jacob Gildenblat](https://github.com/jacobgil/pytorch-grad-cam).
