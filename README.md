# PanoDiffusion

This repo is the code for ICLR 2024 submission 821 - PanoDiffusion: 360-degree Panorama Outpainting via Diffusion.

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="95%">
  </a>
</p>

## Setup

### Installation
This code has been tested using python 3.8.5 with torch 1.7.0 & CUDA 11.0 on a V100.
You need to first download the code and our [pretrained model](https://drive.google.com/file/d/1fUL7NL7_iBKHb5x4_aLHvIEjMllN6Xmd/view?usp=drive_link). It should include checkpoints for RGB/Depth VQ model, LDM and RefineNet model.

```
git clone https://github.com/PanoDiffusion/PanoDiffusion.git
cd PanoDiffusion
conda env create -f environment.yml
```


### Play with PanoDiffusion

We have already prepared some images and masks under 'example' folder. To test the model, you can simply run:
```
python inference.py \
--indir PanoDiffusion/example \
--outdir PanoDiffusion/example/output \
--ckpt PanoDiffusion/pretrain_model/ldm/ldm.ckpt \
--config PanoDiffusion/config/outpainting.yaml \
--refinenet_ckpt PanoDiffusion/pretrain_model/refinenet/refinenet.pth.tar

or 

bash inference.sh
```
The results will be saved in the 'output' folder. Each time you run the code you will get a new outpainting result.
