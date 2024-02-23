<p align="center">

  <h1 align="center">PanoDiffusion: 360-degree Panorama Outpainting via Diffusion</h1>
  <p align="center">
    <a href="https://sm0kywu.github.io/CV/CV.html">Tianhao Wu</a>
    ·
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    ·
    <a href="https://personal.ntu.edu.sg/astjcham/index.html">Tat-Jen Cham</a>

  </p>
  <h3 align="center">ICLR 2024</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2307.03177">Paper</a> | <a href="https://sm0kywu.github.io/panodiffusion/">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="95%">
  </a>
</p>

## Setup

### Installation
This code has been tested using python 3.8.5 with torch 1.7.0 & CUDA 11.0 on a V100.
You need to first download the code and our [pretrained model](https://drive.google.com/file/d/1xSL_Qr7VYQRItxPYLw0C7qdcRUr2bhdq/view?usp=drive_link). It should include checkpoints for RGB/Depth VQ model, LDM and RefineNet model.

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

# Citation
If you find our code or paper useful, please cite our work.
```BibTeX
@inproceedings{wu2023panodiffusion,
  title={PanoDiffusion: 360-degree Panorama Outpainting via Diffusion},
  author={Wu, Tianhao and Zheng, Chuanxia and Cham, Tat-Jen},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
