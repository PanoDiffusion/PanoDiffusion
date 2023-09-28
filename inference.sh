python /mnt/lustre/thwu/inpainting/PanoDiffusion/inference.py \
--indir PanoDiffusion/example \
--outdir PanoDiffusion/example/output \
--ckpt PanoDiffusion/pretrain_model/ldm/ldm.ckpt \
--config PanoDiffusion/config/outpainting.yaml \
--refinenet_ckpt PanoDiffusion/pretrain_model/refinenet/refinenet.pth.tar