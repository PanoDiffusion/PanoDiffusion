python inference.py \
--indir ./example \
--outdir ./example/output \
--ckpt ./pretrain_model/ldm/ldm.ckpt \
--config ./config/outpainting.yaml \
--refinenet_ckpt ./pretrain_model/refinenet/refinenet.pth.tar
