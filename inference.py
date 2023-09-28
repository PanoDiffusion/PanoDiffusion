import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from utils import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import cv2
import albumentations
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from torchvision.utils import make_grid
import imageio
from refinenet.network import SemGenerator
from refinenet.refinenet_config import refinenet_config

image_rescaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)
mask_rescaler = albumentations.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_NEAREST)

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image_rescaler(image=image)["image"]
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L")).astype(np.float32)

    rescaled_mask = mask_rescaler(image=mask)["image"]
    rescaled_mask = rescaled_mask[None,None]
    rescaled_mask = torch.from_numpy(rescaled_mask)
    
    rescaled_mask = torch.cat([rescaled_mask, rescaled_mask, rescaled_mask, torch.zeros(rescaled_mask.shape)], dim=1)

    # depth is set as 0
    depth = torch.from_numpy(np.zeros((1,1,256,512)).astype(np.float32))

    batch = {"image": image, "rescaled_mask": rescaled_mask, "depth": depth}
    for k in batch:
        batch[k] = batch[k].to(device=device)
    batch["image"] = batch["image"]*2.0-1.0
    batch["depth"] = batch["depth"]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="config file to use",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="?",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--refinenet_ckpt",
        type=str,
        nargs="?",
        help="path to refinenet checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="?",
        default="rgb", # rgb or depth
        help="outpainting mode, rgb or depth",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir,"mask", "*.png")))
    images = sorted(glob.glob(os.path.join(opt.indir,"rgb", "*.png")))
    
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.ckpt)["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # refinenet
    refinenet = SemGenerator(refinenet_config)
    refinenet_ckpt = torch.load(opt.refinenet_ckpt)
    refinenet.load_state_dict(refinenet_ckpt['state_dict'])
    refinenet = refinenet.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                rgb = batch["image"]
                d = batch["depth"]
                encoder_posterior_1, encoder_posterior_2 = model.encode_first_stage(rgb, d)
                z1 = model.get_first_stage_encoding(encoder_posterior_1)
                z2 = model.get_first_stage_encoding(encoder_posterior_2)
                z = torch.cat([z1, z2], dim=1)
                cc = batch["rescaled_mask"]
                shape = (4,)+z.shape[2:]

                samples_ddim, intermediates = sampler.sample(S=200,
                                                 conditioning=None,
                                                 batch_size=cc.shape[0],
                                                 shape=shape,
                                                 x0=z,
                                                 mask=cc,
                                                 eta=1.0,
                                                 verbose=False,
                                                 rotate=True)


                
                rgb_samples_ddim, _ = model.decode_first_stage(samples_ddim)

                predicted_rgb = torch.clamp((rgb_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                
                predicted_rgb = cv2.resize(predicted_rgb.cpu().numpy().transpose(0,2,3,1)[0] * 255.0, (1024, 512))
                predicted_rgb = predicted_rgb / 255.0 * 2.0 - 1.0
                predicted_rgb = torch.tensor(predicted_rgb).transpose(2,1).transpose(1,0).unsqueeze(0).to(device)
                predicted_rgb_sr = refinenet(predicted_rgb)

                predicted_rgb_sr = predicted_rgb_sr.cpu().squeeze().transpose(0,1).transpose(2,1).numpy()
                predicted_rgb_sr = (255*(predicted_rgb_sr+1)/2).astype(np.uint8)

                Image.fromarray(predicted_rgb_sr.astype(np.uint8)).save(outpath)
