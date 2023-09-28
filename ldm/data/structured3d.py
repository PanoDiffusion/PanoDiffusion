import os
import numpy as np
import cv2
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset
from functools import partial
from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
import random


class Structured3D(Dataset):
    def __init__(self,
                 data_csv, data_root,
                 mask_csv, mask_root,
                 cond_csv, cond_root,
                 depth_csv, depth_root,
                 pred_csv, pred_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 no_crop=False,
                 no_rescale=False
                 ):
        # self.n_labels = n_labels
        self.data_csv = data_csv
        self.data_root = data_root
        self.mask_csv = mask_csv
        self.mask_root = mask_root
        self.cond_csv = cond_csv
        self.cond_root = cond_root
        self.depth_csv = depth_csv
        self.depth_root = depth_root
        self.pred_csv = pred_csv
        self.pred_root = pred_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        with open(self.mask_csv, "r") as f_1:
            self.mask_paths = f_1.read().splitlines()
        with open(self.cond_csv, "r") as f_2:
            self.cond_paths = f_2.read().splitlines()
        with open(self.depth_csv, "r") as f_3:
            self.depth_paths = f_3.read().splitlines()
        with open(self.pred_csv, "r") as f_4:
            self.pred_paths = f_4.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            # "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "mask_path_": [os.path.join(self.mask_root, l)
                           for l in self.mask_paths],
            "cond_path_": [os.path.join(self.cond_root, l)
                           for l in self.cond_paths],
            "depth_path_": [os.path.join(self.depth_root, l)
                           for l in self.depth_paths],
            "pred_path_": [os.path.join(self.pred_root, l)
                           for l in self.pred_paths]
        }
        # self.coord = coord
        # if self.coord:
        #     print("Cylinderical coordinate for 360 image.")

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=self.interpolation)
            self.image_rescaler_256 = albumentations.SmallestMaxSize(max_size=256, # Sun360 images are 256x512
                                                                 interpolation=self.interpolation)                                                     
            self.mask_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=cv2.INTER_NEAREST)
            self.cond_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=cv2.INTER_NEAREST)
            self.depth_rescaler = albumentations.SmallestMaxSize(max_size=self.size, # Sun360 images are 256x512
                                                                 interpolation=cv2.INTER_NEAREST)
            self.lr_image_rescaler = albumentations.SmallestMaxSize(max_size=self.size/4, # Sun360 images are 256x512
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)

            self.preprocessor = self.cropper
        self.no_crop = no_crop
        self.no_rescale = no_rescale
        # self.degradation_process = albumentations.SmallestMaxSize(max_size=int(self.size/4), interpolation=cv2.INTER_AREA)
        self.degradation_process = partial(degradation_fn_bsr_light, sf=4)

    def __len__(self):
        return self._length

    def rotation_augmentation(self, input_z, degree):
        new_z = np.ones_like(input_z)
        new_z[:,degree:,:] = input_z[:,:(self.size * 2 - degree),:]
        new_z[:,:degree,:] = input_z[:,(self.size * 2 - degree):,:]
        # input_z[:,:,:,degree:] = input_z[:,:,:,:(self.size * 2 - degree)]
        # input_z[:,:,:,:degree] = input_z[:,:,:,(self.size * 2 - degree):]
        return new_z

    # def rotation_augmentation_lr(self, input_z, degree):
    #     new_z = np.ones_like(input_z)
    #     new_z[:,degree:,:] = input_z[:,:(self.size * 2 - degree),:]
    #     new_z[:,:degree,:] = input_z[:,(self.size * 2 - degree):,:]
    #     # input_z[:,:,:,degree:] = input_z[:,:,:,:(self.size * 2 - degree)]
    #     # input_z[:,:,:,:degree] = input_z[:,:,:,(self.size * 2 - degree):]
    #     return new_z

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        rand_degree = random.randint(1, self.size * 2 - 1)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        
        if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
            # image = self.image_rescaler_256(image=image)["image"]
            # print(image.shape,"11111111")
            image = self.image_rescaler(image=image)["image"]
            # print(image.shape,"222222222")
            # lr_image = self.lr_image_rescaler(image=image)["image"]
            # print(lr_image.shape,"333333333333")
            # exit(0)

        image = self.rotation_augmentation(image, rand_degree)
        # lr_image = self.rotation_augmentation(lr_image, rand_degree)
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        # example["LR_image"] = (lr_image/127.5 - 1.0).astype(np.float32)

        # pred = Image.open(example["pred_path_"])
        # if not pred.mode == "RGB":
        #     pred = pred.convert("RGB")
        # pred = np.array(pred).astype(np.uint8)
        # if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
        #     pred = self.image_rescaler(image=pred)["image"]
        #     # lr_image = self.lr_image_rescaler(image=image)["image"]

        # pred = self.rotation_augmentation(pred, rand_degree)
        # example["pred"] = (pred/127.5 - 1.0).astype(np.float32)

        depth = Image.open(example["depth_path_"])
        depth = np.array(depth)
        depth = np.expand_dims(depth, axis=2)
        if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
            depth = self.depth_rescaler(image=depth)["image"]
        
        depth = self.rotation_augmentation(depth, rand_degree)
        
        # random add 0 to depth
        # p_noise = random.random()
        mask = np.random.choice([False, True], size=depth.shape, p=[0.1, 0.9])
        depth[mask] = 0
        # cv2.imwrite(os.path.join("/mnt/lustre/thwu/inpainting/latent-diffusion/test_out", str(i) + "_depth.png"), depth / (2 ** 16 - 1) * 255)
        # exit(0)
        

        # depth[depth > 0.85 * (2**16-1)] = 0
        # example["depth"] = (depth / 7000).astype(np.float32)
        # example["depth"] = example["depth"] * 2.0 - 1.0
        
        # example["depth"] = np.clip((depth / 7000), 0, 1).astype(np.float32)
        example["depth"] = (depth / (2**16 - 1)).astype(np.float32)
        example["depth"] = example["depth"] * 2.0 - 1.0

        # example["depth"] = (depth / (2**16-1) * 93.62).astype(np.float32)
        # print(np.max((depth / 7000)), np.min((depth / 7000)))
        # exit(0)

        # depth[depth > 0.85 * (2**16-1)] = 0
        # example["depth"] = (depth / 7000).astype(np.float32)
        # example["image"] = example["depth"] * 2.0 - 1.0

        # example["image"] = np.clip((depth / 7000), 0, 1).astype(np.float32)
        # # depth = np.concatenate([depth, depth, depth],axis=2)
        # # print(np.max(depth), np.min(depth))
        # # exit(0)
        # example["image"] = (depth / (2**16-1) * 93.62).astype(np.float32)
        # example["depth"] = ((depth - depth.min())/max(1e-8, depth.max()-depth.min()) * 2.0 - 1.0).astype(np.float32)
        # example["depth"] = ((depth - depth.min())/max(1e-8, depth.max()-depth.min()) * 2.0 - 1.0).astype(np.float32)
        # example["image"] = (depth / 7000).astype(np.float32)

        # cords = self.read_cord(example["cond_path_"])

        # bbox = Image.open(example["cond_path_"])
        # if not bbox.mode == "RGB":
        #     bbox = bbox.convert("RGB")
        # bbox = np.array(bbox).astype(np.uint8)
        # if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
        #     bbox = self.cond_rescaler(image=bbox)["image"]
        # example["bbox"] = (bbox/127.5 - 1.0).astype(np.float32)

        # cv2.imwrite("/mnt/lustre/thwu/inpainting/latent-diffusion/test_output/test_mask.png", mask*255)
        # mask = np.stack([mask, mask, mask], axis=2)
        # cv2.imwrite("/mnt/lustre/thwu/inpainting/latent-diffusion/test_output/test_mask_stack.png", mask*255)



            # mask = self.mask_rescaler(image=mask)["image"]

            # depth = self.depth_rescaler(image=depth)["image"]

        # mask = Image.open(example["mask_path_"])
        # mask = np.array(mask).astype(np.uint8)
        # mask = np.expand_dims(mask, axis=2)
        # if not self.no_rescale and self.size is not None: # Default True. False when refine net training.
        #     mask = self.mask_rescaler(image=mask)["image"]
        # # masked_image, mask = self.masking(image, mask) # Make input image with holes.
        # masked_image = mask * image
        # masked_image = self.rotation_augmentation(masked_image, rand_degree)
        # mask = self.rotation_augmentation(mask, rand_degree)
        # example["concat_input"] = np.concatenate(((masked_image/127.5 - 1.0).astype(np.float32), mask.astype(np.float32)), axis=2)

        # LR_image = self.degradation_process(image=image)["image"]
        # print(LR_image.shape)
        # exit(0)
        # if False:
        #     # random scale
        #     random_h = torch.randint(256,512,(1,))
        #     image = cv2.resize(image, (random_h*2, random_h), interpolation = cv2.INTER_AREA)
        #     mask = cv2.resize(mask, (random_h*2, random_h), interpolation = cv2.INTER_NEAREST)

        # Masking here


        # if not self.coord:
        # if self.size is not None:
        #     processed = self.preprocessor(image=image)
        # else:
        #     processed = {"image": image}
        # else: # when using coord (coord=True)
        #     # Change for cylinderical coord
        #     h,w,_ = image.shape
        #     #coord = np.arange(h*w).reshape(h,w,1)/(h*w) * 2 - 1 # -1~1 # old style. arange is insufficient
        #     # coord = np.tile(np.arange(h).reshape(h,1,1), (1,w,1)) / (h-1) * 2 - 1 # -1~ 1
        #     # sin, cos
        #     # coord = self.add_cylinderical(coord) # -1 ~ 1
        #     # rotation augmentation
        #     image, masked_image, binary_mask = self.rotation_augmentation(image, masked_image, binary_mask)

        #     if not self.no_crop and self.size is not None: # self.no_crop = True when training Transformer with 1:2 images, or icip(256x512)
        #         processed = self.cropper(image=image, coord=coord, masked_image=masked_image, binary_mask=binary_mask)
        #     else:
        #         processed = {"image": image, "masked_image": masked_image, "mask": mask}

        #     example["masked_image"] = (processed["masked_image"]/127.5 - 1.0).astype(np.float32)
        #     example["mask"] = processed["mask"]


        # example["image"] = (image/127.5 - 1.0).astype(np.float32)
        # example["image"] = np.concatenate(((image/127.5 - 1.0).astype(np.float32), ((depth / (2**16-1)) * 2.0 - 1.0).astype(np.float32) ), axis=2)
        # example["image"] = ((depth / (2**16-1)) * 2.0 - 1.0).astype(np.float32)
        # example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)
        # example["coordinates_bbox"] = cords
        # example["mask"] = mask
        # example["masked_image"] = (masked_image/127.5 - 1.0).astype(np.float32)

        # example["masked_image"] = (masked_image/127.5 - 1.0).astype(np.float32)
        # masked_image = (masked_image/127.5 - 1.0).astype(np.float32)

        return example

    # def rotation_augmentation(self, im, masked_im, binary_mask):
    #     split_point = torch.randint(0, im.shape[1],(1,)) # w
    #     #split_point = im.shape[1] // 2
    #     im = np.concatenate( (im[:,split_point:,:], im[:,:split_point,:]), axis=1 )
    #     # coord = np.concatenate( (coord[:,split_point:,:], coord[:,:split_point,:]), axis=1 )
    #     masked_im = np.concatenate( (masked_im[:,split_point:,:], masked_im[:,:split_point,:]), axis=1 )
    #     binary_mask = np.concatenate( (binary_mask[:,split_point:,:], binary_mask[:,:split_point,:]), axis=1 )
    #     return im, masked_im, binary_mask
    def read_cord(self, cord_path):
        infile = open(cord_path, 'r')
        cords = []
        cnt = 0
        for line in infile:
            data_line = line.strip("\n").split()
            # cords.append([int(data_line[0]), int(int(data_line[1])/2), int(int(data_line[2])/2), int(int(data_line[3])/2), int(int(data_line[4])/2)])
            cords.append([data_line[0], data_line[1], data_line[2], data_line[3], data_line[4]])
            cnt += 1
            if cnt == 64:
                break
        if cnt < 64:
            for i in range(cnt, 64):
                cords.append([data_line[0], data_line[1], data_line[2], data_line[3], data_line[4]])
                # cords.append([int(data_line[0]), int(int(data_line[1])/2), int(int(data_line[2])/2), int(int(data_line[3])/2), int(int(data_line[4])/2)])
        infile.close()
        return np.array(cords)

    def masking(self, im, mask):
        # canvas = np.ones_like(im) * 127.5
        inverted_mask = np.zeros_like(mask)
        inverted_mask[mask == 0] = 1
        inverted_mask[mask == 1] = 0
        canvas = im * (1 - inverted_mask)
        return canvas, inverted_mask

    # def masking(self, im):
    #     h, w, c = im.shape
    #     binary_mask = np.zeros((h,w,1))
    #     # random mask position
    #     margin_h = int( (180 - torch.randint(70, 95, (1,)) ) / 360 * h )
    #     #print(margin_h, (h - margin_h))
    #     binary_mask[margin_h:(h - margin_h), int(w/4):int(w/4)*3, :] = 1.
    #     canvas = np.ones_like(im) * 127.5
    #     canvas = im * binary_mask + canvas * (1 - binary_mask)
    #     return canvas

    # def add_cylinderical(self, coord):
    #     h, w, _ = coord.shape
    #     sin_img = np.sin(np.radians(np.arange(w) / w * 360))
    #     sin_img[np.abs(sin_img) < 1e-6] = 0
    #     sin_img = np.tile(sin_img, (h,1))[:,:,np.newaxis]
    #     cos_img = np.cos(np.radians(np.arange(w) / w * 360))
    #     cos_img[np.abs(cos_img) < 1e-6] = 0
    #     cos_img = np.tile(cos_img, (h,1))[:,:,np.newaxis]
    #     return np.concatenate((coord, sin_img, cos_img), axis=2)

class Structured3DTrain(Structured3D):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False, no_crop=False, no_rescale=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_train.txt",
                         data_csv="data/structured3d/structured3d_1024_train.txt", #defalut
                         data_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/rgb", #defalut
                         mask_csv="data/structured3d/structured3d_mask_1024_train.txt", #defalut
                         mask_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/mask", #defalut
                         cond_csv="data/structured3d/structured3d_bbox_1024_train.txt", #defalut
                         cond_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/bbox", #defalut
                         depth_csv="/mnt/lustre/thwu/inpainting/latent-diffusion/data/structured3d/structured3d_depth_1024_train.txt", #defalut
                         depth_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/D", #defalut
                         pred_csv="data/structured3d/structured3d_1024_train.txt", #defalut
                         pred_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/pred/uncond_rgbd", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/train",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         no_crop=no_crop, no_rescale=no_rescale)


class Structured3DVal(Structured3D):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False, no_crop=False, no_rescale=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_val.txt",
                         data_csv="data/structured3d/structured3d_1024_val.txt", #defalut
                         data_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/rgb", #defalut
                         mask_csv="data/structured3d/structured3d_mask_1024_val.txt", #defalut
                         mask_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/mask", #defalut
                         cond_csv="data/structured3d/structured3d_bbox_1024_val.txt", #defalut
                         cond_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/bbox", #defalut
                         depth_csv="/mnt/lustre/thwu/inpainting/latent-diffusion/data/structured3d/structured3d_depth_1024_val.txt", #defalut
                         depth_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/D", #defalut
                         pred_csv="data/structured3d/structured3d_1024_val.txt", #defalut
                         pred_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/pred/uncond_rgbd", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/test",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         no_crop=no_crop, no_rescale=no_rescale)

class Structured3DTrainSmall(Structured3D):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False, no_crop=False, no_rescale=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_train.txt",
                         data_csv="data/structured3d/train_small.txt", #defalut
                         data_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/rgb", #defalut
                         mask_csv="data/structured3d/mask_train_small.txt", #defalut
                         mask_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/mask", #defalut
                         cond_csv="data/structured3d/mask_train_small.txt", #defalut
                         cond_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/train/bbox", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/train",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         no_crop=no_crop, no_rescale=no_rescale)


class Structured3DValSmall(Structured3D):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False, no_crop=False, no_rescale=False):
        super().__init__(# set data_csv
                         #data_csv="data/sun360_val.txt",
                         data_csv="data/structured3d/val_small.txt", #defalut
                         data_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/rgb", #defalut
                         mask_csv="data/structured3d/mask_val_small.txt", #defalut
                         mask_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/mask", #defalut
                         cond_csv="data/structured3d/mask_val_small.txt", #defalut
                         cond_root="/mnt/lustre/thwu/inpainting/360_outpainting/dataset/val/bbox", #defalut
                         #data_root="/home/ubuntu/tmp_local/laval1024/test",
                         # set args through a configs yaml.
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         no_crop=no_crop, no_rescale=no_rescale)