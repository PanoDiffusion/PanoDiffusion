import torch
import random
import math
import numpy as np




def generate_nfov_mask(imsize):
    # this function generates mask of nfov images of multiple cams on the ERP image grid using pitch angle.
    # in current implementation, the yaw angles between the cameras are set to have the same angular intervals.
    # this function takes approximately 1ms using torch cuda tensor operations.
    
    hor_fov = float(random.randint(60,90))
    ver_fov = float(random.randint(60,90))
    pitch = float(random.randint(-90, 90))
    n_cam = random.randint(1,int(max(min((90/(abs(pitch)+1e-9)),4),1))) # n_cam is adaptive to pitch angle (if pitch=90, n_cam is always 1.)
    #print(f'hor_fov:{hor_fov}, ver_fov:{ver_fov}, pitch:{pitch}, n_cam:{n_cam}')
    flip = random.random()

    assert hor_fov >=0 and ver_fov >= 0 and hor_fov <= 180 and ver_fov <= 180, "fov of nvof image should be in [0, 180]."
    assert -90 <= pitch <= 90, "pitch angle should lie in [-90,90]."

    x = np.linspace(-math.pi , math.pi , 2*imsize)
    y = np.linspace(math.pi/2 , -math.pi/2 , imsize)
    theta, phi = np.meshgrid(x, y)
    theta = torch.tensor(theta).cuda()
    phi = torch.tensor(phi).cuda()

    # get the position along the principal axis(z-axis) on the unit sphere, in erp image coordinate.
    z_erp = torch.cos(phi) * torch.cos(theta)
    x_erp = torch.cos(phi) * torch.sin(theta)
    y_erp = torch.sin(phi)

    pitch = pitch * math.pi / 180
    yaw = math.pi * 2 / n_cam

    # max x and y value in the NFoV image plane.  
    max_y = math.tan( (ver_fov+2*pitch)*math.pi/(2*180))
    max_x = math.tan( hor_fov*math.pi/(2*180))
    min_y = math.tan( (-ver_fov+2*pitch)*math.pi/(2*180))
    min_x = math.tan( -hor_fov*math.pi/(2*180))

    # tilt the coordinate along x-axis using the pitch angle to get the nfov image coordinate.
    x_nfov = x_erp
    z_nfov = z_erp*math.cos(pitch) + y_erp*math.sin(pitch)
    y_nfov = -z_erp*math.sin(pitch) + y_erp*math.cos(pitch)

    mask = torch.zeros_like(phi) # initializing mask var.

    for cam in range(n_cam):
        # tilt the coordinate along y-axis using the pitch angle to get the nfov image coordinate.
        tilt_angle = yaw * cam

        y_nfov_ = y_nfov
        z_nfov_ = z_nfov*math.cos(tilt_angle) + x_nfov*math.sin(tilt_angle)
        x_nfov_ = -z_nfov*math.sin(tilt_angle) + x_nfov*math.cos(tilt_angle)

        # shoot the ray until z_nfov=1. We set the NFoV image plane is z_nfov=1 plane. (in other words, z axis is the principal axis)
        y = y_nfov_ / (z_nfov_ + 1e-9)
        x = x_nfov_ / (z_nfov_ + 1e-9)
        z = z_nfov_
        
        # masking the values that have appropriate FoV with positive z_nfov position.
        mask += ((y > min_y) * (y < max_y) * (x > min_x) * (x < max_x) * (z >= 0)).double()

    if flip < 1/2:
        mask = torch.flip(mask, [1])
    mask = mask > 0
    
    return torch.tensor(mask).float()


