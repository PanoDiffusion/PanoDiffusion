import numpy as np
import torch

def encode_bbox(shape, cord):
    token = []
    # for 64 * 128 image, the encode result should be 7 + 8 = 15 bit 
    if shape == 64: 
        token_1 = int(str(bin(cord[0])).replace("b", "").zfill(7) + str(bin(cord[1])).replace("b", "").zfill(8))
        token_2 = int(str(bin(cord[0] + cord[2])).replace("b", "").zfill(7) + str(bin(cord[1] + cord[3])).replace("b", "").zfill(8))
        token_3 = cord[4]
        token = [token_1, token_2, token_3]
        return token



if __name__ == '__main__':
    print(encode_bbox(64, [0, 0, 107, 25, 30]))

        