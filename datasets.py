from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import random


class PartDataset(data.Dataset):
    def __init__(self, root, train = True):
        self.root = root
        self.path=os.listdir(root)
        self.maxn=100

        if train:
            self.path = self.path[:int(len(self.path) * 0.9)]
        else:
            self.path = self.path[int(len(self.path) * 0.9):]

    def __getitem__(self, index):
        pathi = self.path[index]
        f = open(os.path.join(self.root, pathi), 'r')
        sdf = list(map(float, f.readline().split()))
        sdf = np.array(sdf).reshape(self.maxn, self.maxn, self.maxn)
        out_sdf = np.zeros((1, 64,64,64))
        aug1=random.random()
        aug2=random.random()
        aug3=random.random()
        for i in range(64):
            i1=i
            if aug1<0.5:
                i1=63-i
            for j in range(64):
                j1=j
                if aug2<0.5:
                    j1=63-j
                for k in range(64):
                    k1=k
                    if aug3<0.5:
                        k1=63-k
                    out_sdf[0, i, j, k] = sdf[18+i1,18+j1,18+k1]
        #truncate
        #out_sdf=np.clip(out_sdf,-20,20)
        in_voxel = np.sign(out_sdf)

        in_voxel = torch.from_numpy(in_voxel)
        out_sdf = torch.from_numpy(out_sdf)

        in_voxel = in_voxel.float()
        out_sdf = out_sdf.float()

        return in_voxel, out_sdf

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'sdf',train=True)
    print(len(d))
    inn,out = d[0]
    print(inn.size(), inn.type(), out.size(),out.type())
