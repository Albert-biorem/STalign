#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:33:03 2024

@author: albert
"""

#Import dependencies

import numpy as np
#%matplotlib notebook
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd # for csv.
from matplotlib import cm
from matplotlib.lines import Line2D
import os
from os.path import exists,split,join,splitext
from os import makedirs
import glob
import requests
from collections import defaultdict
import nrrd
import torch
from torch.nn.functional import grid_sample
import tornado
import copy
import skimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
## OPTION B: skip cell if installed STalign with pip or pipenv
import sys
sys.path.append("../../STalign") 

## import STalign from upper directory
import STalign
#Load file

filename = '__'
df = pd.read_csv('../merfish_data/s1r1_metadata.csv.gz')

#Load x position
x = np.array(df['center_x']) #change to x positions of cells

#Load y position
y = np.array(df['center_y']) #change to column y positions of cells
dx=10
blur = 1
#Rasterize Image
X_,Y_,W_ = STalign.rasterize(x,y,dx=dx, blur = blur,draw=False)
W = W_[0]
imagefile="/home/albert/Documentos/GitHub/STalign/docs/notebooks/aba_nissl.nrrd"
labelfile="/home/albert/Documentos/GitHub/STalign/docs/notebooks/aba_annotation.nrrd"
slice = 178
vol,hdr = nrrd.read(imagefile)
A = vol    
vol,hdr = nrrd.read(labelfile)
L = vol

dxA = np.diag(hdr['space directions'])
nxA = A.shape
xA = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nxA,dxA)]
XA = np.meshgrid(*xA,indexing='ij')     
from scipy.ndimage import rotate

theta_deg = -5

# fig,ax = plt.subplots(1,2)
# extentA = STalign.extent_from_x(xA[1:])
# ax[0].imshow(rotate(A[slice], angle=theta_deg),extent=extentA,interpolation='none')
# ax[0].set_title('Atlas Slice')

# ax[1].imshow(W,extent=extentA,interpolation='none')
# ax[1].set_title('Target Image')
# fig.savefig('_image.png', dpi = 1200)
# #fig.show()
# fig.canvas.draw()  
points_atlas = np.array([[0,2580]])
points_target = np.array([[8,2533]])
Li,Ti = STalign.L_T_from_points(points_atlas,points_target)
xJ = [Y_,X_]
J = W[None]/np.mean(np.abs(W))
xI = xA
I = A[None] / np.mean(np.abs(A),keepdims=True)
I = np.concatenate((I,(I-np.mean(I))**2))
sigmaA = 2 #standard deviation of artifact intensities
sigmaB = 2 #standard deviation of background intensities
sigmaM = 2 #standard deviation of matching tissue intenities
muA = torch.tensor([3,3,3],device='cpu') #average of artifact intensities
muB = torch.tensor([0,0,0],device='cpu') #average of background intensities
# initialize variables
scale_x = 0.9 #default = 0.9
scale_y = 0.9 #default = 0.9A
scale_z = 0.9 #default = 0.9
theta0 = (np.pi/180)*theta_deg

# get an initial guess
if 'Ti' in locals():
    T = np.array([-xI[0][slice],np.mean(xJ[0])-(Ti[0]*scale_y),np.mean(xJ[1])-(Ti[1]*scale_x)])
else:
    T = np.array([-xI[0][slice],np.mean(xJ[0]),np.mean(xJ[1])])

scale_atlas = np.array([[scale_z,0,0],
                        [0,scale_x,0],
                        [0,0,scale_y]])
L = np.array([[1.0,0.0,0.0],
             [0.0,np.cos(theta0),-np.sin(theta0)],
              [0.0,np.sin(theta0),np.cos(theta0)]])
L = np.matmul(L,scale_atlas)#np.identity(3)


# run LDDMM
# specify device (default device for STalign.LDDMM is cpu)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
    
#returns mat = affine transform, v = velocity, xv = pixel locations of velocity points
transform = STalign.LDDMM_3D_to_slice(
    xI,I,xJ,J,
    T=T,L=L,
    nt=4,niter=100,
    device='cpu',
    sigmaA = sigmaA, #standard deviation of artifact intensities
    sigmaB = sigmaB, #standard deviation of background intensities
    sigmaM = sigmaM, #standard deviation of matching tissue intenities
    muA = muA, #average of artifact intensities
    muB = muB #average of background intensities
)

