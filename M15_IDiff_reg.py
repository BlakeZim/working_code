import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCA.Core as ca
import numpy as np
import PyCA.Common as common
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
from PyCAApps import IDiff
from PyCAApps import AffineReg
from PyCAApps import ElastReg
plt.ion()   
plt.close('all')
import pickle
import glob
import sys
import gc
import json
import os


memT = ca.MEM_DEVICE

inDir = os.path.expanduser('~/korenbergNAS/3D_database/Working/Blockface/T2_registered/Thin_plate_spline/M15/')
mrDir = os.path.expanduser('/home/sci/blakez/M15-01/Data/MRI/') 
outDir = os.path.expanduser('~/korenbergNAS/3D_database/Working/Blockface/self_registered/M15/')


MRI = cc.LoadMHA(mrDir + 'HARVEY_OSAMA_TOBLAKE/T2_MRI.mha', memT)
BFI_VE = cc.LoadMHA(inDir + 'M15_01_to_MRI_TPS_bw_256_VE.mha', memT)
mask = cc.LoadMHA(mrDir + 'M15_01_MRI_Full_Mask.mha', memT)

MRI /= ca.Max(MRI)
MRI_VE = MRI.copy()
ca.Copy(MRI_VE,MRI)
cc.SetRegionLTE(MRI_VE, MRI, 0.04, 0.4)
MRI_VE *= -1
cc.VarianceEqualize_I(MRI_VE, sigma=5)
MRI_VE *= mask

# Blur the blocks to improve registration
gausFilt = ca.GaussianFilterGPU()
gausFilt.updateParams(MRI.size(), ca.Vec3Df(5,5,5),ca.Vec3Di(1,1,1))
temp = ca.Image3D(MRI.grid(), MRI.memType())
MRI_blur = MRI_VE.copy()
BFI_blur = BFI_VE.copy()
gausFilt.filter(MRI_blur, MRI_VE, temp)
gausFilt.filter(BFI_blur, BFI_VE, temp)

dispSlice = 128
fp = [1,1,0.001]

[BFI_el, h, energy] = ElastReg(BFI_blur, MRI_blur, step=0.008, sigma=0.05, nIters=500, fluidParams=fp, plot=False, verbose=1)

# diff = BFI_def - MRI_VE
# cd.Disp3Pane(diff,rng=[-3,3],sliceIdx=[128,dispSlice,128])
# cd.EnergyPlot(energy, legend=['Reg','Data','Total'])
# cd.DispHGrid(h,sliceIdx=dispSlice)





nIter = 300
sigma = 0.2
eps = 0.001

[BFI_id, phi, energy] = IDiff(BFI_el, MRI_blur, eps, sigma, nIter, plot=False, verbose=1)

diff = BFI_id - MRI_VE
cd.Disp3Pane(diff,rng=[-3,3],sliceIdx=[128,dispSlice,128])
cd.EnergyPlot(energy, legend=['Reg','Data','Total'])
cd.DispHGrid(phi,sliceIdx=dispSlice)

BFI_def = BFI_el.copy()
sys.exit()
#ca.ComposeHH(
ca.ApplyH(BFI_def, BFI_VE, h)
