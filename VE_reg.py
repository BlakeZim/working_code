import PyCA.Core as ca
import numpy as np
import sys
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import matplotlib.pyplot as plt
plt.close('all')

block = 3                      # 1, 2, 3, 4



dir_in = '/home/sci/crottman/korenberg/results/landmark/'
dir_out = '/home/sci/crottman/korenberg/results/ve_reg/'

Imri = cc.LoadMHA('/home/sci/crottman/korenberg/results/MRI/brain_seg.mha')
cc.SetRegionLT(Imri, Imri, 1.0, 20000)
VEmri = ca.Image3D(Imri.grid(), Imri.memType())
cc.VarianceEqualize(VEmri, Imri, sigma=2.0)

# # load landmark registered blocks
# Ibfi = cc.LoadMHA(dir_in + 'block' + str(block) + '_as_MRI_bw.mha')
# ca.Neg_I(Ibfi)
# VEbfi = ca.Image3D(Imri.grid(), Imri.memType())
# cc.VarianceEqualize(VEbfi, Ibfi, sigma=2.0)
# cd.DispImage(VEbfi, ca.MinMax(VEmri))

# load landmark registered pre-VEd block
VEbfi = cc.LoadMHA(dir_in + 'block' + str(block) + '_as_MRI_ve.mha')
ca.Neg_I(VEbfi)
# cc.WritePNG(VEbfi, 'VEbfi_block' + str(block) + 'new.png', rng = [-2.8, 2.8])
# cd.DispImage(Ibfi)


bfi_reg = ca.Image3D(Imri.grid(), Imri.memType())




# cd.DispImage3D(VEmri)
# cd.DispImage3D(VEbfi)
cd.DispImage(VEmri)
cd.DispImage(VEbfi)
# print ca.MinMax(VEmri)
# print ca.MinMax(VEbfi)
rng = [-2.8, 2.8]

# cc.WritePNG(VEmri, 'VEmri.png', rng=rng)
# cc.WritePNG(VEbfi, 'VEbfi_block' + str(block) + '.png', rng=rng)
# sys.exit()

from AffineReg import AffineReg

A = AffineReg(VEbfi, VEmri, maxIter=150, constraint='rigid')
np.set_printoptions(linewidth=100)
print repr(A)

# ca.Neg_I(Ibfi)
# cc.ApplyAffineReal(bfi_reg, Ibfi, A)

# cc.WriteMHA(bfi_reg, dir_out + 'block' + str(block) + '_as_MRI.mha')
