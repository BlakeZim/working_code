'''For an entire block, we do a 2D registration a blockface slice to
the equivalent MRI slice (from the VE affine registration)

i.e. we take the original block#_reg_bw.mha and deform it to
MRI_as_block#.mha from ve_reg
'''

import PyCA.Core as ca
import numpy as np
import sys
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')
import PyCA.Common as common

from PyCAApps import IDiff, AffineReg, ElastReg
import time

block = 1
dir_bf = '/home/sci/blakez/monkey13/results/spline/block1_480'
#if block in [1, 2]:
dir_mri = '/home/sci/blakez/data/MRI/'
#else:
#    dir_mri = '/home/sci/crottman/korenberg/results/landmark/'
dir_save = '/home/sci/blakez/results/elast_reg/block1_480/'

SaveInMRICoords = True
MRI_sz = [256]             # list of sizes
SaveInBFICoords = False

SaveRGB = False
SaveVE = False
SaveBW = True
SaveVF = False
debug = False

# load BFI slice and MRI_as_BFI slices
BFI3D = cc.LoadMHA(dir_bf + 'block' + str(block) + 'as_MRI_bw_256.mha', ca.MEM_HOST)  # B/W
MRI3D = cc.LoadMHA(dir_mri + 'T2Seg.mha', ca.MEM_HOST)

# Initialize Deformed 3D Volumes
if SaveRGB:
    BFI_color3D = cc.LoadMHA(dir_bf + 'block' + str(block) + '_reg_rgb.mha', ca.MEM_HOST)
    BFIDef3D_RGB = ca.Field3D(BFI3D.grid(), BFI3D.memType())
    ca.SetMem(BFIDef3D_RGB, 0.0)
if SaveVE:
    BFIDef3D_VE = ca.Image3D(BFI3D.grid(), BFI3D.memType())
    ca.SetMem(BFIDef3D_VE, 0.0)
if SaveBW:
    BFIDef3D_BW = ca.Image3D(BFI3D.grid(), BFI3D.memType())
    ca.SetMem(BFIDef3D_BW, 0.0)
if SaveVF:
    BFIDef3D_VF = ca.Image3D(BFI3D.grid(), BFI3D.memType())
    ca.SetMem(BFIDef3D_VF, 0.0)

# Initialize the 2D slices
BFI = common.ExtractSliceIm(BFI3D, 0)
BFI.toType(ca.MEM_DEVICE)
BFI.setOrigin(ca.Vec3Df(0, 0, 0))
print BFI.grid()
grid2D = BFI.grid().copy()
if SaveRGB:
    BFI_def_RGB = ca.Field3D(grid2D, BFI.memType())
    ca.SetMem(BFI_def_RGB, 0.0)
if SaveVE:
    BFI_def_VE = ca.Image3D(grid2D, BFI.memType())
    ca.SetMem(BFI_def_VE, 0.0)
if SaveBW:
    BFI_def_BW = ca.Image3D(grid2D, BFI.memType())
    ca.SetMem(BFI_def_BW, 0.0)

for sliceIdx in xrange(BFI3D.grid().size().z):
    print '###### Slice {} ######'.format(sliceIdx)
    # Get 2D slice
    BFI = common.ExtractSliceIm(BFI3D, sliceIdx)
    if ca.Max(BFI) == 0.0:
        print "slice {} is blank".format(sliceIdx)
        # insert previous slice (fill in)
        if SaveRGB:
            cc.InsertSlice(BFIDef3D_RGB, BFI_def_RGB, sliceIdx)
        if SaveVE:
            cc.InsertSlice(BFIDef3D_VE, BFI_def_VE, sliceIdx)
        if SaveBW:
            cc.InsertSlice(BFIDef3D_BW, BFI_def_BW, sliceIdx)
        continue

    # Run IDiff on VE B/W Images
    MRI = common.ExtractSliceIm(MRI3D, sliceIdx)
    # Make Memory all fast
    MRI.toType(ca.MEM_DEVICE)
    BFI.toType(ca.MEM_DEVICE)
    # Standardize Grids
    MRI.setGrid(grid2D)
    BFI.setGrid(grid2D)
    MRI /= ca.Max(MRI)
    BFI_VE = ca.Image3D(grid2D, BFI.memType())
    MRI_VE = ca.Image3D(grid2D, MRI.memType())
    ca.Copy(MRI_VE, MRI)
    ca.Copy(BFI_VE, BFI)

    cc.SetRegionLTE(MRI_VE, MRI, 0.13, 1)
    MRI_VE *= -1

    square = ca.Image3D(grid2D, BFI.memType())
    cc.CreateRect(square, [0, 0], [440, 440])
    BFI_VE *= square

    cc.VarianceEqualize_I(BFI_VE, sigma=5.0)
    cc.VarianceEqualize_I(MRI_VE, sigma=5.0)

    grid_orig = BFI_VE.grid().copy()
    grid_new = cc.MakeGrid(grid_orig.size(), [1, 1, 1], 'center')
    BFI_VE.setGrid(grid_new)
    MRI_VE.setGrid(grid_new)
    BFI_VE_def = ca.Image3D(grid_new, BFI_VE.memType())

    # do rigid reg first
    # A = AffineReg(BFI_VE, MRI_VE, constraint='rigid', plot=debug)
    A = AffineReg(BFI_VE, MRI_VE, plot=debug, maxIter=400, verbose=0)[1]
    cc.ApplyAffineReal(BFI_VE_def, BFI_VE, A)
    # cd.DispImage(BFI_VE_def)
    # cd.DispImage(MRI_VE)
    # hA = ca.Field3D(BFI_VE.grid(), BFI_VE.memType())
    # cc.AtoH(hA, A)
    # cd.DispHGrid(hA, splat=False)
    # sys.exit()

    if block == 1:
        nIters = 100
        sigma = .03
        step = .00002
    elif block == 2:
        nIters = 100
        sigma = .003
        step = .00002
    elif block == 3:            # untested
        nIters = 400
        sigma = .06
        step = .002
    elif block == 4:            # untested
        nIters = 400
        sigma = .06
        step = .002

    Idef, phi0 = IDiff(BFI_VE_def, MRI_VE, step=step, sigma=sigma,
                       nIters=nIters, plot=debug, verbose=1)[:2]
    if debug:
        time.sleep(3)
        plt.close('all')

    phi = phi0.copy()
    hA = phi0.copy()
    cc.AtoH(hA, A)              # convert A to h field
    ca.ComposeHH(phi, hA, phi0)

    # convert grid back
    hA.setGrid(grid_orig)
    phi0.setGrid(grid_orig)
    phi.setGrid(grid_orig)
    Idef.setGrid(grid_orig)
    BFI_VE.setGrid(grid_orig)
    MRI_VE.setGrid(grid_orig)

    # Insert Slice back into Volume
    if SaveRGB:
        BFI_color = common.ExtractSliceVF(BFI_color3D, sliceIdx)
        BFI_color.toType(ca.MEM_DEVICE)
        BFI_color.setGrid(grid2D)

        BFI_def_RGB = ca.Field3D(grid2D, BFI.memType())
        ca.ApplyH(BFI_def_RGB, BFI_color, phi, ca.BACKGROUND_STRATEGY_PARTIAL_ZERO)
        cc.InsertSlice(BFIDef3D_RGB, BFI_def_RGB, sliceIdx)
    if SaveVE:
        BFI_def_VE = ca.Image3D(grid2D, BFI.memType())
        ca.ApplyH(BFI_def_VE, BFI_VE, phi, ca.BACKGROUND_STRATEGY_PARTIAL_ZERO)
        cc.InsertSlice(BFIDef3D_VE, BFI_def_VE, sliceIdx)
    if SaveBW:
        BFI_def_BW = ca.Image3D(grid2D, BFI.memType())
        ca.ApplyH(BFI_def_BW, BFI, phi, ca.BACKGROUND_STRATEGY_PARTIAL_ZERO)
        cc.InsertSlice(BFIDef3D_BW, BFI_def_BW, sliceIdx)

# Save BFIDef3D
if SaveInBFICoords:
    print 'saving blockface in blockface coords...'
    if SaveVE:
        cc.WriteMHA(BFIDef3D_VE, dir_save + 'BFI_block_' + str(block) + '_ve.mha')
    if SaveRGB:
        print 'Writing RGB'
        cc.WriteMHA(BFIDef3D_RGB, dir_save + 'BFI_block_' + str(block) + '_rgb.mha')
    if SaveBW:
        cc.WriteMHA(BFIDef3D_BW, dir_save + 'BFI_block_' + str(block) + '_bw.mha')


if not SaveInMRICoords:
    sys.exit()

# Save HD version of the blockface in MRI Coordinates
print 'saving blockface in blockface coords...'
mType = ca.MEM_HOST
import Affine_transforms
from ResampleAffine import ApplyAffine
if block in [3, 4]:
    A = Affine_transforms.get_landmark_transform(block)
else:
    A = np.dot(Affine_transforms.get_ve_transform(block),
               Affine_transforms.get_landmark_transform(block))

for sz in MRI_sz:
    grid = cc.MakeGrid([sz, sz, sz], 256.0/sz, origin='center')

    if SaveRGB:
        BFI_new = ca.Field3D(grid, mType)
        ApplyAffine(BFI_new, BFIDef3D_RGB, A)
        cc.WriteMHA(BFI_new, dir_save
                    + 'block' + str(block) + '_as_MRI_rgb_' + str(sz)+'.mha')

        print ca.MinMax(BFI_new)
        del BFI_new

    if SaveVE:
        BFI_new = ca.Image3D(grid, mType)
        ApplyAffine(BFI_new, BFIDef3D_VE, A)
        cc.WriteMHA(BFI_new, dir_save
                    + 'block' + str(block) + '_as_MRI_ve_' + str(sz)+'.mha')
        print ca.MinMax(BFI_new)
        del BFI_new

    if SaveBW:
        BFI_new = ca.Image3D(grid, mType)
        ApplyAffine(BFI_new, BFIDef3D_BW, A)
        cc.WriteMHA(BFI_new, dir_save
                    + 'block' + str(block) + '_as_MRI_bw_' + str(sz)+'.mha')
        print ca.MinMax(BFI_new)
        del BFI_new
