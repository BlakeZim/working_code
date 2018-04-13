import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
import PyCAApps as apps
import sys
import gc
from BFI_reg_landmarks import get_new_landmarks
import csv
# import matplotlib.pyplot as plt
# cc.SelectGPU()
plt.ion()
plt.close('all')

block = 4

MRI_to_BFI = False
BFI_to_MRI = True
# MRIsizes = [256, 512]
MRIsizes = [256]
colors = ['bw']  #, 'rgb', 'weight']
# colors = ['weight']

Write = True

if max(MRIsizes) > 256:
    mType = ca.MEM_HOST
else:
    mType = ca.MEM_HOST

MRIdir = '/home/sci/blakez/M15-01/Data/MRI/HARVEY_OSAMA_TOBLAKE/'
MRI_fname = MRIdir + 'T2_MRI.mha'
BFIdir = '/local/blakez/korenbergNAS/3D_database/Working/Blockface/self_registered/M15/'
outdir = '/local/blakez/korenbergNAS/3D_database/Working/Blockface/T2_registered/Thin_plate_spline/M15/'

# for block in [1, 2, 3, 4]:
#     fname = 'block{0}_reg_fillblanks_bw.mha'.format(block)
#     tmp = cc.LoadMHA(BFIdir + fname)
#     cc.WriteGrid(tmp.grid(), BFIdir + 'block{0}_grid.txt'.format(block))
# tmp = cc.LoadMHA(MRI_fname)
# cc.WriteGrid(tmp.grid(), MRIdir + 'T2_grid.txt'.format(block))
# sys.exit()

#BFIgrid = cc.LoadGrid(BFIdir + 'block{0}_grid.txt'.format(block))
#MRIgrid = cc.LoadGrid(MRIdir + 'T2_grid.txt'.format(block))
MRI = cc.LoadMHA(MRI_fname,ca.MEM_HOST)
MRIgrid = MRI.grid()
#MRIgrid = cc.MakeGrid(MRI.size(),MRI.spacing(),'center')
#MRI.setGrid(MRIgrid)
#BFI = cc.LoadMHA(BFIdir + 'M15_01_seg_crop_hd8.mha',ca.MEM_HOST)
BFI_full = cc.LoadMHA(BFIdir + 'M15_01_hd8_VE.mha',ca.MEM_HOST)
BFI = cc.SubVol(BFI_full, yrng=[146,626])
BFIgrid = cc.MakeGrid(ca.Vec3Di(480,480,2239),ca.Vec3Df(0.1185,0.1185,0.030),'center')
BFI.setGrid(BFIgrid)
#BFIgrid = BFI.grid()
print MRIgrid
print BFIgrid
# print 'b4 memory', 480*480*874*4/1024./1024 *7  # 7 blocks, mem size
# print 'mri memory', 256*256*256*4/1024./1024 *7  # 7 blocks, mem size
# print 'mri big memory', 512*512*512*4/1024./1024 *7  # 7 blocks, mem size
# print 'single mri ', 256*256*256*4/1024./1024

#landmarks = get_new_landmarks(block)
# Use these landmarks for an affine
### landmarks = [[[1893.0,123.0,401.0], [187.0,135.0,202.0]],
###              [[1798.0,416.0,333.0], [183.0,131.0,48.0]],
###              [[892.0,222.0,251.0], [84.0,109.0,132.0]],
###              [[2142.0,260.0,332.0], [217.0,114.0,125.0]],
###              [[1764.0,238.0,265.0], [176.0,84.0,129.0]],
###              [[53.0,184.0,198.0], [3.0,91.0,143.0]],
###              [[545.0,352.0,304.0], [57.0,149.0,74.0]],
###              [[613.0,135.0,392.0], [60.0,150.0,187.0]],
###              [[1526.0,193.0,120.0], [145.0,24.0,139.0]],
###              [[1485.0,227.0,107.0], [141.0,24.0,123.0]],
###              [[783.0,328.0,363.0], [78.0,175.0,83.0]],
###              [[1122.0,151.0,385.0], [105.0,157.0,176.0]],
###              [[2082.0,292.0,451.0], [214.0,159.0,119.0]],
###              [[1392.0,409.0,256.0], [138.0,114.0,37.0]]]

### #             [[2221.0,195.0,283.0], [225.0,91.0,103.0]],
### #common.DebugHere()

### for lm in landmarks:
###     #print lm
### #    common.DebugHere()
###     lm[0] = lm[0][::-1]
###     lm[1] = lm[1][::-1]
### #    lm[1][0:2] = lm[1][-2:-4:-1]
###     lm[0] = np.ndarray.tolist(np.multiply(lm[0],BFIgrid.spacing().tolist()) + BFIgrid.origin().tolist())
###     lm[1] = np.ndarray.tolist(np.multiply(lm[1],MRIgrid.spacing().tolist()) + MRIgrid.origin().tolist())
###     #print lm

with open(outdir + 'blockface_AFF_landmarks_index.txt', 'r') as bfp:
    bfAff=[[float(v) for v in line.split()] for line in bfp]

with open(outdir + 'mri_AFF_landmarks_index.txt', 'r') as mrp:
    mrAff=[[float(v) for v in line.split()] for line in mrp]

AFF_landmarks = [[bfAff[x],mrAff[x]] for x in range(0,np.shape(mrAff)[0])]

for lm in AFF_landmarks:
    lm[0] = np.ndarray.tolist(np.multiply(lm[0],BFIgrid.spacing().tolist()) + BFIgrid.origin().tolist())
    lm[1] = np.ndarray.tolist(np.multiply(lm[1],MRIgrid.spacing().tolist()) + MRIgrid.origin().tolist())
print np.array(AFF_landmarks)


Afw = apps.SolveAffine(AFF_landmarks)
BFI_aff = MRI.copy()
cc.ApplyAffineReal(BFI_aff, BFI, Afw)
cd.Disp3Pane(BFI_aff)

sys.exit()
# Use the resulting affine transformed block to define landmarks for TPS
with open(outdir + 'blockface_TPS_points_index.txt', 'r') as bfp:
    bfPoints=[[float(v) for v in line.split()] for line in bfp]

with open(outdir + 'mri_TPS_points_index.txt', 'r') as mrp:
    mrPoints=[[float(v) for v in line.split()] for line in mrp]

### NOT THE RIGHT WAY TO COMBINE###
TPS_landmarks = [[bfPoints[x],mrPoints[x]] for x in range(0,np.shape(mrPoints)[0])]
for lm in TPS_landmarks:
    lm[0] = np.ndarray.tolist(np.multiply(lm[0],MRIgrid.spacing().tolist()) + MRIgrid.origin().tolist())
    lm[1] = np.ndarray.tolist(np.multiply(lm[1],MRIgrid.spacing().tolist()) + MRIgrid.origin().tolist())
print np.array(TPS_landmarks)

# Test image for comparison
#BFI_test = cc.LoadMHA(BFIdir + 'block1_reg_fillblanks_filt128_bw_VE.mha',mType)
#cd.Disp3Pane(BFI_test)
#outimage = common.ExtractSliceIm(BFI_test,100)
#cd.DispImage(outimage)

if MRI_to_BFI:
    ca.ThreadMemoryManager.init(BFIgrid, mType, 6)
    spline = SolveSpline(TPS_landmarks)
    h = SplineToHField(spline, BFIgrid, mType)
    print ca.MinMax(h)
    MRIdef = ca.ManagedImage3D(BFIgrid, mType)
    #MRI = cc.LoadMHA(MRI_fname, mType)
    cc.ApplyHReal(MRIdef, MRI, h)

    # write data
    fname = 'MRI_TPS_to_block{0}.mha'.format(block)
    cd.Disp3Pane(MRIdef)
    print "Writing", outdir+fname
    if Write:
        cc.WriteMHA(MRIdef, outdir + fname)
    del MRIdef, MRI, h
    cc.MemInfo()


if BFI_to_MRI:
    TPS_landmarks = [[lmpair[1], lmpair[0]] for lmpair in TPS_landmarks]  # reverse
    spline = apps.SolveSpline(TPS_landmarks)  # only do once, b/c real coords#
    print spline
    #np.save(spline, outdir + 'block{0}_TPS_Spline_{1}.mha'.format(block,MRIsizes))
    for sz in MRIsizes:
        print "for size", sz
        # ca.ThreadMemoryManager.destroy()
        # MRIgrid = cc.MakeGrid([sz, sz, sz], [256.0/sz, 256.0/sz, 256.0/sz], 'center')
        # ca.ThreadMemoryManager.init(MRIgrid, mType, 6)
        
        h = apps.SplineToHField(spline, MRIgrid, mType)
        print ca.MinMax(h)
        for color in colors:
            "for color", color
            # cc.MemInfo()
            # if sz <= 256:
            #     BFIfname = 'block{0}_reg_fillblanks_filt256_{1}.mha'.format(block, color)
            # else:
            #     BFIfname = 'block{0}_reg_fillblanks_filt256_{1}.mha'.format(block, color)
            if color == 'weight' and sz <=512:
                BFIfname = 'block{0}_weight.mha'.format(block)
            elif color == 'weight' and sz == 1024:
                BFIfname = 'block{0}_weight_hd4.mha'.format(block)
            elif sz <= 512:
                BFIfname = 'block{0}_reg_fillblanks_filt128_{1}.mha'.format(block, color)
            elif sz == 1024:    # use unfiltered hd data
                BFIfname = 'block{0}_reg_fillblanks_{1}_hd4.mha'.format(block, color)
            # BFI = cc.LoadMHA(BFIdir + BFIfname, mType)
#            outimage = common.ExtractSliceIm(BFI,100)
            cd.Disp3Pane(BFI_aff)
            
            if color in ['bw', 've', 'weight']:
                BFIdef = ca.ManagedImage3D(MRIgrid, mType)  # these should be small enough
            else:
                BFIdef = ca.ManagedField3D(MRIgrid, mType)
            cc.ApplyHReal(BFIdef, BFI_aff, h)
            if sz == MRIsizes[-1] and color == colors[-1]:
                cd.Disp3Pane(BFIdef)

            # write data
            if Write:
                if color == 'rgb':
                    fname = 'block{0}_as_MRI_rgba_{1}.mha'.format(block, sz)
                    cc.WriteColorMHA(BFIdef, outdir + fname)
                    fname = 'block{0}_as_MRI_rgb_{1}.mha'.format(block, sz)
                    cc.WriteMHA(BFIdef, outdir + fname)
                else:
                    #fname = 'block{0}_as_MRI_{1}_{2}_NEWLANDMARKS.mha'.format(block, color, sz)
                    fname = 'M15_01_to_MRI_TPS_bw_256_VE.mha'
                    cc.WriteMHA(BFIdef, outdir + fname)
                    cc.WriteMHA(h, outdir + 'M15_01_to_MRI_TPS_def_256.mha')
                    # cc.WriteMHA(h, outdir + 'block{0}_TPS_HField_{1}.mha'.format(block,sz))
                    cd.Disp3Pane(BFIdef)
            common.DebugHere()
            del BFIdef, BFI
        del h
