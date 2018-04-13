import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCA.Core as ca
import numpy as np
import PyCA.Common as common
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCAApps as apps
plt.ion()
plt.close('all')
import pickle
import glob
import sys
import gc
import json
import os

memT = ca.MEM_DEVICE
M15_to_M13=True
M13dir = os.path.expanduser('~/korenbergNAS/3D_database/Working/MRI/Post-Mortem_MRI/Subject_to_Subject/M13/')
M15dir = os.path.expanduser('~/korenbergNAS/3D_database/Working/MRI/Post-Mortem_MRI/Subject_to_Subject/M15/')


with open(M13dir + 'Affine/M13_01_MRI_affineLandmarks_affineApplied_3.txt', 'r') as m13:
        m13pts=[[float(v) for v in line.split()] for line in m13]

with open(M15dir + 'Affine/M15_01_MRI_affineLandmarks_3.txt', 'r') as m15:
        m15pts=[[float(v) for v in line.split()] for line in m15]

aff_M13 = np.load(M13dir + 'Affine/rotationMatrix.npy')

M13_aff = cc.LoadMHA(M13dir + 'Affine/T2Seg_roty-119_flipy.mha', memT)
M15 = cc.LoadMHA(M15dir + 'Affine/T2_MRI.mha',memT)
M15_mask = cc.LoadNRRD(M15dir + 'Affine/M15_01_MRI_Full_Mask_lessBrainStem.nrrd', memT)
M13_mask = cc.LoadNRRD(M13dir + 'Affine/M13_01_MRI_Full_Mask_lessBrainStem.nrrd', memT)

M15_mask.setGrid(M15.grid())
M13_mask.setGrid(M13_aff.grid())

M15 *= M15_mask
M13_aff *= M13_mask


# Points in the M15 were chosen in Flipped Y coordiante system, so flip them back
# for pts in m15pts:
#     pts[0]= 256-pts[0]
#     pts[1]= 256-pts[1]

landmarks = [[m13pts[x],m15pts[x]] for x in range(0,np.shape(m13pts)[0])]

# Convert to real coordinates
for lm in landmarks:
    lm[0] = np.ndarray.tolist(np.multiply(lm[0],M13_aff.spacing().tolist()) + M13_aff.origin().tolist())
    lm[1] = np.ndarray.tolist(np.multiply(lm[1],M15.spacing().tolist()) + M15.origin().tolist())

#Solve for the affine, inverse is M15 to M13, forward is M13 to M15
aff = apps.SolveAffine(landmarks)

with open(M13dir + 'TPS/M13_01_TPSLandmarks_5.txt', 'r') as m13:
    TPS13=[[float(v) for v in line.split()] for line in m13]

with open(M15dir + 'TPS/M15_01_TPSLandmarks_5.txt', 'r') as m15:
    TPS15=[[float(v) for v in line.split()] for line in m15]


if M15_to_M13:
    write = True

    def_aff = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyAffineReal(def_aff, M15, np.linalg.inv(aff))

   
    if write:
        cc.WriteMHA(def_aff, M15dir + 'Affine/M15_01_MRI_affine_to_M13.mha')
        np.save(M15dir + 'Affine/M15_01_MRI_affMat_to_M13.npy',np.linalg.inv(aff))

    landmarks = [[TPS13[x],TPS15[x]] for x in range(0,np.shape(TPS15)[0])]

    # Convert to real coordinates
    for lm in landmarks:
        lm[0] = np.ndarray.tolist(np.multiply(lm[0],M13_aff.spacing().tolist()) + M13_aff.origin().tolist())
        lm[1] = np.ndarray.tolist(np.multiply(lm[1],M13_aff.spacing().tolist()) + M13_aff.origin().tolist())

    spline = apps.SolveSpline(landmarks)
    h = apps.SplineToHField(spline, M13_aff.grid(), memT)
    def_TPS = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyHReal(def_TPS, def_aff, h)

    if write:
        cc.WriteMHA(def_TPS, M15dir + 'TPS/M15_01_TPS_to_M13.mha')
        cc.WriteMHA(h, M15dir + 'TPS/M15_01_TPS_Field_to_M13.mha')
        del def_aff, h
        gc.collect()

    M13_aff /= ca.Max(M13_aff)
    def_TPS /= ca.Max(def_TPS)

    cc.VarianceEqualize_I(M13_aff, sigma=2, eps=0.01)
    cc.VarianceEqualize_I(def_TPS, sigma=7, eps=0.1)

    eps=0.02
    sigma_I=0.09
    nIter_I=200
    [def_ID, theta, energy] = apps.IDiff(def_TPS, M13_aff, eps, sigma_I, nIter_I, plot=True, verbose=1)

    if write:
        cc.WriteMHA(def_ID, M15dir + 'IDiff/M15_01_ID_to_M13.mha')
        cc.WriteMHA(theta, M15dir + 'IDiff/M15_01_ID_Field_to_M13.mha')
    

    h = cc.LoadMHA(M15dir + 'TPS/M15_01_TPS_Field_to_M13.mha',memT)
    compDef = ca.Field3D(M13_aff.grid(), memT)
    ca.ComposeHH(compDef, h, theta, bg=ca.BACKGROUND_STRATEGY_CLAMP)

    common.DebugHere()

    Final_aff = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyAffineReal(Final_aff, M15, np.linalg.inv(aff))
    
    Final = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyHReal(Final, Final_aff, compDef)

    if write:
        cc.WriteMHA(Final, M15dir + 'FullDef/M15_01_MRI_as_M13.mha')
        cc.WriteMHA(compDef, M15dir + 'FullDef/M15_01_Field_to_M13.mha')


if not M15_to_M13:
    write = True

    M15_aff = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyAffineReal(M15_aff, M15, np.linalg.inv(aff))

    def_aff = ca.Image3D(M15.grid(), memT)
    cc.ApplyAffineReal(def_aff, M13_aff, aff)

    if write:
        cc.WriteMHA(def_aff, M13dir + 'Affine/M13_01_MRI_affine_to_M15.mha')
        np.save(M13dir + 'Affine/M13_01_MRI_affMat_to_M15.npy', aff)
        del def_aff
        gc.collect()

    landmarks = [[TPS15[x], TPS13[x]] for x in range(0,np.shape(TPS15)[0])]
    # Convert to real coordinates
    for lm in landmarks:
        lm[0] = np.ndarray.tolist(np.multiply(lm[0],M13_aff.spacing().tolist()) + M13_aff.origin().tolist())
        lm[1] = np.ndarray.tolist(np.multiply(lm[1],M13_aff.spacing().tolist()) + M13_aff.origin().tolist())

    spline = apps.SolveSpline(landmarks)
    h = apps.SplineToHField(spline, M13_aff.grid(), memT)
    def_TPS = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyHReal(def_TPS, M13_aff, h)

    def_TPS_aff = ca.Image3D(M15.grid(), memT)
    cc.ApplyAffineReal(def_TPS_aff, def_TPS, aff)
    
    if write:
         cc.WriteMHA(def_TPS_aff, M13dir + 'TPS/M13_01_TPS_to_M15.mha')
         cc.WriteMHA(h, M13dir + 'TPS/M13_01_TPS_Field_to_M15_in_M13space.mha')
         del def_TPS_aff, h
         gc.collect()

    M15_aff /= ca.Max(M15_aff)
    def_TPS /= ca.Max(def_TPS)

    cc.VarianceEqualize_I(def_TPS, sigma=2, eps=0.01)
    cc.VarianceEqualize_I(M15_aff, sigma=7, eps=0.1)

    eps=0.02
    sigma_I=0.09
    nIter_I=200
    [def_ID, theta, energy] = apps.IDiff(def_TPS, M15_aff, eps, sigma_I, nIter_I, plot=True, verbose=1)

    def_ID_aff = ca.Image3D(M15.grid(),memT)
    cc.ApplyAffineReal(def_ID_aff, def_ID, aff)

    if write:
        cc.WriteMHA(def_ID_aff, M13dir + 'IDiff/M13_01_ID_to_M15.mha')
        cc.WriteMHA(theta, M13dir + 'IDiff/M13_01_ID_Field_to_M15_in_M13space.mha')
        gc.collect()

    h = cc.LoadMHA(M13dir + 'TPS/M13_01_TPS_Field_to_M15_in_M13space.mha', memT)
    compDef = ca.Field3D(M13_aff.grid(), memT)
    ca.ComposeHH(compDef, h, theta, bg=ca.BACKGROUND_STRATEGY_CLAMP)

    Final_aff = ca.Image3D(M13_aff.grid(), memT)
    cc.ApplyHReal(Final_aff, M13_aff, compDef)

    Final = ca.Image3D(M15.grid(), memT)
    cc.ApplyAffineReal(Final, Final_aff, aff)    

    if write:
        cc.WriteMHA(Final, M13dir + 'FullDef/M13_01_MRI_as_M15.mha')
        cc.WriteMHA(compDef, M13dir + 'FullDef/M13_01_Field_to_M15_in_M13space.mha')



common.DebugHere()



