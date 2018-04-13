import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCA.Core as ca
import gc
import numpy as np
import sys
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import csv
import json
import sys
import PyCAApps as apps
plt.ion()
plt.close('all')
import PyCA.Common as common
from PyCAApps import ElastReg, SolveSpline, SplineToHField, DefPoint, IDiff


livedir = '/home/sci/blakez/M15-01/Data/MRI/liveMRI/'
# livedir = '/local/blakez/korenbergNAS/3D_database/Raw/MRI/in_vivo_MRI/M15/HARVEY_IN_VIVO/'
# T2dir = '/home/sci/blakez/M13_01/data/MRI/'
T2dir = '/home/sci/blakez/M15-01/Data/MRI/HARVEY_OSAMA_TOBLAKE/'
ex = 'Korenberg09_MonkeyBrain_Korenberg09_MonkeyBrain_081815_Live_Harv__E16_P1_2.16.756.5.5.100.1654368313.31476.1439925928.1/'
SaveDir = '/home/sci/blakez/M15-01/Results/MRI/invivo_MRI/081815_E16/'

memT = ca.MEM_DEVICE
write=False

# T2 = cc.LoadMHA(T2dir + 'T2Seg_roty-119_flipy.mha',memT)
T2 = cc.LoadMHA(T2dir + 'T2_MRI.mha',memT)
T2_mask = cc.LoadMHA(T2dir + '../M15_01_MRI_Full_Mask.mha',memT)
T2 *= T2_mask
# rotMat = np.load(T2dir + 'rotationMatrix.npy')
# live = cc.LoadMHA(livedir + 'M13_01_live_MRI.mha',memT)
live = cc.LoadMHA(livedir + 'M15_01_live_MRI_E16.mha',memT)
live_mask = cc.LoadNRRD(SaveDir + 'M15_01_live_MRI_mask.nrrd',memT)


live_mask.setOrigin(live.origin())
live -= ca.Min(live)
live /= ca.Max(live)
live *= live_mask


with open(SaveDir + 'M15_01_LiveE16-T2_Landmarks.json', 'r') as f:
    landmarks = json.load(f)

# live.setOrigin(ca.Vec3Df(-119.5,-59.5,-99.5))
# live.setSpacing(ca.Vec3Df(1,1,1))
T2Grid = T2.grid()

# Landmark pairs for the registration
# landmarks = [[[ 99.0, 50.0, 121.0] , [102.0,118.0,135.0]],  #good
#              [[ 123.0, 50.0, 121.0] , [151.0,120.0,135.0]], #good
#              [[ 112.0, 50.0, 191.0] , [124.0,248.0,135.0]], #good
#              [[ 69.0, 85.0, 109.0] , [66.0,101.0,88.0]], #good
#              [[ 112.0, 72.0, 154.0] , [128.0,185.0,101.0]], #good
#              [[ 85.0, 82.0, 170.0] , [78.0,208.0,83.0]], 
#              [[ 140.0, 82.0, 169.0] , [177.0,210.0,88.0]], 
#              [[ 73.0, 92.0, 97.0] , [60.0, 82.0, 77.0]], # TEST
#              [[ 124.0, 66.0, 66.0] , [147.0,34.0,127.0]],
#              [[ 100.0, 66.0, 66.0] , [119.0,32.0,128.0]],
#              [[ 110.0, 35.0, 121.0] , [125.0,135.0,168.0]],
#              [[ 111.0, 60.0, 97.0] , [128.0,87.0,135.0]]] #good

# with open(SaveDir + 'T2_registered/M13_01_Live-T2_Landmarks.json', 'w') as f:
#     json.dump(landmarks, f)

realLM = np.array(landmarks)
realLM[:,0] = realLM[:,0]*live.spacing() + live.origin()
realLM[:,1] = realLM[:,1]*T2.spacing() + T2.origin()

A = apps.SolveAffine(realLM)
liveDef = T2.copy()
cc.ApplyAffineReal(liveDef,live,A)


# # Convert the landmarks to real coordinates and exchange the ordering of the so live is going to T2


# realLM[:,1] = realLM[:,1] - 127.5
# flipLM = np.fliplr(realLM)

# # Solve for the TPS based off of the landmakrs
# spline = SolveSpline(flipLM)
# h = SplineToHField(spline, T2Grid, memT)
# print ca.MinMax(h)
# liveDef = T2.copy()
# cc.ApplyHReal(liveDef,live,h)

# Variance equalize the volumes and blur the live
T2_VE = ca.Image3D(T2.grid(),memT)
live_VE = ca.Image3D(liveDef.grid(),memT)
ca.Copy(T2_VE,T2)
cc.VarianceEqualize_I(T2_VE, sigma=5)
ca.Copy(live_VE,liveDef)
cc.VarianceEqualize_I(live_VE, sigma=5)
gausfilt = ca.GaussianFilterGPU()
gausfilt.updateParams(live_VE.size(),ca.Vec3Df(3,3,3),ca.Vec3Di(3,3,3))
live_VEfilt = ca.Image3D(live_VE.grid(),memT)
temp = ca.Image3D(live_VE.grid(),memT)
gausfilt.filter(live_VEfilt,live_VE,temp)
dispslice = [128,120,128]

# Display some initial images
cd.Disp3Pane(live_VEfilt,rng=[-3,3],sliceIdx=dispslice,title='Live VE Filtered')
cd.Disp3Pane(T2_VE,rng=[-3,3],sliceIdx=dispslice,title='T2 VE')
print 'MinMax of Live = '+str(ca.MinMax(live_VE))
print 'MinMax of T2 = '+str(ca.MinMax(T2_VE))
preRegDiff = T2_VE-live_VEfilt
cd.Disp3Pane(preRegDiff,rng=[-3,3],sliceIdx=dispslice,title = 'T2 less Live')

# Perform Elastic Registration
EnIter = 100
Esigma = 0.35
Eeps = 0.01
Efp = [1,1,0.001]


live_VEfilt_ER,phi,energy = ElastReg(live_VEfilt,T2_VE, Eeps,Esigma,EnIter,
                                 fluidParams=Efp,plot=False,verbose=1)
diff = T2_VE - live_VEfilt_ER
cd.Disp3Pane(diff,rng=[-3,3],sliceIdx=dispslice)
cd.EnergyPlot(energy, legend=['Reg','Data','Total'])
cd.DispHGrid(phi)
print ca.MinMax(phi)


# Perform IDiff Registration
InIter = 200
Isigma = 0.29
Ieps = 0.01

live_VEfilt_ID,Iphi,Ienergy = IDiff(live_VEfilt_ER,T2_VE, Ieps, Isigma,
                                  InIter, plot=False, verbose=1)

Idiff = T2_VE - live_VEfilt_ID
cd.Disp3Pane(Idiff,rng=[-3,3],sliceIdx=dispslice)
cd.EnergyPlot(Ienergy, legend=['Reg','Data','Total'])
cd.DispHGrid(Iphi)
print ca.MinMax(Iphi)


# Compose the deformations and apply the total deformation to the initial live volume
# tempDef = ca.Field3D(phi.grid(), memT)
totDef = ca.Field3D(phi.grid(), memT)
ca.ComposeHH(totDef, phi, Iphi, ca.BACKGROUND_STRATEGY_CLAMP)
# ca.ComposeHH(totDef, h, tempDef, ca.BACKGROUND_STRATEGY_CLAMP)

#Apply the deformation to the TPS live volume and rotate to the original volume
live_T2reg_rot = T2.copy()
cc.HtoReal(totDef)
cc.ApplyHReal(live_T2reg_rot,liveDef,totDef)
cd.Disp3Pane(live_T2reg_rot)
# live_T2reg = T2.copy()
# cc.ApplyAffineReal(live_T2reg,live_T2reg_rot,np.linalg.inv(rotMat))
# cd.Disp3Pane(live_T2reg)

if write:
    cc.WriteMHA(live_T2reg_rot, SaveDir + 'M13_01_live_as_MRI_full_bw_256_roty-119_flipy.mha')
    cc.WriteMHA(live_T2reg, SaveDir + 'M13_01_live_as_MRI_full_bw_256.mha')
    cc.WriteMHA(totDef, SaveDir + 'M13_01_live_to_MRI_roty-119_flipy_full_256.mha')
    paramDict = {}
    paramDict['Number of Itterations ER: '] = EnIter
    paramDict['Sigma ER: '] = Esigma
    paramDict['Step Size ER: '] = Eeps
    paramDict['Fluid Parameters ER: '] = Efp
    paramDict['Number of Itterations ID: '] = InIter
    paramDict['Sigma ID: '] = Isigma
    paramDict['Step Size ID: '] = Ieps
    with open(SaveDir + 'M13_01_live_to_MRI_reg_param.txt','w') as f:
        w = csv.writer(f)
        w.writerows(paramDict.items())

