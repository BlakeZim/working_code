import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
import sys
import os
import glob
import dicom
from PyCAApps import IDiff, ElastReg, IDiffWeighted
# import matplotlib.pyplot as plt
# cc.SelectGPU()
plt.ion()
plt.close('all')

# indir = '/home/sci/blakez/M15-01/Data/MRI/HARVEY_OSAMA_TOBLAKE/Korenberg09_MonkeyBrain_Korenberg09_MonkeyBrain_082615_Harvey_ex__E'
#indir = '/home/sci/blakez/M15-01/Data/MRI/HARVEY_OSAMA_TOBLAKE/'
indir = '/usr/sci/brain/caleb/data2/DICOM/'
#ex = '_P1_2.16.756.5.5.100.1654368313.15323/'
#### outdir = '/home/sci/blakez/M15-01/Results/MRI/registration/'
outdir = '/home/sci/blakez/M15-01/Results/MRI/'

# folderList = sorted(glob.glob(indir + '*'))
# test = 46
# string = '__E'+str(test)
# curDir = [x for x in folderList if '__E'+str(test) in x]
 
# for a in folderList:
#     print a



#Load in the T2 image and downsample it to the resolution of the DW images
####
T2_list = sorted(glob.glob(indir + 'T2DICOM_scan26/*'))
refIm = dicom.read_file(T2_list[0])
PixelDims = (int(refIm.Rows),int(refIm.Columns), len(T2_list))
# PixelSpacing = (0.5,0.5,0.5)
T2Array = np.zeros(PixelDims, dtype = refIm.pixel_array.dtype)
for filename in T2_list:
    ds = dicom.read_file(filename)
    T2Array[:,:,T2_list.index(filename)] = ds.pixel_array
T2MRI = common.ImFromNPArr(T2Array)
T2MRI.setGrid(cc.MakeGrid(T2MRI.grid().size(), 0.5))
T2MRI.toType(ca.MEM_DEVICE)



#Swap the axis of the images so they align with the gradient directions
T2MRI = cc.SwapAxes(T2MRI,0,1)
T2MRI = cc.SwapAxes(T2MRI,0,2)
T2MRI = cc.FlipDim(T2MRI,2)
# T2MRI = cc.FlipDim(T2MRI,2)

DWIgrid = cc.MakeGrid([120,144,120],0.5,[0,0,0])
down_T2 = ca.Image3D(DWIgrid,ca.MEM_DEVICE)
ca.Resample(down_T2,T2MRI)
####
#Display the list






# cc.WriteMHA(down_T2,outdir + 'Images/T2_downSample_to_DWImha')


#Variance equalize the T2 image for registration
####
# T2_VE = ca.Image3D(DWIgrid,ca.MEM_DEVICE)
# ca.Copy(T2_VE,down_T2)
# cc.VarianceEqualize_I(T2_VE,sigma=5)
####
scanNums = range(46,68)
total_dirs = 0

for num in scanNums:

    folderList = sorted(glob.glob(indir + '*'))
    curDir = [x for x in folderList if '__E'+str(num) in x]

    #Reference Image for dimensions and data type
    dirList = sorted(glob.glob(curDir[0] + '/*'))
    refIm = dicom.read_file(dirList[0])
    PixelDims = (int(refIm.Rows),int(refIm.Columns), len(dirList))
    


    #Display the list
    # for a in dirList:
    #     print a

    
    #Load the B0 image for registration
    names = []
    allFiles = np.zeros(PixelDims,dtype=refIm.pixel_array.dtype)
    for filename in dirList:
        ds = dicom.read_file(filename)
        names.append(filename)
        allFiles[:,:,dirList.index(filename)] = ds.pixel_array
    B0 = common.ImFromNPArr(allFiles[:,:,0:120],mType=3)
    B0.setGrid(cc.MakeGrid(B0.grid().size(), 0.5))
    

    B0 = cc.SwapAxes(B0,0,1)
    B0 = cc.SwapAxes(B0,0,2)
    B0 = cc.FlipDim(B0,2)
    
    #Variance Equalize the B0 scan for registration
    ####
    # B0_VE = ca.Image3D(B0.grid(),ca.MEM_DEVICE)
    # ca.Copy(B0_VE,B0)
    # cc.VarianceEqualize_I(B0_VE,sigma=5)
    ####

    # cc.WriteMHA(B0, outdir + 'Images/scan{0}/B0_orig_scan{0}.mha'.format(num))
    
    #Idiff registration
    # DnIter = 1000
    # Dsigma = .9
    # Deps = .001
    # B0_reg_VE, Dphi, Denergy = IDiff(B0_VE, T2_VE, Deps, Dsigma, DnIter,False,1)
    # diff_reg = T2_VE-B0_reg_VE
    # cd.Disp3Pane(diff_reg)
    # cd.DispHGrid(Dphi,dim='x')

    #Elastic registration
    ####
    # EnIter = 1600
    # Esigma = .03
    # Eeps = .0005
    ####
    # while True:
        ####
        # B0_elast, Ephi, Eenergy = ElastReg(B0_VE, T2_VE, Eeps, Esigma, EnIter, plot=False,verbose=1)
        
        # Ediff_reg = T2_VE-B0_elast
        

        # plt.close('all')
        # cd.Disp3Pane(Ediff_reg)
        # cd.DispHGrid(Ephi,dim='x',sliceIdx=110)
        # print 'Are you satified with the registration (yes/No): '
        # answer = raw_input()
        
        # if answer=='yes':
        #     break
        # else:
        #     print 'Enter a new number of iterations (currently '+str(EnIter)+'): '
        #     EnIter = int(raw_input())
        #     print 'Enter a new penalty parameter (currently '+str(Esigma)+'): '    
        #     Esigma = float(raw_input())
        #     print 'Enter a new step size (currently '+str(Eeps)+'): '
        #     Eeps = float(raw_input())
        ####
    
    #Convert the found deformation field to world coordiantes to apply it
    #B0_reg = ca.Image3D(DWIgrid,ca.MEM_DEVICE)
    # break
    # cc.HtoReal(Ephi)
    # print ca.MinMax(Ephi)
    
    #Find the number of scans (over 240 because for some reason the dicom files were being counted 2 times)
#    path = indir+str(num)+ex
    scanCount = len(dirList)/120  #len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))])/240
    scanVec = range(0,scanCount) 
    scanList = [common.ImFromNPArr(allFiles[:,:,(scanVec[x]*120):(scanVec[x]*120+120)],mType=3,sp=ca.Vec3Df(0.5,0.5,0.5)) for x in scanVec]
    common.DebugHere()
    scanList = [cc.SwapAxes(scanList[x],0,1) for x in scanVec]
    scanList = [cc.SwapAxes(scanList[x],0,2) for x in scanVec]
    scanList = [cc.FlipDim(scanList[x],2) for x in scanVec]

    #scanList = [cc.LoadDICOM(indir+str(num)+ex+'MRIm****',slices=[scanVec[x]*120,scanVec[x]*120+120]) for x in scanVec]
    print 'There are {0} directions in this directory.'.format(scanCount-1)

    #Generate a list of Image3D's to contain the registered volumes
    def_list = [ca.Image3D(DWIgrid,ca.MEM_DEVICE) for x in scanVec]
    
    #Apply the deformation to all of the DW images
    ####[ca.ApplyH(def_list[x],scanList[x],Ephi) for x in scanVec]
    
    # Check for the sub directories and if they don't exist, make them
    if not os.path.exists(outdir + 'DWI_scan_{}'.format(num)):
        os.makedirs(outdir + 'DWI_scan_{}'.format(num))
    if not os.path.exists(outdir + 'DWI_scan_{}/individual_volumes'.format(num)):
        os.makedirs(outdir + 'DWI_scan_{}/individual_volumes'.format(num))
    if not os.path.exists(outdir + 'DWI_scan_{}/individual_volumes/MHA'.format(num)):
        os.makedirs(outdir + 'DWI_scan_{}/individual_volumes/MHA'.format(num))
    if not os.path.exists(outdir + 'DWI_scan_{}/individual_volumes/NIFTI'.format(num)):
        os.makedirs(outdir + 'DWI_scan_{}/individual_volumes/NIFTI'.format(num))

#    [cc.WriteMHA(def_list[x],outdir + 'T2_registered_scan{0}_direction{1}.mha'.format(num,str(x).zfill(2))) for x in scanVec]

 
    ####
    [cc.WriteMHA(scanList[x],outdir + 'DWI_scan_{0}/individual_volumes/MHA/non-registered_scan{0}_direction{1}.mha'.format(num,str(x).zfill(2))) for x in scanVec]
        
    total_dirs += scanCount    
        


        
        
        
        
        
        


















