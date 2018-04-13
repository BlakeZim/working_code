import PyCA.Core as ca
import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend('gtkagg')
import numpy as np
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Common as common
import PyCABlakeExtras.Common as cb
import AppUtils.Config as Config
import IDiff.Matching
import PyCAApps as apps
import matplotlib
from PIL import Image
import scipy
from scipy import ndimage
plt.ion()
plt.close('all')
import PointPicker2D as pp
import pickle
import glob
import sys
import gc
import json
import os
import os.path as pth
import yaml

cc.SelectGPU(2)

plt.close('all')

#Necessary function for loading YAML files
def _valspecdef(cf):
    '''Configuration Validation'''
    pass

secSpec = {
    'secNum': Config.Param(required=True,
                           comment='Section number'),
    'mkyNum': Config.Param(required=True,
                           comment='Monkey number'),
    'yamlList': Config.Param(required=True,
                             default='!include section_1438_frag0.yaml',
                             comment='Yaml file list for all of the fragments'),
    'ssiSrcPath': Config.Param(required=True,
                               comment='Path to sidescape source images'),
    'ssiMskPath': Config.Param(required=True,
                               comment='Path to sidescape mask images'),
    'ssiOutPath': Config.Param(required=True,
                               comment='Path to sidescape output images'),
    'bfiSrcPath': Config.Param(required=True,
                               comment='Path to blockface source images'),
    'bfiMskPath': Config.Param(required=True,
                               comment='Path to blockface mask iamges'),
    'bfiOutPath': Config.Param(required=True,
                               comment='Path to blockface ouput images'),
    'ssiOrgPath': Config.Param(required=False,
                                default='~/korenbergNAS/3D_database/Working/Microscopic/epi-fluorescence_microscopic/M15/SideLight_Zeiss/Tiffs/',
                                comment='Path to full resolution sidescape images'),
    '_validation_hook':_valspecdef
}

frgSpec = {
    'bfiSrcName': Config.Param(required=True,
                               comment='Name of the blockface source image'),
    'ssiSrcName': Config.Param(required=True,
                               comment='Name of the sidescape source image'),
    'ssiMskName': Config.Param(required=True,
                               comment='Name of the sidescape mask image'),
    'bfiMskName': Config.Param(required=True,
                               comment='Name of the blockface mask image'),
    'affine': Config.Param(required=False,
                           default='None',
                           comment='Affine transform relating the image fragments'),
    'scale': Config.Param(required=False,
                           default=np.array([8,4,2,1]).tolist(),
                           comment='Scales for the multi-scale image registration'),
    'sigVarSsi': Config.Param(required=False,
                              default=40,
                              comment='Sigma for variance equalization of the sidescape image'),
    'sigVarBfi': Config.Param(required=False,
                              default=40,
                              comment='Sigma for variance equalization of the blockface image'),
    'epsVar': Config.Param(required=False,
                           default=0.001,
                           comment='Division regularizer for variance equalization of both blockface and sidescape images'),
    'sigBlur': Config.Param(required=False,
                            default=np.array([10,8,5,3]).tolist(),
                            comment='Sigma of the gaussian blur function for the different levels of reigstration'),
    'kerBlur': Config.Param(required=False,
                            default=np.array([30,30,15,15]).tolist(),
                            comment='Kernel size for the gaussian blur at different levels of registration'),
    'iters': Config.Param(required=False,
                          default=np.array([1000,800,600,500]).tolist(),
                          comment='Number of iterations for the different levels of regisrtaion'),
    'sigReg': Config.Param(required=False,
                           default=np.array([5.0,3.0,1.7,0.7]).tolist(),
                           comment='Sigma (smoothness) constraint for the different levels of registration'),
    'epsReg': Config.Param(required=False,
                           default=np.array([0.0005, 0.0004, 0.0004, 0.0008]).tolist(),
                           comment='Epsilon step size for the different levels of registration'),
    'imSize': Config.Param(required=False,
                           comment='Size of the image'),
    'imOrig': Config.Param(required=False,
                           comment='Origin of the image'),
    'imSpac': Config.Param(required=False,
                           comment='Spacing of the image'),
    'rgnBfi': Config.Param(required=False,
                           comment='Upper left to lower right corner of the blockfacefragment region relative to image size'),
    'rgnSsi': Config.Param(required=False,
                           comment='Upper left to lower right corner of the blockfacefragment region relative to image size'),
    'orgSsiList': Config.Param(required=False,
                                default=None,
                                comment='List of high resolution image names to apply the deformation to'),
    '_validation_hook': _valspecdef
}

def updateFragOb(cfOb):
    with open(os.path.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/section_{1}_frag{2}.yaml'.format(secOb.mkyNum,str(secOb.secNum).zfill(4),frgNum)), 'w') as f:
        f.write(Config.ConfigToYAML(frgSpec, cfOb))

def Loader(cfOb,memT):
    '''Function for loading all of the images. All images get normalized between 0 and 1'''

    #Load Source Images
    bfiSrc = cc.LoadColorMHA(pth.expanduser(secOb.bfiSrcPath + cfOb.bfiSrcName), memT)
    ssiSrc = common.LoadITKImage(pth.expanduser(secOb.ssiSrcPath + cfOb.ssiSrcName), memT)
    ssiSrc /= ca.Max(ssiSrc)

    #Load Mask Image
    bfiMsk = common.LoadITKImage(pth.expanduser(secOb.bfiMskPath + cfOb.bfiMskName), memT)
    bfiMsk /= ca.Max(bfiMsk)
    bfiMsk.setGrid(bfiSrc.grid())
    ssiMsk = common.LoadITKImage(pth.expanduser(secOb.ssiMskPath + cfOb.ssiMskName), memT)
    ssiMsk /= ca.Max(ssiMsk)
    
    return ssiSrc, bfiSrc, ssiMsk, bfiMsk
    
def Affine(I_src, I_tar, cfOb):
    '''Function for finding and applyiing the affine between two image fragments, I_src and I_tar. Returns both images with the affine applied and the affine trasform'''

    if cfOb.affine=='None':
        landmarks = pp.LandmarkPicker([np.squeeze(I_src.asnp()), np.squeeze(I_tar.asnp())]) #will be forward from BFI to SSI
        for lm in landmarks:
            lm[0] = np.ndarray.tolist(np.multiply(lm[0],I_src.spacing().tolist()[0:2]) + I_src.origin().tolist()[0:2])
            lm[1] = np.ndarray.tolist(np.multiply(lm[1],I_tar.spacing().tolist()[0:2]) + I_tar.origin().tolist()[0:2])
        aff = apps.SolveAffine(landmarks)
        cfOb.affine = aff.tolist()
 
    I_src_aff = I_tar.copy()
    I_tar_aff = I_src.copy()
    
    cc.ApplyAffineReal(I_src_aff, I_src, cfOb.affine)
    cc.ApplyAffineReal(I_tar_aff, I_tar, np.linalg.inv(np.array(cfOb.affine)))

    return I_src_aff, I_tar_aff, cfOb.affine

def DefReg(I_src, I_tar, config, memT, idConf):

    I_src.toType(memT)
    I_tar.toType(memT)

    # Convert to 2D spacing (because it really matters)
    sp2D = I_src.spacing().tolist()
    sp2D = ca.Vec3Df(sp2D[0],sp2D[1],1)
    
    I_tar.setSpacing(sp2D)
    I_src.setSpacing(sp2D)
    gridReg = I_tar.grid()
        
    # Blur the images
    I_tar_blur = I_tar.copy()
    I_src_blur = I_src.copy()
    temp = ca.Image3D(I_tar.grid(), memT)
    gausFilt = ca.GaussianFilterGPU()
    
    scaleList = config.scale

    # Initiate the scale manager
    scaleManager=ca.MultiscaleManager(gridReg)
    for s in scaleList:
        scaleManager.addScaleLevel(s)
    if memT == ca.MEM_HOST:
        resampler = ca.MultiscaleResamplerGaussCPU(gridReg)
    else:
        resampler = ca.MultiscaleResamplerGaussGPU(gridReg)

    # Generate the scratch images
    scratchITar = ca.Image3D(gridReg, memT)
    scratchISrc = ca.Image3D(gridReg, memT)
    scratchI = ca.Image3D(gridReg, memT)
    scratchF = ca.Field3D(gridReg, memT)
    compF = ca.Field3D(gridReg, memT)
    
    def SetScale(scale):
        '''Scale Management for Multiscale'''
        scaleManager.set(scale)
        resampler.setScaleLevel(scaleManager)
        curGrid = scaleManager.getCurGrid()
        curGrid.spacing().z = 1 # Because only 2D
        
        print 'Inside setScale(). Current grid is ', curGrid

        if scaleManager.isLastScale():
            print 'Inside setScale(): **Last Scale**'
        if scaleManager.isFirstScale():
            print 'Inside setScale(): **First Scale**'

        
        scratchISrc.setGrid(curGrid)
        scratchITar.setGrid(curGrid)
        scratchI.setGrid(curGrid)
        compF.setGrid(curGrid)
        idConf.study.I0 = ca.Image3D(curGrid, memT)
        idConf.study.I1 = ca.Image3D(curGrid, memT)
        
        if scaleManager.isLastScale():
	    	s = config.sigBlur[scaleList.index(sc)]
	        r = config.kerBlur[scaleList.index(sc)]
	        gausFilt.updateParams(I_tar.size(), ca.Vec3Df(r,r,r), ca.Vec3Di(s,s,s))
	        gausFilt.filter(scratchITar, I_tar, temp)
	        gausFilt.filter(scratchI, I_src, temp)
	        # ca.Copy(scratchI, I_src)
	        # ca.Copy(scratchITar, I_tar)
 
        else:
            s = config.sigBlur[scaleList.index(sc)]
            r = config.kerBlur[scaleList.index(sc)]
            gausFilt.updateParams(I_tar.size(), ca.Vec3Df(r,r,r), ca.Vec3Di(s,s,s))
            gausFilt.filter(I_tar_blur, I_tar, temp)
            gausFilt.filter(I_src_blur, I_src, temp)
            resampler.downsampleImage(scratchI, I_src_blur)
            resampler.downsampleImage(scratchITar, I_tar_blur)
            
        if scaleManager.isFirstScale():
            scratchF.setGrid(curGrid)
            scratchITar.setGrid(curGrid)
            ca.SetToIdentity(scratchF)
            ca.ApplyH(scratchISrc, scratchI, scratchF)
            
        else:
            compF.setGrid(scratchF.grid())
            ca.ComposeHH(compF,scratchF,h)
            resampler.updateHField(scratchF)
            resampler.updateHField(compF)
            ca.Copy(scratchF, compF)
            ca.ApplyH(scratchISrc, scratchI, compF)

        
    for sc in scaleList:
        SetScale(scaleList.index(sc))

        #Set the optimize parameters in the IDiff configuration object
        idConf.optim.Niter=config.iters[scaleList.index(sc)]
        idConf.optim.stepSize=config.epsReg[scaleList.index(sc)]
        idConf.idiff.regWeight=config.sigReg[scaleList.index(sc)]
        ca.Copy(idConf.study.I0, scratchISrc)
        ca.Copy(idConf.study.I1, scratchITar)
        idConf.io.plotEvery = config.iters[scaleList.index(sc)]

        h = IDiff.Matching.Matching(idConf)
        tempScr = scratchISrc.copy()
        ca.ApplyH(tempScr, scratchISrc, h)

        #Plot the images to see the change
        cd.DispImage(scratchISrc - scratchITar, rng=[-2,2], title='Orig Diff', colorbar=True)
        cd.DispImage(tempScr - scratchITar, rng=[-2,2], title='Reg Diff', colorbar=True)
        

        # common.DebugHere()
        
        # I_src_def = idConf.study.I0.copy()
 
        # scratchITar = idConf.study.I1
        # eps = config.epsReg[scaleList.index(sc)]
        # sigma = config.sigReg[scaleList.index(sc)]
        # nIter = config.iters[scaleList.index(sc)]
        # # common.DebugHere()
        # [I_src_def, h, energy] = apps.IDiff(scratchISrc, scratchITar, eps, sigma, nIter, plot=True, verbose=1)
    ca.ComposeHH(scratchF, compF, h)
    I_src_def = idConf.study.I0.copy()

    return I_src_def, scratchF


def ConvertGrid(I_src_grid, I_tar_grid):
    '''Given a source grid and target grid, returns a grid for the source that is in the world coordinates of the target grid'''
    nS = np.multiply( np.divide(I_tar_grid.size().tolist(),[float(i) for i in I_src_grid.size().tolist()]), I_tar_grid.spacing().tolist())
    return cc.MakeGrid(I_src_grid.size(), ca.Vec3Df(nS[0], nS[1], nS[2]), 'center')

# def PreProcess(bfi, ssi):
#     cc.VarianceEqualize_I(bfi, sigma=cf.sigVe, eps=cf.bfiVeEps)
#     cc.VarianceEqualize_I(ssi, sigma=cf.sigVe, eps=cf.ssiVeEps)

def Fragmenter():
    tmpOb = Config.Load(frgSpec,pth.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/section_{1}_frag0.yaml'.format(secOb.mkyNum,secOb.secNum)))
    dictBuild = {}
    #Load in the whole image so that the fragment can cropped out
    ssiSrc, bfiSrc, ssiMsk, bfiMsk = Loader(tmpOb, ca.MEM_HOST)

    #Because some of the functions only woth with gray images
    bfiGry = ca.Image3D(bfiSrc.grid(), bfiSrc.memType())
    ca.Copy(bfiGry,bfiSrc,1)

    lblSsi,_ = ndimage.label(np.squeeze(ssiMsk.asnp()) > 0)
    lblBfi,_ = ndimage.label(np.squeeze(bfiMsk.asnp()) > 0)

    seedPt = np.squeeze(pp.LandmarkPicker([lblBfi, lblSsi]))
    subMskBfi = common.ImFromNPArr(lblBfi==lblBfi[seedPt[0,0],seedPt[0,1]].astype('int8'),sp=bfiSrc.spacing(),orig=bfiSrc.origin())
    subMskSsi = common.ImFromNPArr(lblSsi==lblSsi[seedPt[1,0],seedPt[1,1]].astype('int8'),sp=ssiSrc.spacing(),orig=ssiSrc.origin())

    bfiGry *= subMskBfi
    bfiSrc *= subMskBfi
    ssiSrc *= subMskSsi
    #Pick points that are the bounding box of the desired subvolume
    corners = np.array(pp.LandmarkPicker([np.squeeze(bfiGry.asnp()), np.squeeze(ssiSrc.asnp())]))
    bfiCds = corners[:,0]
    ssiCds = corners[:,1]

    #Extract the region from the source images
    bfiRgn = cc.SubVol(bfiSrc, xrng=[bfiCds[0,0], bfiCds[1,0]], yrng = [bfiCds[0,1],bfiCds[1,1]])
    ssiRgn = cc.SubVol(ssiSrc, xrng=[ssiCds[0,0], ssiCds[1,0]], yrng = [ssiCds[0,1],ssiCds[1,1]])

    #Extract the region from the mask images
    rgnMskSsi = cc.SubVol(subMskSsi, xrng=[ssiCds[0,0], ssiCds[1,0]], yrng = [ssiCds[0,1],ssiCds[1,1]])
    rgnMskBfi = cc.SubVol(subMskBfi, xrng=[bfiCds[0,0], bfiCds[1,0]], yrng = [bfiCds[0,1],bfiCds[1,1]])
    
    dictBuild['rgnBfi'] = np.divide(bfiCds,np.array(bfiSrc.size().tolist()[0:2],'float')).tolist()
    dictBuild['rgnSsi'] = np.divide(ssiCds,np.array(ssiSrc.size().tolist()[0:2],'float')).tolist()

    #Check the output directory for the source files of the fragment
    if not pth.exists(pth.expanduser(secOb.ssiSrcPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.ssiSrcPath + 'frag{0}'.format(frgNum)))
    if not pth.exists(pth.expanduser(secOb.bfiSrcPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.bfiSrcPath + 'frag{0}'.format(frgNum)))
    #Check the output directory for the mask files of the fragment
    if not pth.exists(pth.expanduser(secOb.ssiMskPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.ssiMskPath + 'frag{0}'.format(frgNum)))
    if not pth.exists(pth.expanduser(secOb.bfiMskPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.bfiMskPath + 'frag{0}'.format(frgNum)))

        
    dictBuild['ssiSrcName'] = 'frag{0}/M{1}_01_ssi_section_{2}_frag1.tif'.format(frgNum,secOb.mkyNum,secOb.secNum)
    dictBuild['bfiSrcName'] = 'frag{0}/M{1}_01_bfi_section_{2}_frag1.mha'.format(frgNum,secOb.mkyNum,secOb.secNum)
    dictBuild['ssiMskName'] = 'frag{0}/M{1}_01_ssi_section_{2}_frag1_mask.tif'.format(frgNum,secOb.mkyNum,secOb.secNum)
    dictBuild['bfiMskName'] = 'frag{0}/M{1}_01_bfi_section_{2}_frag1_mask.tif'.format(frgNum,secOb.mkyNum,secOb.secNum)


    #Write out the masked and cropped images so that they can be loaded from the YAML file
    #The BFI region needs to be saved as color and mha format so that the grid information is carried over. 
    common.SaveITKImage(ssiRgn, pth.expanduser(secOb.ssiSrcPath + dictBuild['ssiSrcName']))
    cc.WriteColorMHA(bfiRgn, pth.expanduser(secOb.bfiSrcPath + dictBuild['bfiSrcName']))
    common.SaveITKImage(rgnMskSsi, pth.expanduser(secOb.ssiMskPath + dictBuild['ssiMskName']))
    common.SaveITKImage(rgnMskBfi, pth.expanduser(secOb.bfiMskPath + dictBuild['bfiMskName']))

    frgOb = Config.MkConfig(dictBuild, frgSpec)
    updateFragOb(frgOb)

    return None

# def appLarge(ssiLrgLst, h):
#     h.toType(ca.MEM_HOST)
#     #Load the original resolution image
#     ssiOrg = common.LoadITKImage(pth.expanduser(secOb.ssiOrgPath + ssiLrgLst[0]), ca.MEM_HOST)

#     common.DebugHere()

#     #Create the Grid that the deformation needs to be upsampled to
#     lrgGrd = ConvertGrid(cc.MakeGrid(ssiOrg.size(), ca.Vec3Df(1,1,1), ca.Vec3Df(0,0,0)), h.grid())
#     upH = ca.Field3D(lrgGrd, ca.MEM_HOST)
#     cc.ResampleWorld(upH, h, bg=2)

#     common.DebugHere()


def main():
    # Extract the Monkey number and section number from the command line
    global frgNum
    global secOb
    
    mkyNum = sys.argv[1]
    secNum = sys.argv[2]
    frgNum = int(sys.argv[3])
    write=True

    # if not os.path.exists(os.path.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/include_configFile.yaml'.format(mkyNum,secNum))):
    #     cf = initial(secNum, mkyNum)
    
    try:
        secOb = Config.Load(secSpec,pth.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/include_configFile.yaml'.format(mkyNum,secNum)))
    except IOError as e:
        try:
            temp = Config.LoadYAMLDict(pth.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/include_configFile.yaml'.format(mkyNum,secNum)),include=False)
            secOb = Config.MkConfig(temp, secSpec)
        except IOError:
            print 'It appears there is no configuration file for this section. Please initialize one and restart.'
            sys.exit()
        if frgNum==int(secOb.yamlList[frgNum][-6]):
            Fragmenter()
            try:
                secOb = Config.Load(secSpec,pth.expanduser('~/korenbergNAS/3D_database/Working/configuration_files/SidescapeRelateBlockface/M{0}/section_{1}/include_configFile.yaml'.format(mkyNum,secNum)))
            except IOError:
                print 'It appeas that the include yaml file list does not match your fragmentation number. Please check them and restart.'
                sys.exit()

    if not pth.exists(pth.expanduser(secOb.ssiOutPath + 'frag{0}'.format(frgNum))):
        common.Mkdir_p(pth.expanduser(secOb.ssiOutPath + 'frag{0}'.format(frgNum)))
    if not pth.exists(pth.expanduser(secOb.bfiOutPath + 'frag{0}'.format(frgNum))):
        common.Mkdir_p(pth.expanduser(secOb.bfiOutPath + 'frag{0}'.format(frgNum)))
    if not pth.exists(pth.expanduser(secOb.ssiSrcPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.ssiSrcPath + 'frag{0}'.format(frgNum)))
    if not pth.exists(pth.expanduser(secOb.bfiSrcPath + 'frag{0}'.format(frgNum))):
        os.mkdir(pth.expanduser(secOb.bfiSrcPath + 'frag{0}'.format(frgNum)))
        
    frgOb = Config.MkConfig(secOb.yamlList[frgNum], frgSpec)
    ssiSrc, bfiSrc, ssiMsk, bfiMsk = Loader(frgOb, ca.MEM_HOST)

    #Extract the saturation Image from the color iamge
    bfiHsv = common.FieldFromNPArr(matplotlib.colors.rgb_to_hsv(np.rollaxis(np.array(np.squeeze(bfiSrc.asnp())),0,3)),ca.MEM_HOST)
    bfiHsv.setGrid(bfiSrc.grid())
    bfiSat = ca.Image3D(bfiSrc.grid(), bfiHsv.memType())
    ca.Copy(bfiSat,bfiHsv,1)
    #Histogram equalize, normalize and mask the blockface saturation image
    bfiSat = cb.HistogramEqualize(bfiSat,256)
    bfiSat.setGrid(bfiSrc.grid())
    bfiSat *= -1
    bfiSat -= ca.Min(bfiSat)
    bfiSat /= ca.Max(bfiSat)
    bfiSat *= bfiMsk
    bfiSat.setGrid(bfiSrc.grid())

    #Write out the blockface region after adjusting the colors with a format that supports header information
    if write:
        common.SaveITKImage(bfiSat, pth.expanduser(secOb.bfiSrcPath + 'frag{0}/M{1}_01_bfi_section_{2}_frag{0}_sat.nrrd'.format(frgNum, secOb.mkyNum,secOb.secNum)))

    #Set the sidescape grid relative to that of the blockface
    ssiSrc.setGrid(ConvertGrid(ssiSrc.grid(), bfiSat.grid()))
    ssiMsk.setGrid(ConvertGrid(ssiMsk.grid(), bfiSat.grid()))
    ssiSrc *= ssiMsk

    #Write out the sidescape masked image in a format that stores the header information
    if write:
        common.SaveITKImage(ssiSrc, pth.expanduser(secOb.ssiSrcPath + 'frag{0}/M{1}_01_ssi_section_{2}_frag{0}.nrrd'.format(frgNum, secOb.mkyNum,secOb.secNum)))
    
    #Update the image parameters of the sidescape image for future use
    frgOb.imSize = ssiSrc.size().tolist()
    frgOb.imOrig = ssiSrc.origin().tolist()
    frgOb.imSpac = ssiSrc.spacing().tolist()
    updateFragOb(frgOb)
    
    #Find the affine transform between the two fragments
    bfiAff,ssiAff,aff = Affine(bfiSat, ssiSrc, frgOb)
    updateFragOb(frgOb)

    #Write out the affine transformed images in a format that stores header information
    if write:
        common.SaveITKImage(bfiAff, pth.expanduser(secOb.bfiOutPath + 'frag{0}/M{1}_01_bfi_section_{2}_frag{0}_aff_ssi.nrrd'.format(frgNum, secOb.mkyNum,secOb.secNum)))
        common.SaveITKImage(ssiAff, pth.expanduser(secOb.ssiOutPath + 'frag{0}/M{1}_01_ssi_section_{2}_frag{0}_aff_bfi.nrrd'.format(frgNum, secOb.mkyNum,secOb.secNum)))
        
    bfiVe = bfiAff.copy()
    ssiVe = ssiSrc.copy()
    cc.VarianceEqualize_I(bfiVe, sigma=frgOb.sigVarBfi, eps=frgOb.epsVar)
    cc.VarianceEqualize_I(ssiVe, sigma=frgOb.sigVarSsi, eps=frgOb.epsVar)

    #As of right now, the largest pre-computed FFT table is 2048, so resample onto that grid for registration
    regGrd = ConvertGrid(cc.MakeGrid(ca.Vec3Di(2048,2048,1),ca.Vec3Df(1,1,1),ca.Vec3Df(0,0,0)), ssiSrc.grid())
    ssiReg = ca.Image3D(regGrd, ca.MEM_HOST)
    bfiReg = ca.Image3D(regGrd, ca.MEM_HOST)
    cc.ResampleWorld(ssiReg, ssiVe)
    cc.ResampleWorld(bfiReg, bfiVe)

    #Create the default configuration object for IDiff Matching and then set some parameters
    idCf = Config.SpecToConfig(IDiff.Matching.MatchingConfigSpec)
    idCf.compute.useCUDA = True
    idCf.io.outputPrefix='/home/sci/blakez/IDtest/'

    #Run the registration
    ssiDef, phi = DefReg(ssiReg, bfiReg, frgOb, ca.MEM_DEVICE, idCf)


    #Turn the deformation into a displacement field so it can be applied to the large tif with C++ code
    affV = phi.copy()
    cc.ApplyAffineReal(affV, phi, np.linalg.inv(frgOb.affine))
    ca.HtoV_I(affV)

    #Apply the found deformation to the input ssi 
    ssiSrc.toType(ca.MEM_DEVICE)
    cc.HtoReal(phi)
    affPhi = phi.copy()
    ssiBfi = ssiSrc.copy()
    upPhi = ca.Field3D(ssiSrc.grid(), phi.memType())

    cc.ApplyAffineReal(affPhi, phi, np.linalg.inv(frgOb.affine))
    cc.ResampleWorld(upPhi, affPhi, bg=2) 
    cc.ApplyHReal(ssiBfi, ssiSrc, upPhi)

    # ssiPhi = ca.Image3D(ssiSrc.grid(), phi.memType())
    # upPhi = ca.Field3D(ssiSrc.grid(), phi.memType())
    # cc.ResampleWorld(upPhi, phi, bg=2)
    # cc.ApplyHReal(ssiPhi, ssiSrc, upPhi)
    # ssiBfi = ssiSrc.copy()
    # cc.ApplyAffineReal(ssiBfi, ssiPhi, np.linalg.inv(frgOb.affine))
    
    # #Apply affine to the deformation
    # affPhi = phi.copy()
    # cc.ApplyAffineReal(affPhi, phi, np.linalg.inv(frgOb.affine))

    if write:
        common.SaveITKImage(ssiBfi, pth.expanduser(secOb.ssiOutPath + 'frag{0}/M{1}_01_ssi_section_{2}_frag{0}_def_bfi.nrrd'.format(frgNum, secOb.mkyNum,secOb.secNum)))
        cc.WriteMHA(affPhi, pth.expanduser(secOb.ssiOutPath + 'frag{0}/M{1}_01_ssi_section_{2}_frag{0}_to_bfi_real.mha'.format(frgNum, secOb.mkyNum,secOb.secNum)))
        cc.WriteMHA(affV, pth.expanduser(secOb.ssiOutPath + 'frag{0}/M{1}_01_ssi_section_{2}_frag{0}_to_bfi_disp.mha'.format(frgNum, secOb.mkyNum,secOb.secNum)))


    #Create the list of names that the deformation should be applied to
    # nameList = ['M15_01_0956_SideLight_DimLED_10x_ORG.tif',
    #             'M15_01_0956_TyrosineHydroxylase_Ben_10x_Stitching_c1_ORG.tif',
    #             'M15_01_0956_TyrosineHydroxylase_Ben_10x_Stitching_c2_ORG.tif',
    #             'M15_01_0956_TyrosineHydroxylase_Ben_10x_Stitching_c3_ORG.tif']

    # appLarge(nameList, affPhi)

    common.DebugHere()
    
if __name__ == '__main__':
    main()
