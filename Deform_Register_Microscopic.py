import PyCA.Core as ca
import PyCACalebExtras.SetBackend

plt = PyCACalebExtras.SetBackend.SetBackend('tkagg')
import numpy as np
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import AppUtils.Config as Config
import PyCA.Common as common
import PyCABlakeExtras.Common as cb
import PointPicker2D as pp
import PyCAApps as apps
import matplotlib
import scipy
import yaml
from scipy import ndimage

plt.ion()
plt.close('all')
import glob
import json
import sys
import gc
import os
import os.path as pth

cc.SelectGPU(2)


def load_and_solve(target_points_file, source_points_file):
    target_points = np.loadtxt(target_points_file, delimiter=' ')[:, 0:2]
    source_points = np.loadtxt(source_points_file, delimiter=' ')[:, 0:2]

    landmarks = []
    for pt in range(0, len(target_points)):
        landmarks.append([source_points[pt].tolist(), target_points[pt].tolist()])

    return apps.SolveAffine(np.array(landmarks))


def ConvertGrid(I_src_grid, I_tar_grid):
    '''Given a source grid and target grid, returns a grid for the source that is in the world coordinates of the target grid'''
    nS = np.multiply(np.divide(I_tar_grid.size().tolist(), [float(i) for i in I_src_grid.size().tolist()]),
                     I_tar_grid.spacing().tolist())
    return cc.MakeGrid(I_src_grid.size(), ca.Vec3Df(nS[0], nS[1], nS[2]), 'center')


def bb_grid_solver(source_image, affine):
    in_sz = source_image.size().tolist()
    in_sp = source_image.spacing().tolist()
    in_or = source_image.origin().tolist()

    C1temp = in_or[0:2]
    C1temp.append(1)
    C1 = C1temp
    C2 = np.array([in_sz[0] * in_sp[0] + in_or[0], in_or[1], 1])
    C3 = np.array([in_or[0], in_sz[1] * in_sp[1] + in_or[1], 1])
    C4 = np.array([in_sz[0] * in_sp[0] + in_or[0], in_sz[1] * in_sp[1] + in_or[1], 1])

    corners = np.matrix([C1, C2, C3, C4])
    tCorners = affine * corners.transpose()

    bbMax = np.max(tCorners[:, 0:4], 1)
    bbMin = np.min(tCorners[:, 0:4], 1)

    dim = np.ceil(bbMax - bbMin)

    out_sp = (np.squeeze(np.array(dim)) / source_image.size()[0:3])  # * (1/np.sqrt(2))
    if out_sp[2] == 0.0:
        out_sp[2] = 1.0
    out_sz = np.squeeze(np.array(dim)) / out_sp.transpose()
    out_or = np.squeeze(np.array(bbMin))  # Maybe needs to be the center of the image??

    grid = cc.MakeGrid([np.int(np.ceil(out_sz[0])), np.int(np.ceil(out_sz[1])), 1],
                       [out_sp[0], out_sp[1], 1],
                       [out_or[0], out_or[1], 0])

    return grid


def SubRegion(src_grid, sub_grid, def_grid):
    # The sapcing of the src image was changed, so we have to find the new spacing of the sub region
    reg_grid = ConvertGrid(src_grid, def_grid)
    spacing = (np.array(reg_grid.spacing().tolist()) / np.array(src_grid.spacing().tolist())) * np.array(
        sub_grid.spacing().tolist())

    # Because we updated the spacing, the origin of the sub region will change and we have to find the new origin
    # The new origin is the upper left real coordinate of the subregion to extract
    pixel_origin = np.array(sub_grid.origin().tolist()) / np.array(src_grid.spacing().tolist())
    upL_real = pixel_origin * np.array(reg_grid.spacing().tolist()) + np.array(reg_grid.origin().tolist())

    # Now need to find the bottom right in real coordinates
    btR_real = (spacing * np.array(sub_grid.size().tolist())) + upL_real

    # Now that we have the top left and bottom right, we need to convert them to index corrdiantes of the deformation grid
    upL = np.floor((upL_real - np.array(def_grid.origin().tolist())) / np.array(def_grid.spacing().tolist()))
    btR = np.floor((btR_real - np.array(def_grid.origin().tolist())) / np.array(def_grid.spacing().tolist()))

    xrng = [int(upL[0]), int(btR[0])]
    yrng = [int(upL[1]), int(btR[1])]

    new_sub_grid = cc.MakeGrid(ca.Vec3Di(sub_grid.size()[0], sub_grid.size()[1], 1),
                               ca.Vec3Df(spacing[0], spacing[1], 1), ca.Vec3Df(upL_real[0], upL_real[1], 0))

    return xrng, yrng, new_sub_grid


def main():
    secNum = sys.argv[1]
    mkyNum = sys.argv[2]
    region = str(sys.argv[3])
    # channel = sys.argv[3]
    ext = 'M{0}/section_{1}/{2}/'.format(mkyNum, secNum, region)
    ss_dir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/side_light_microscope/'
    conf_dir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/confocal/'
    memT = ca.MEM_DEVICE

    try:
        with open(ss_dir + 'src_registration/M{0}/section_{1}/M{0}_01_section_{1}_regions.txt'.format(mkyNum, secNum), 'r') as f:
            region_dict = json.load(f)
            f.close()
    except IOError:
    	region_dict = {}
    	region_dict[region] = {}
    	region_dict['size'] = map(int, raw_input("What is the size of the full resolution image x,y? ").split(','))
        region_dict[region]['bbx'] = map(int, raw_input("What are the x indicies of the bounding box (Matlab Format x_start,x_stop? ").split(','))
        region_dict[region]['bby'] = map(int, raw_input("What are the y indicies of the bounding box (Matlab Format y_start,y_stop? ").split(','))

    if region not in region_dict:
    	region_dict[region] = {}
        region_dict[region]['bbx'] = map(int, raw_input("What are the x indicies of the bounding box (Matlab Format x_start,x_stop? ").split(','))
        region_dict[region]['bby'] = map(int, raw_input("What are the y indicies of the bounding box (Matlab Format y_start,y_stop? ").split(','))

    img_region = common.LoadITKImage(ss_dir + 'src_registration/M{0}/section_{1}/M{0}_01_section_{1}_{2}.tiff'.format(mkyNum, secNum, region), ca.MEM_HOST)
    ssiSrc = common.LoadITKImage(ss_dir + 'src_registration/M{0}/section_{1}/frag0/M{0}_01_ssi_section_{1}_frag0.nrrd'.format(mkyNum, secNum), ca.MEM_HOST)
    bfi_df = common.LoadITKField(ss_dir + 'Blockface_registered/M{0}/section_{1}/frag0/M{0}_01_ssi_section_{1}_frag0_to_bfi_real.mha'.format(mkyNum, secNum), ca.MEM_DEVICE)

    # Figure out the same region in the low resolution image: There is a transpose from here to matlab so dimensions are flipped
    low_sz = ssiSrc.size().tolist()
    yrng_raw = [(low_sz[1] * region_dict[region]['bbx'][0]) / np.float(region_dict['size'][0]), (low_sz[1] * region_dict[region]['bbx'][1]) / np.float(region_dict['size'][0])]
    xrng_raw = [(low_sz[0] * region_dict[region]['bby'][0]) / np.float(region_dict['size'][1]), (low_sz[0] * region_dict[region]['bby'][1]) / np.float(region_dict['size'][1])]
    yrng = [np.int(np.floor(yrng_raw[0])), np.int(np.ceil(yrng_raw[1]))]
    xrng = [np.int(np.floor(xrng_raw[0])), np.int(np.ceil(xrng_raw[1]))]
    low_sub = cc.SubVol(ssiSrc, xrng, yrng) 

    # Figure out the grid for the sub region in relation to the sidescape
    originout = [ssiSrc.origin().x + ssiSrc.spacing().x*xrng[0],
                 ssiSrc.origin().y + ssiSrc.spacing().y*yrng[0],
                 0]
    spacingout = [(low_sub.size().x * ssiSrc.spacing().x) / (img_region.size().x),
                  (low_sub.size().y * ssiSrc.spacing().y) / (img_region.size().y),
                  1]

    gridout = cc.MakeGrid(img_region.size().tolist(), spacingout, originout)
    img_region.setGrid(gridout)
        
    only_sub = np.zeros(ssiSrc.size().tolist()[0:2])
    only_sub[xrng[0]:xrng[1], yrng[0]:yrng[1]] = np.squeeze(low_sub.asnp())
    only_sub = common.ImFromNPArr(only_sub)
    only_sub.setGrid(ssiSrc.grid())

    # Deform the only sub region to 
    only_sub.toType(ca.MEM_DEVICE)
    def_sub = ca.Image3D(bfi_df.grid(), bfi_df.memType())
    cc.ApplyHReal(def_sub, only_sub, bfi_df)
    def_sub.toType(ca.MEM_HOST)

    # Now have to find the bounding box in the deformation space (bfi space)
    if 'deformation_bbx' not in region_dict[region]:
    	bb_def = np.squeeze(pp.LandmarkPicker([np.squeeze(def_sub.asnp())]))
    	bb_def_y = [bb_def[0][0], bb_def[1][0]]
    	bb_def_x = [bb_def[0][1], bb_def[1][1]]
    	region_dict[region]['deformation_bbx'] = bb_def_x
    	region_dict[region]['deformation_bby'] = bb_def_y

    with open(ss_dir + 'src_registration/M{0}/section_{1}/M{0}_01_section_{1}_regions.txt'.format(mkyNum, secNum), 'w') as f:
       json.dump(region_dict, f)
       f.close()

    # Now need to extract the region and create a deformation and image that have the same resolution as the img_region
    deform_sub = cc.SubVol(bfi_df, region_dict[region]['deformation_bbx'], region_dict[region]['deformation_bby'])

    common.DebugHere()
    sizeout = [int(np.ceil((deform_sub.size().x * deform_sub.spacing().x) / img_region.spacing().x)),
               int(np.ceil((deform_sub.size().y * deform_sub.spacing().y) / img_region.spacing().y)),
               1]

    region_grid = cc.MakeGrid(sizeout, img_region.spacing().tolist(), deform_sub.origin().tolist())

    def_im_region = ca.Image3D(region_grid, deform_sub.memType())
    up_deformation = ca.Field3D(region_grid, deform_sub.memType())

    img_region.toType(ca.MEM_DEVICE)
    cc.ResampleWorld(up_deformation, deform_sub, ca.BACKGROUND_STRATEGY_PARTIAL_ZERO)
    cc.ApplyHReal(def_im_region, img_region, up_deformation)

    ss_out = ss_dir + 'Blockface_registered/M{0}/section_{1}/{2}/'.format(mkyNum, secNum, region)

    if not pth.exists(pth.expanduser(ss_out)):
    	os.mkdir(pth.expanduser(ss_out))

    common.SaveITKImage(def_im_region, pth.expanduser(ss_out) + 'M{0}_01_section_{1}_{2}_def_to_bfi.nrrd'.format(mkyNum, secNum, region))
    common.SaveITKImage(def_im_region, pth.expanduser(ss_out) + 'M{0}_01_section_{1}_{2}_def_to_bfi.tiff'.format(mkyNum, secNum, region))
    del img_region, def_im_region, ssiSrc, deform_sub

    # Now apply the same deformation to the confocal images
    conf_grid = cc.LoadGrid(conf_dir + 'sidelight_registered/M{0}/section_{1}/{2}/affine_registration_grid.txt'.format(mkyNum, secNum, region))
    cf_out = conf_dir + 'blockface_registered/M{0}/section_{1}/{2}/'.format(mkyNum, secNum, region)
    # confocal.toType(ca.MEM_DEVICE)
    # def_conf = ca.Image3D(region_grid, deform_sub.memType())
    # cc.ApplyHReal(def_conf, confocal, up_deformation)
    
    for channel in range(0, 4):
        z_stack = []
        num_slices = len(glob.glob(conf_dir + 'sidelight_registered/M{0}/section_{1}/{3}/Ch{2}/*.tiff'.format(mkyNum, secNum, channel, region)))
        for z in range(0, num_slices):
    	    src_im = common.LoadITKImage(conf_dir + 'sidelight_registered/M{0}/section_{1}/{3}/Ch{2}/M{0}_01_section_{1}_LGN_RHS_Ch{2}_conf_aff_sidelight_z{4}.tiff'.format(mkyNum, secNum, channel, region, str(z).zfill(2)))
    	    src_im.setGrid(cc.MakeGrid(ca.Vec3Di(conf_grid.size().x, conf_grid.size().y, 1), conf_grid.spacing(), conf_grid.origin()))
    	    src_im.toType(ca.MEM_DEVICE)
    	    def_im = ca.Image3D(region_grid, ca.MEM_DEVICE)
    	    cc.ApplyHReal(def_im, src_im, up_deformation)
    	    def_im.toType(ca.MEM_HOST)
    	    common.SaveITKImage(def_im, cf_out + 'Ch{2}/M{0}_01_section_{1}_{3}_Ch{2}_conf_def_blockface_z{4}.tiff'.format(mkyNum, secNum, channel, region, str(z).zfill(2)))
            if z==0:
            	common.SaveITKImage(def_im, cf_out + 'Ch{2}/M{0}_01_section_{1}_{3}_Ch{2}_conf_def_blockface_z{4}.nrrd'.format(mkyNum, secNum, channel, region, str(z).zfill(2)))
            z_stack.append(def_im)
            print('==> Done with Ch {0}: {1}/{2}'.format(channel, z, num_slices - 1))
        stacked = cc.Imlist_to_Im(z_stack)
        stacked.setSpacing(ca.Vec3Df(region_grid.spacing().x, region_grid.spacing().y, conf_grid.spacing().z))
        common.SaveITKImage(stacked, cf_out + 'Ch{2}/M{0}_01_section_{1}_{3}_Ch{2}_conf_def_blockface_stack.nrrd'.format(mkyNum, secNum, channel, region))
        if channel == 0:
        	cc.WriteGrid(stacked.grid(), cf_out + 'deformed_registration_grid.txt'.format(mkyNum, secNum, region))


if __name__ == '__main__':
    main()
