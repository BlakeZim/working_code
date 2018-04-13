import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
import PyCAApps as apps
# import matplotlib.pyplot as plt
# cc.SelectGPU()
plt.ion()
plt.close('all')

import json
import sys
import csv
import scipy.signal as sig
import os.path
import pickle
import ast

    


# Ask for the monkey number and the slice number within the block
monkeyNum = int(raw_input('Enter the monkey number: '))
slicestart = int(raw_input('Enter the starting slice of the stack: '))
slicefinish = int(raw_input('Enter the ending slice of the stack: '))

out_path = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/side_light_microscope/stacks/M{0}/'.format(monkeyNum)

ssi_path = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/side_light_microscope/Blockface_registered/M{0}/'.format(monkeyNum)
bfi_path = '/home/sci/blakez/korenbergNAS/3D_database/Working/Blockface/src_registration/M{0}/'.format(monkeyNum)

reg_list = []
aff_list = []
bfi_list = []

for sec in range(slicestart, slicefinish + 1):

    ssi_aff = common.LoadITKImage(ssi_path + 'section_{1}/frag0/M{0}_01_ssi_section_{1}_frag0_aff_bfi.nrrd'.format(monkeyNum, str(sec).zfill(4)), ca.MEM_HOST)
    ssi_reg = common.LoadITKImage(ssi_path + 'section_{1}/frag0/M{0}_01_ssi_section_{1}_frag0_def_bfi.nrrd'.format(monkeyNum, str(sec).zfill(4)), ca.MEM_HOST)
    bfi_org = common.LoadITKImage(bfi_path + 'section_{1}/M{0}_01_slice{1}_seg_crop_hd1.mha'.format(monkeyNum, str(sec).zfill(4)), ca.MEM_HOST)

    bfi_list.append(bfi_org)

    if sec==slicestart:
        inplane_grid = ssi_reg.grid()
        reg_list.append(ssi_reg)
        aff_list.append(ssi_aff)
        print "Registered Grid: "
        print ssi_reg.grid()
        print "Affine Grid: "
        print ssi_aff.grid()
        print "Added Section {0}".format(sec)
    else:
        stack_temp = ca.Image3D(inplane_grid, ca.MEM_HOST)
        cc.ResampleWorld(stack_temp, ssi_reg)
        reg_list.append(stack_temp)

        aff_list.append(ssi_aff)
        print "Added Section {0}".format(sec)

print 'Attempting to make volumes'
ssi_reg_vol = cc.Imlist_to_Im(reg_list)
ssi_aff_vol = cc.Imlist_to_Im(aff_list)
bfi_org_vol = cc.Imlist_to_Im(bfi_list)

ssi_reg_vol.setSpacing(ca.Vec3Df(ssi_reg_vol.spacing()[0], ssi_reg_vol.spacing()[1], 0.030))
ssi_aff_vol.setSpacing(ca.Vec3Df(ssi_aff_vol.spacing()[0], ssi_aff_vol.spacing()[1], 0.030))
bfi_org_vol.setSpacing(ca.Vec3Df(bfi_org_vol.spacing()[0], bfi_org_vol.spacing()[1], 0.030))

print 'Volumes Created and Attempting to Save'

common.SaveITKImage(ssi_reg_vol, out_path + 'M{0}_01_section_{1}_to_section_{2}_reg_ssi_stack.nrrd'.format(monkeyNum, slicestart, slicefinish)) 
common.SaveITKImage(ssi_aff_vol, out_path + 'M{0}_01_section_{1}_to_section_{2}_aff_ssi_stack.nrrd'.format(monkeyNum, slicestart, slicefinish)) 
common.SaveITKImage(bfi_org_vol, out_path + 'M{0}_01_section_{1}_to_section_{2}_org_bfi_stack.nrrd'.format(monkeyNum, slicestart, slicefinish)) 


