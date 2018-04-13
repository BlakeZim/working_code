import PyCA.Core as ca
import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend('tkagg')
import numpy as np
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Common as common
import PyCABlakeExtras.Common as cb
import PyCAApps as apps
import matplotlib
import scipy
from scipy import ndimage
plt.ion()
plt.close('all')
import glob
import sys
import gc
import os
import os.path as pth


def load_and_solve(target_points_file, source_points_file):

	target_points = np.loadtxt(target_points_file, delimiter=' ')[:,0:2]
	source_points = np.loadtxt(source_points_file, delimiter=' ')[:,0:2]

	landmarks = []
	for pt in range(0, len(target_points)):
		landmarks.append([source_points[pt].tolist(), target_points[pt].tolist()])

	return apps.SolveAffine(np.array(landmarks))


def bb_grid_solver(source_image, affine):

	in_sz = source_image.size().tolist()
	in_sp = source_image.spacing().tolist()
	in_or = source_image.origin().tolist()

	C1temp = in_or[0:2]
	C1temp.append(1)
	C1 = C1temp
	C2 = np.array([in_sz[0]*in_sp[0] + in_or[0], in_or[1], 1])
	C3 = np.array([in_or[0], in_sz[1]*in_sp[1] + in_or[1], 1])
	C4 = np.array([in_sz[0]*in_sp[0] + in_or[0], in_sz[1]*in_sp[1] + in_or[1], 1])

	corners = np.matrix([C1,C2,C3,C4])
	tCorners = affine*corners.transpose()

	bbMax = np.max(tCorners[:,0:4],1)
	bbMin = np.min(tCorners[:,0:4],1)

	dim = np.ceil(bbMax-bbMin)

	out_sp = (np.squeeze(np.array(dim))/source_image.size()[0:3]) #* (1/np.sqrt(2))
	if out_sp[2] == 0.0:
		out_sp[2] = 1.0
	out_sz = np.squeeze(np.array(dim))/out_sp.transpose()
	out_or = np.squeeze(np.array(bbMin)) # Maybe needs to be the center of the image??

	grid = cc.MakeGrid([np.int(np.ceil(out_sz[0])), np.int(np.ceil(out_sz[1])), 1],
					   [out_sp[0], out_sp[1], 1],
					   [out_or[0], out_or[1], 0])

	return grid


def main():

	secNum = sys.argv[1]
	mkyNum = sys.argv[2]
	channel = sys.argv[3]
	region = str(sys.argv[4])

	conf_dir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/confocal/src_registration/'
	side_dir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/side_light_microscope/src_registration/'
	save_dir = '/home/sci/blakez/korenbergNAS/3D_database/Working/Microscopic/confocal/sidelight_registered/'

	# DIC = '/home/sci/blakez/Reflect Affine/DIC_to_Reflect.txt'
	src_pt = conf_dir + 'M{0}/section_{1}/{2}/section_{1}_confocal_relation_with_sidelight.txt'.format(mkyNum, secNum, region)
	tar_pt = side_dir + 'M{0}/section_{1}/section_{1}_sidelight_relation_with_confocal.txt'.format(mkyNum, secNum)
	# SID = '/home/sci/blakez/Reflect Affine/sidelight_to_DIC.txt'

	src_im = common.LoadITKImage(conf_dir + 'M{0}/section_{1}/{3}/Ch{2}/M{0}_{1}_LGN_RHS_Ch{2}_z00.tif'.format(mkyNum, secNum, channel, region))
	# tar_im = common.LoadITKImage('M{0}/{1}/Crop_ThirdNerve_EGFP_z16.tiff'.format(mkyNum, secNum))

	# The points need to be chosen in the origin corrected sidescape for downstream purposes
	affine = load_and_solve(tar_pt, src_pt)
	out_grid = bb_grid_solver(src_im, affine)

	z_stack = []
	num_slices = len(glob.glob(conf_dir + 'M{0}/section_{1}/{3}/Ch{2}/*'.format(mkyNum, secNum, channel, region)))

	for z in range(0, num_slices):

		src_im = common.LoadITKImage(conf_dir + 'M{0}/section_{1}/{4}/Ch{2}/M{0}_{1}_LGN_RHS_Ch{2}_z{3}.tif'.format(mkyNum, secNum, channel, str(z).zfill(2), region))
		aff_im = ca.Image3D(out_grid, ca.MEM_HOST)
		cc.ApplyAffineReal(aff_im, src_im, affine)
		common.SaveITKImage(aff_im, save_dir + 'M{0}/section_{1}/{4}/Ch{2}/M{0}_01_section_{1}_LGN_RHS_Ch{2}_conf_aff_sidelight_z{3}.tiff'.format(mkyNum, secNum, channel, str(z).zfill(2), region))
		z_stack.append(aff_im)
		print('==> Done with {0}/{1}'.format(z, num_slices - 1))


	stacked = cc.Imlist_to_Im(z_stack)
	stacked.setSpacing(ca.Vec3Df(out_grid.spacing()[0], out_grid.spacing()[1], 0.03/num_slices))
	common.SaveITKImage(stacked, save_dir + 'M{0}/section_{1}/{3}/Ch{2}/M{0}_01_section_{1}_Ch{2}_conf_aff_sidelight_stack.nrrd'.format(mkyNum, secNum, channel, region))
	common.DebugHere()
	if channel==0:
		cc.WriteGrid(stacked.grid(), save_dir + 'M{0}/section_{1}/{2}/affine_registration_grid.txt'.format(mkyNum, secNum, region))


if __name__ == '__main__':
	main()
