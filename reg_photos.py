'''Finds the rigid registrations between blocks neiboring BFIs.
Registers using low res (default) B/W photos, then saves the
transformations in "photo_rigid_transformations_#.pkl"
'''

import PyCA.Core as ca

import matplotlib.pyplot as plt
plt.ion()   # tell it to use interactive mode -- see results immediately
import numpy as np

import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import RigidReg as rr
import get_file_list as files

isHD = False
block = 4                       # 1, 2, 3, 4

filelist = files.get_great_files(block-1)
if isHD:
    # mType=ca.MEM_HOST
    mType = ca.MEM_DEVICE
    ds = 2
else:
    mType = ca.MEM_DEVICE
    ds = 1

plt.close('all')

# Rigid Reg parameters
theta_step = 0
t_step = 300
a_step = 0
maxIter = 2000

Imprev = cc.LoadTIFF(filelist[0], mType, ds)
origin = [(Imprev.grid().size().x+1)/2.0, # origin for Affine matrix
          (Imprev.grid().size().y+1)/2.0,
          (Imprev.grid().size().z+1)/2.0]
scratchI = ca.Image3D(Imprev.grid(), Imprev.memType())
scratchI2 = ca.Image3D(Imprev.grid(), Imprev.memType())

# initialize dictionary
Adict = {'origin': origin}
Adict[files.get_file_dist(filelist[0])] = np.identity(3)

# if 'block1' in filelist[0]:
# move first image in block 1
if block == 1:
    tcentx, tcenty = cc.CenterImage(Imprev)
    ca.Copy(scratchI, Imprev)
    # first moves image up, second moves image left
    t = ca.Vec3Df(-75, 55, 0)   # double check this!
    ca.ComposeTranslation(Imprev, scratchI, t)
    ttot = [tcentx + t.x, tcenty + t.y]
    Adict[files.get_file_dist(filelist[0])] = np.array([[1, 0, tcentx + t.x],
                                                        [0, 1, tcenty + t.y],
                                                        [0, 0, 1]])

dist_prev = -30                 # assure correct numbering
for filename in filelist[1:]:
    dist = files.get_file_dist(filename)

    num_blanks = (dist - dist_prev)/30 - 1
    for _i in xrange(num_blanks):
        print 'blank'
    dist_prev = dist
    print dist

    Im = cc.LoadPNG(filename, mType, ds)
    # if 'block1' in filename:
    if block == 1:
        tcentx, tcenty = cc.CenterImage(Im, justReturnT=True)
        # ca.Copy(scratchI, Im)
        # ca.ComposeTranslation(Im, scratchI, t) # move centered image
        t_init = [tcentx+t.x, tcenty+t.y]
        if dist > 12300:
            theta = -.085
        else:
            theta = 0
    # elif 'block2' in filename:
    elif block == 2:
        t_init = None
        if dist > 9570:
            theta = .026
        else:
            theta = 0
    else:
        theta = 0
        t_init = None
    # theta = 0
    # plt.close('all')
    A = rr.RigidReg(Im, Imprev, theta_step=theta_step, t_step=t_step,
                    a_step=a_step, maxIter=maxIter, plot=False, theta=theta,
                    origin=origin, t=t_init)
    Imreg = ca.Image3D(Im.grid(), Im.memType())
    cc.ApplyAffine(Imreg, Im, A)
    Adict[dist] = A

    Imprev = Imreg

import pickle
fnameout = 'photo_rigid_transformations_' + str(block) + '.pkl'
f = open(fnameout, 'w')
print "saved", fnameout
#pickle.dump(Adict, f)
f.close()
