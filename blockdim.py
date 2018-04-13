import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
from PyCAApps import SolveSpline, SplineToHField
import sys
from BFI_reg_landmarks import get_new_landmarks
# import matplotlib.pyplot as plt
# cc.SelectGPU()
plt.ion()
plt.close('all')


BFIdir = '/home/sci/crottman/korenberg/results/blockface/'
MRIdir = '/home/sci/crottman/korenberg/data/MRI/'

for block in [1, 2, 3, 4]:
    grid = cc.LoadGrid(BFIdir + 'block{0}_grid.txt'.format(block))
    sz = grid.size().tolist()
    sp = grid.spacing().tolist()
    o = grid.origin().tolist()
    print 'block'+str(block), 'size:', sz
    print 'block'+str(block), 'bound:', \
        o, 'to', [(sz[0]-1)*sp[0]+o[0],
                  (sz[1]-1)*sp[1]+o[1],
                  (sz[2]-1)*sp[2]+o[2]]

grid = cc.LoadGrid(MRIdir + 'T2_grid.txt'.format(block))
sz = grid.size().tolist()
sp = grid.spacing().tolist()
o = grid.origin().tolist()
print 'MRI size:', sz
print 'MRI bound:', \
    o, 'to', [(sz[0]-1)*sp[0]+o[0],
              (sz[1]-1)*sp[1]+o[1],
              (sz[2]-1)*sp[2]+o[2]]
