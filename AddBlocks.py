'''combines all the block#asMRI.mha's '''
import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCA.Core as ca
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import sys

# reg_type = 'landmark'
# reg_type = 've_reg'
# reg_type = 'BFI_2D_Reg'
reg_type = 'spline'
# reg_type = 'best'

col = 'rgb'                      # 'rgb', 'bw', or 've'
# sz = 512
sz = 1024

#############################

imagedir = '/home/sci/crottman/korenberg/results/' + reg_type + '/'
if reg_type in ['landmark', 've_reg']:
    fname_end = '_as_MRI_' + col + '.mha'
else:
    fname_end = '_as_MRI_' + col + '_' + str(sz) + '.mha'

if sz >= 512:
    mType = ca.MEM_HOST
else:
    mType = ca.MEM_DEVICE

if reg_type is not 'best':
    grid = cc.MakeGrid([sz, sz, sz], [256.0/sz, 256.0/sz, 256.0/sz], 'center')
    blocks = ca.Field3D(grid, mType)
    ca.SetMem(blocks, 0.0)
    weights = blocks.copy()
    ca.SetMem(weights, 0.0)

    for i in xrange(1, 5):
        fname = imagedir + 'block' + str(i) + fname_end
        try:
            blk = cc.LoadMHA(fname, mType)
        except IOError:
            print 'Warning... block ' + str(i) + ' does not exist'
            continue
        blocks += blk
        weight3 = blk.copy()
        try:
            weight = cc.LoadMHA(imagedir + 'block{0}_as_MRI_weight_{1}.mha'.format(i,sz))
        except IOError:
            print 'Warning, weight block does not exist'
            weight = ca.Image3D(blk.grid(), blk.memType())
            ca.Copy(weight, blk, 0)  # take red
            cc.SetRegionGTE(weight, weight, .1, 1)
        for i in xrange(3):
            ca.Copy(weight3, weight, i)
        weights += weight3
        print ca.MinMax(weights)

    for i in xrange(3):
        ca.Copy(weight, weights, i)
        cc.SetRegionLT(weight, weight, 1, 1)
        ca.Copy(weights, weight, i)
    print ca.MinMax(weights)

    ca.Div_I(blocks, weights)

else:                           # best
    imagedir = '/home/sci/crottman/korenberg/results/'
    blocks = cc.LoadMHA(imagedir + 'BFI_2D_Reg/block1_as_MRI_' + col + '_256.mha', ca.MEM_HOST)
    blk = cc.LoadMHA(imagedir + 'BFI_2D_Reg/block2_as_MRI_' + col + '_256.mha', ca.MEM_HOST)
    blocks += blk
    blk = cc.LoadMHA(imagedir + 'landmark/block3_as_MRI_' + col + '.mha', ca.MEM_HOST)
    blocks += blk
    blk = cc.LoadMHA(imagedir + 'landmark/block4_as_MRI_' + col + '.mha', ca.MEM_HOST)
    blocks += blk
    imagedir = '/home/sci/crottman/korenberg/results/best/'

# Save Blocks

maxval = ca.MinMax(blocks)[1]

blocks /= maxval

fname_out = imagedir + 'blocks' + fname_end
cc.WriteMHA(blocks, fname_out)
if col == 'rgb':
    fname_out = imagedir + 'blocks_as_MRI_rgba_' + str(sz) + '.mha'
    cc.WriteColorMHA(blocks, fname_out)
