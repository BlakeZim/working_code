'''
Functions given the file list of korenberg photography

For each block, the entier file list is split into 3 groups:
filelist = bad + moderate + great

good files = moderate + great

'''

import matplotlib.pyplot as plt
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import time

plt.ion()   # tell it to use interactive mode -- see results immediately

import glob
from subprocess import Popen, PIPE

plt.close('all')


def get_file_dist(fname):
    return int(fname[fname.rfind('_')+1: fname.rfind('.')])


def get_imset(hd=False, col='bw'):
    if hd is False:
        if col == 'bw':
            imdir = '/home/sci/crottman/korenberg/data/photo/seg_low_res_crop8/bw/'
        elif col == 'rgb':
            imdir = '/home/sci/crottman/korenberg/data/photo/seg_low_res_crop8/color/'
    else:
        if col == 'bw':
            imdir = '/home/sci/crottman/korenberg/data/photo/seg_high_res_crop/bw/'
        elif col == 'rgb':
            imdir = '/home/sci/crottman/korenberg/data/photo/seg_high_res_crop/color/'

    files = glob.glob(imdir + 'block1/' + '*.png')
    newfiles1 = sorted(files, key=lambda x: get_file_dist(x))

    files = glob.glob(imdir + 'block2/' + '*.png')
    newfiles2 = sorted(files, key=lambda x: get_file_dist(x))

    files = glob.glob(imdir + 'block3/' + '*.png')
    newfiles3 = sorted(files, key=lambda x: get_file_dist(x))

    files = glob.glob(imdir + 'block4/' + '*.png')
    newfiles4 = sorted(files, key=lambda x: get_file_dist(x))

    # Very Bad Files
    b1 = [
        '/DDP_Seg_660.png',
        '/DDP_Seg_12180.png',
        '/DDP_Seg_12210.png',
        '/DDP_Seg_12240.png',
        '/DDP_Seg_12270.png'
        ]
    b2 = [
        '/DDP_Seg_3030.png',
        ]
    b3 = [
        '/DDP_Seg_3300.png',
        ]
    b4 = [
        '/DDP_Seg_3030.png',
        '/DDP_Seg_3060.png',
        '/DDP_Seg_4560.png',
        '/DDP_Seg_15450.png',
        '/DDP_Seg_18420.png',
        '/DDP_Seg_22920.png',
        '/DDP_Seg_25530.png',           # wrong size
        '/DDP_Seg_5070.png',
        '/DDP_Seg_10200.png',   # out of order

        ]
    # Moderately bad files
    m1 = []
    m2 = [
        '/DDP_Seg_12570.png',
        '/DDP_Seg_12660.png',
        '/DDP_Seg_12690.png',
        '/DDP_Seg_12720.png',
        '/DDP_Seg_12750.png',
        '/DDP_Seg_12780.png',
        '/DDP_Seg_12810.png',
        '/DDP_Seg_12840.png',
        '/DDP_Seg_12870.png',
        '/DDP_Seg_12900.png',
        '/DDP_Seg_12930.png',
        '/DDP_Seg_12990.png',
        '/DDP_Seg_13020.png',
        '/DDP_Seg_13140.png',
        '/DDP_Seg_13200.png',
        '/DDP_Seg_13230.png',
        '/DDP_Seg_13260.png',
        '/DDP_Seg_13350.png',
        '/DDP_Seg_13380.png',
        '/DDP_Seg_13410.png',
        '/DDP_Seg_13440.png',
        '/DDP_Seg_13470.png',
        '/DDP_Seg_13500.png',
        '/DDP_Seg_13530.png',
        '/DDP_Seg_13590.png',
        '/DDP_Seg_13620.png',
        '/DDP_Seg_13680.png',
        '/DDP_Seg_13710.png',
        '/DDP_Seg_13740.png',
        '/DDP_Seg_13770.png'
        ]
    m3 = [
        # Ice damage:
        '/DDP_Seg_0.png',
        '/DDP_Seg_30.png',
        '/DDP_Seg_60.png',
        '/DDP_Seg_90.png',
        '/DDP_Seg_120.png',
        '/DDP_Seg_150.png',
        '/DDP_Seg_180.png',
        '/DDP_Seg_210.png',
        '/DDP_Seg_390.png',
        '/DDP_Seg_420.png',
        '/DDP_Seg_450.png',
        '/DDP_Seg_480.png',
        '/DDP_Seg_510.png',
        '/DDP_Seg_540.png',
        '/DDP_Seg_570.png',
        '/DDP_Seg_600.png',
        '/DDP_Seg_630.png',
        '/DDP_Seg_660.png',
        '/DDP_Seg_690.png',
        '/DDP_Seg_1440.png',
        '/DDP_Seg_1470.png',
        '/DDP_Seg_1500.png',
        '/DDP_Seg_1530.png',
        '/DDP_Seg_1560.png',
        '/DDP_Seg_1590.png',
        '/DDP_Seg_1620.png',

        '/DDP_Seg_240.png',
        '/DDP_Seg_270.png',
        '/DDP_Seg_300.png',
        '/DDP_Seg_330.png',
        '/DDP_Seg_360.png',
        '/DDP_Seg_720.png',
        '/DDP_Seg_750.png',
        '/DDP_Seg_780.png',
        '/DDP_Seg_810.png',
        '/DDP_Seg_840.png',
        '/DDP_Seg_870.png',
        '/DDP_Seg_900.png',
        '/DDP_Seg_930.png',
        '/DDP_Seg_960.png',
        '/DDP_Seg_990.png',
        '/DDP_Seg_1020.png',
        '/DDP_Seg_1050.png',
        '/DDP_Seg_1080.png',
        '/DDP_Seg_1110.png',
        '/DDP_Seg_1140.png',
        '/DDP_Seg_1170.png',
        '/DDP_Seg_1200.png',
        '/DDP_Seg_1230.png',
        '/DDP_Seg_1260.png',
        '/DDP_Seg_1290.png',
        '/DDP_Seg_1320.png',
        '/DDP_Seg_1350.png',
        '/DDP_Seg_1380.png',
        '/DDP_Seg_1410.png',
        '/DDP_Seg_1440.png',

        '/DDP_Seg_4950.png',
        '/DDP_Seg_4980.png',
        '/DDP_Seg_5010.png',
        '/DDP_Seg_5040.png',
        '/DDP_Seg_5070.png',
        '/DDP_Seg_5100.png',
        '/DDP_Seg_5130.png',
        '/DDP_Seg_5190.png',
        '/DDP_Seg_5220.png',
        '/DDP_Seg_5250.png',
        '/DDP_Seg_5280.png',
        '/DDP_Seg_5310.png',
        '/DDP_Seg_5340.png',
        '/DDP_Seg_5370.png',
        '/DDP_Seg_5400.png',
        '/DDP_Seg_5430.png',
        ]
    m4 = []
    imset = [[newfiles1, b1, m1],
             [newfiles2, b2, m2],
             [newfiles3, b3, m3],
             [newfiles4, b4, m4]]
    return imset

def get_all_good_files(hd=False, col='bw'):
    '''returns a list of all of the good (moderate + great) files'''
    return get_good_files(0, hd, col) + \
        get_good_files(1, hd, col) + \
        get_good_files(2, hd, col) + \
        get_good_files(3, hd, col)


def get_all_great_files(hd=False, col='bw'):
    '''returns a list of all of the great (neither moderate or poor)
    files'''
    return get_great_files(0, hd, col) + \
        get_great_files(1, hd, col) + \
        get_great_files(2, hd, col) + \
        get_great_files(3, hd, col)


def get_all_files(hd=False, col='bw'):
    '''returns a list of all (poor + moderate + great) files'''
    [[newfiles1, l1, m1],
     [newfiles2, l2, m2],
     [newfiles3, l3, m3],
     [newfiles4, l4, m4]] = get_imset(hd, col)
    return newfiles1+newfiles2+newfiles3+newfiles4


def get_bad_files(idx=0, hd=False, col='bw'):
    '''returns a list of all of the bad (neither moderate or great)
    files for a given set'''

    imset = get_imset(hd, col)
    filelist = imset[idx][0]
    badfilelist = imset[idx][1]

    badlist = []
    for badstr in badfilelist:
        badlist += [f for f in filelist if badstr in f]
    return badlist


def get_mod_files(idx=0, hd=False, col='bw'):
    '''returns a list of all of the moderate (neither poor or great)
    files for a given set'''

    imset = get_imset(hd, col)
    filelist = imset[idx][0]
    modfilelist = imset[idx][2]

    modlist = []
    for modstr in modfilelist:
        modlist += [f for f in filelist if modstr in f]
    return modlist


def get_good_files(idx=0, hd=False, col='bw'):
    '''returns a list of all of the good (moderate and great)
    files for a given set'''

    imset = get_imset(hd, col)
    filelist = imset[idx][0]
    badfilelist = imset[idx][1]

    goodlist = filelist[:]
    for badfile in badfilelist:
        goodlist = [f for f in goodlist if badfile not in f]
    return goodlist


def get_great_files(idx=0, hd=False, col='bw'):
    '''returns a list of all of the great (neither moderate or poor)
    files for a given set'''

    imset = get_imset(hd, col)
    filelist = imset[idx][0]
    badfilelist = imset[idx][1] + imset[idx][2]

    goodlist = filelist[:]
    for badfile in badfilelist:
        goodlist = [f for f in goodlist if badfile not in f]
    return goodlist


def show_files(filelist, sleeptime=.6):
    '''given a list of files, it shows all those images (Doesn't yet
    work for color images'''

    plt.figure()
    loadtimelist = []
    if type(filelist) is not list:
        filelist = [filelist]
    for i in xrange(len(filelist)):
        fname = filelist[i]
        t = time.time()
        name = fname[fname.rfind('/') : fname.rfind('.')]
        Im = cc.LoadTIFF(fname, ca.MEM_HOST)
        print name, i
        cd.DispImage(Im, newFig=False, title=name)
        p = Popen(['xsel', '-pi'], stdin=PIPE)
        p.communicate(input="'" + name + ".png'" + ',\n')
        loadtime = time.time() - t
        loadtimelist.append(loadtime)
        time.sleep(sleeptime)
    # plt.figure()
    # plt.plot(loadtimelist)


if __name__ == '__main__':
    # show_files(get_bad_files(3), .5)
    # show_files(get_good_files(3)[310:], 1)
    # print 'block1:', len(get_great_files(0)), ',', len(get_good_files(0)), 'out of', len(newfiles1)
    # print 'block2:', len(get_great_files(1)), ',', len(get_good_files(1)), 'out of', len(newfiles2)
    # print 'block3:', len(get_great_files(2)), ',', len(get_good_files(2)), 'out of', len(newfiles3)
    # print 'block4:', len(get_great_files(3)), ',', len(get_good_files(3)), 'out of', len(newfiles4)
    # show_files(get_great_files(3)[:], .1)
    # show_files(get_pretty_bad_files(1)[:], .5)
    # show_files(get_mod_files(2))
    # print get_great_files(0)
    show_files(get_bad_files(3))
