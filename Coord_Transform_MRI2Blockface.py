import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import pickle
plt.ion()
plt.close('all')
from numpy import linalg
blocks = [1,2,3,4]


caleb_dir = '/home/sci/crottman/korenberg/'
blake_dir = '/home/sci/blakez/monkey13/results/blockface/'
#for block in blocks:
#    defMat = np.load(blake_dir+'block{0}/MRI_to_block{0}_def.npy'.format(block))
#    grid = []
#    with open(caleb_dir+'results/blockface/block{0}_grid_hd1.txt'.format(block))

defMat1=np.load('/home/sci/blakez/monkey13/results/blockface/block1/MRI_to_block1_def.npy')
defMat2=np.load('/home/sci/blakez/monkey13/results/blockface/block2/MRI_to_block2_def.npy')
defMat3=np.load('/home/sci/blakez/monkey13/results/blockface/block3/MRI_to_block3_def.npy')
defMat4=np.load('/home/sci/blakez/monkey13/results/blockface/block4/MRI_to_block4_def_256_idiff.npy')

block1 = np.array([[-28021.5, -28021.5, -8760.0], [117, 117, 1.0]])
block2 = np.array([[-28021.5, -28021.5, -6270.0], [117, 117, 1.0]])
block3 = np.array([[-28021.5, -28021.5, -2460.0], [117, 117, 1.0]])
block4 = np.array([[-28072.6875, -28072.6875, -13095.0], [14.625, 14.625, 1.0]])

MRI_vol = np.load('/home/sci/blakez/monkey13/data/MRI/MRI_T2Seg_np.npy')

rigid = pickle.load(open("photo_rigid_transformations_4.pkl",'r'))


def DispVol(X):

    '''Display 3D numpy Volume'''

    class IndexTracker(object):
        '''Keeps track of the current image being displayed'''
        def __init__(self,X,ax):
            self.ax = ax
            self.X = X.transpose(2,1,0)
            self.slices = X.shape[1]
            self.ind = 190
            self.im = ax.imshow(X[:,self.ind,:],cmap='gray')
            self.update()
            plt.gca().invert_yaxis()
            
        def onKeyPress(self,event):
            '''Defines what happens upon pressing up or down arrow'''
            if event.key == 'w':
                self.ind = np.clip(self.ind+1, 0, self.slices-1)
            elif event.key == 'e':
                self.ind = np.clip(self.ind-1, 0, self.slices-1)
            self.update()

        def onclick(self,event):
            print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                event.button, event.x, event.y, event.xdata, event.ydata)
            xidx = int(round(event.xdata))
            yidx = int(self.ind)
            zidx = int(round(event.ydata))
            print 'Chosen X Coordinate: '+str(xidx)
            print 'Chosen Y Coordinate: '+str(yidx)
            print 'Chosen Z Coordinate: '+str(zidx)
            cords1 = defMat1[xidx,yidx,zidx]
            cords2 = defMat2[xidx,yidx,zidx]
            cords3 = defMat3[xidx,yidx,zidx]
            cords4 = defMat4[xidx,yidx,zidx]
            cords1idx = (cords1-block1[0])/block1[1]
            cords2idx = (cords2-block2[0])/block2[1]
            cords3idx = (cords3-block3[0])/block3[1]
            cords4idx = (cords4-block4[0])/block4[1]
            tempz = round(cords4idx[2]/30)
            z_dist = int(30*tempz)
#            print 'z dist = '+str(z_dist)
            affine = linalg.inv(rigid[z_dist])
            affine[0:2,2]*=8
            cords4idx[2] = 1.0
            final_xy_cords = np.inner(affine,cords4idx)
            final_cords = final_xy_cords
            final_cords[0,2] = z_dist
            filename = caleb_dir + 'data/photo/seg_high_res_crop/color/block4/DDP_Seg_'+str(z_dist)+'.png'
            image = im.imread(filename)
#            array = np.array(image)
            image[final_cords[0, 0]-50:final_cords[0, 0]+50, final_cords[0, 1]-50:final_cords[0, 1]+50, 2] = 10
            plt.figure(2)
            plt.imshow(np.rot90(image,3))
            
        def update(self):
            '''Update the current slice'''
            self.im.set_data(self.X[:,self.ind,:])
            ax.set_title('Slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tracker = IndexTracker(X,ax)
    fig.canvas.mpl_connect('key_press_event',tracker.onKeyPress)
    fig.canvas.mpl_connect('button_press_event',tracker.onclick)
    plt.show()
    return tracker

temp = DispVol(MRI_vol)


    # Load Rigid Transformations


 
    # plt.plot([final_cords[0,0],final_cords[0,0]],[0, len(yarray)-1], 'r-', lw=1) # 
    # plt.plot([0, len(xarray)-1],[final_cords[0,1], final_cords[0,1]], 'r-', lw=1)
    #axes = plt.gca()
    #axes.set_xlim([0,len(xarray)-1])
    #axes.set_ylim([0,len(yarray)-1])
