import numpy as np
import matplotlib.pyplot as plt
import Tkinter as tk
#import matplotlib.image as im
#import pickle
plt.ion()
plt.close('all')
import json
import math
from numpy import linalg
import argparse
from scipy.interpolate import griddata
#import sys

### def surr(x,y,zSlice):

###     upL = zSlice[math.floor(x),math.floor(y)] 
###     btL = zSlice[math.floor(x),math.ceil(y)] 
###     upR = zSlice[math.ceil(x),math.floor(y)] 
###     btR = zSlice[math.ceil(x),math.ceil(y)] 
###     subGrid = np.array([[upL,upR],[btL,btR]])
###     return subGrid

parser = argparse.ArgumentParser(description='Point Transformation')
parser.add_argument('-Block',action='store',type=int,
                    help='Input the block number in which the points were chosen',required=False)
parser.add_argument('-Points', nargs='*', action='store', type=float,
                    help='Input the coordinates of the point you want transformed (x,y,z)',required=False)
parser.add_argument('-f',dest='point_file',action='store',type=str,
                    help='Input the path to the text file containing the source points',required=False)
parser.add_argument('-i',dest='deform_dir',action='store',type=str,
                    help='Input the path to the directory containing the deformations',required=False)
parser.add_argument('-o',dest='output_file',action='store',type=str,
                    help='Input the path and name of the output text file',required=False)

args = parser.parse_args()
blockNum= args.Block


class GetPoints(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.wm_title('Transform Points')
        tk.Label(self, text='Block Number (REQUIRED):').grid(row=0,column=1)
        self.BlockNum = tk.Entry(self,bd=3)
        tk.Label(self, text='X Coordinate (1-3841):').grid(row=1,padx=5)
        self.entry1 = tk.Entry(self,bd=3)
        tk.Label(self, text='Y Coordinate (1-3841):').grid(row=2,padx=5)
        self.entry2 = tk.Entry(self,bd=3)
        tk.Label(self, text='Z Coordinate (1-zBlock):').grid(row=3,padx=5)
        self.entry3 = tk.Entry(self,bd=3)
        tk.Label(self, text='Output X Coordinate: ').grid(row=1,column=2)
        self.outputX = tk.Text(self,width=20, height=1,bd=3)
        tk.Label(self, text='Output Y Coordinate: ').grid(row=2,column=2)
        self.outputY = tk.Text(self,width=20, height=1,bd=3)
        tk.Label(self, text='Output Z Coordinate: ').grid(row=3,column=2)
        self.outputZ = tk.Text(self,width=20, height=1,bd=3)
        tk.Label(self, text='Path to text file of points (/path/to/input/file.txt):',
                 anchor='w').grid(row=5,ipady=1,columnspan=2)
        self.inPath = tk.Entry(self,bd=3,width=62)
        tk.Label(self, text='Path to the output text file (/path/to/output/file.txt):',
                 anchor='w').grid(row=8,ipady=1,columnspan=2)
        self.outPath = tk.Entry(self,bd=3,width=62)


        self.button = tk.Button(self, text='Transform Single Point', 
                                command=self.point_button).grid(row=4,column=1,columnspan=2,pady=15)
        self.esc = tk.Button(self, text='EXIT', command=self.esc).grid(row=0,column=0)
        self.txtFile = tk.Button(self, text='Transform List of Points', 
                                 command=self.list_button).grid(row=10,column=1,columnspan=2,pady=15)

        self.BlockNum.grid(row=0,column=2,columnspan=1,pady=15)
        self.entry1.grid(row=1,column=1)
        self.entry2.grid(row=2,column=1)        
        self.entry3.grid(row=3,column=1)
        self.outputX.grid(row=1,column=3)
        self.outputY.grid(row=2,column=3)
        self.outputZ.grid(row=3,column=3)
        self.inPath.grid(row=7,columnspan=3)
        self.outPath.grid(row=9,columnspan=3)

        
    def esc(self):
        self.destroy()
        
    def surr(self,x,y,zSlice):
        upL = zSlice[math.floor(x),math.floor(y)] 
        btL = zSlice[math.floor(x),math.ceil(y)] 
        upR = zSlice[math.ceil(x),math.floor(y)] 
        btR = zSlice[math.ceil(x),math.ceil(y)] 
        subGrid = np.array([[upL,upR],[btL,btR]])
        return subGrid

    def point_button(self):
        xcord = self.entry1.get() 
        ycord = self.entry2.get()
        zcord = self.entry3.get()
        blockNum = self.BlockNum.get()
        hField = np.load(
            '/sci/blakez/M13_01/results/MRI/fullReg/block{0}/M13_01_B{0}_MRI_to_BF_full_hField_hd8.npy'
            .format(blockNum))
        indexZcord = int(float(zcord)-1)
        indexXcord = int((float(xcord)-1)/8.0)
        indexYcord = int((float(ycord)-1)/8.0)
        zSlice = hField[:,:,indexZcord,:]
        subGrid = self.surr(indexXcord,indexYcord,zSlice)
        px, py, pz = np.mgrid[0:2,0:2,0:3]
        gx, gy, gz = np.mgrid[0:1:8j,0:1:8j,0:3]
        upSubGrid = griddata((px.flatten(),py.flatten(),pz.flatten()),
                             subGrid.flatten(),(gx,gy,gz),method='linear')
        point = upSubGrid[8*(indexXcord-math.floor(indexXcord)),8*(indexYcord-math.floor(indexYcord)),:]
        A = [[-0.48480962,0,-0.87461971],[0,-1,0],[0.87461971,0,-0.48480962]]
        rotPoint = np.dot(A,point)
        rotPoint += (127.5+1)
        rotPoint *= 0.5
        rotPoint[1] = 128 - rotPoint[1]
        self.outputX.delete(1.0,tk.END)
        self.outputX.insert(tk.END,str(rotPoint[0]))
        self.outputY.delete(1.0,tk.END)
        self.outputY.insert(tk.END,str(rotPoint[1]))
        self.outputZ.delete(1.0,tk.END)
        self.outputZ.insert(tk.END,str(rotPoint[2]))
        self.update()

    def list_button(self):
        inpath = self.inPath.get()
        outpath = self.outPath.get()
        blockNum = self.BlockNum.get()
        Points = np.loadtxt(inpath)
        print inpath
        print outpath
        hField = np.load(
            '/home/sci/blakez/M13_01/results/MRI/fullReg/block{0}/M13_01_B{0}_MRI_to_BF_full_hField_hd8.npy'
            .format(blockNum))
        finalPoints = []

        for x in range(0,Points.shape[0]):
            # We know that the z slice will stay constant as there is no downsampling in that dimension
            indexZcord = int(Points[x,2]-1)
            indexXcord = int((Points[x,0]-1)/8.0)
            indexYcord = int((Points[x,1]-1)/8.0)

            # Extract the slice of the deformation field
            zSlice = hField[:,:,indexZcord,:]
            
            # Find the known points around the given point
            subGrid = self.surr(indexXcord,indexYcord,zSlice)
            
            # Define the grid points where we know values, p_, and the grid points where we want to interpolate, g_
            px, py, pz = np.mgrid[0:2,0:2,0:3]
            gx, gy, gz = np.mgrid[0:1:8j,0:1:8j,0:3]
            
            # Interpolte with the known points to get the unknowns
            upSubGrid = griddata((px.flatten(),py.flatten(),pz.flatten()),
                                 subGrid.flatten(),(gx,gy,gz),method='linear')
            
            # Extract the desired point from the subgrid in index cords (add 1 if view index from 1:end, which most do)
            point = upSubGrid[8*(indexXcord-math.floor(indexXcord)),8*(indexYcord-math.floor(indexYcord)),:]
            A = [[-0.48480962,0,-0.87461971],[0,-1,0],[0.87461971,0,-0.48480962]]
            rotPoint = np.dot(A,point)
            rotPoint += (127.5+1)
            rotPoint *= 0.5
            rotPoint[1] = 128 - rotPoint[1]
            # Apply the rotation

            finalPoints.append(rotPoint)
        np.savetxt(outpath,np.array(finalPoints),fmt='%10.5f')
        print 'Points have been transformed, check the output directory.'

test = GetPoints()
test.mainloop()


### if args.point_file==None:
###     Points = np.matrix(args.Points)
### else:
###     # Load the points to be transformed
###     Points = np.loadtxt(args.point_file)
###     mriDir = args.deform_dir

# Define the loading directories
# mriDir = '/home/sci/blakez/M13_01/results/MRI/fullReg/block{0}/'.format(blockNum)

# Define the z_slice knowing the block number
### if blockNum == 1:
###     sizeZ = 585
### if blockNum == 2:
###     sizeZ = 419
### if blockNum == 3:
###     sizeZ = 165
### if blockNum == 4:
###     sizeZ = 874

# Load the full deformation from MRI to BF space
#hField = np.load(mriDir + 'block{0}/M13_01_B{0}_MRI_to_BF_full_hField_hd8.npy'.format(blockNum))
### hField = np.load('/home/sci/blakez/M13_01/results/MRI/fullReg/block{0}/M13_01_B{0}_MRI_to_BF_full_hField_hd8.npy'.format(blockNum))
### finalPoints = []

### for x in range(0,Points.shape[0]):
###     # We know that the z slice will stay constant as there is no downsampling in that dimension
###     indexZcord = int(Points[x,2]-1)
###     indexXcord = int((Points[x,0]-1)/8.0)
###     indexYcord = int((Points[x,1]-1)/8.0)

###     # Extract the slice of the deformation field
###     zSlice = hField[:,:,indexZcord,:]

###     # Find the known points around the given point
###     subGrid = surr(indexXcord,indexYcord,zSlice)

###     # Define the grid points where we know values, p_, and the grid points where we want to interpolate, g_
###     px, py, pz = np.mgrid[0:2,0:2,0:3]
###     gx, gy, gz = np.mgrid[0:1:8j,0:1:8j,0:3]
    
###     # Interpolte with the known points to get the unknowns
###     upSubGrid = griddata((px.flatten(),py.flatten(),pz.flatten()),subGrid.flatten(),(gx,gy,gz),method='linear')

###     # Extract the desired point from the subgrid in index cords (add 1 if view index from 1:end, which most do)
###     point = upSubGrid[8*(indexXcord-math.floor(indexXcord)),8*(indexYcord-math.floor(indexYcord)),:]
###     A = [[-0.48480962,0,-0.87461971],[0,-1,0],[0.87461971,0,-0.48480962]]
###     rotPoint = np.dot(A,point)
###     rotPoint += (127.5+1)
###     rotPoint *= 0.5
###     rotPoint[1] = 128 - rotPoint[1]
###     # Apply the rotation

###     finalPoints.append(rotPoint)

### if args.output_file==None:
###     print finalPoints
### else:
###     np.savetxt(args.output_file,np.array(finalPoints),fmt='%10.5f')
###     disp = np.array([Points,np.array(finalPoints)])
###     print disp
