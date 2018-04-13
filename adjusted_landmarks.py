'''We are trying to come up with better landmarks for blocks 3 and 4.
We previously solved for L such that gives the following approximation
for each landmark pairs {x_1, y}:

L x_1 \approx y

Here, we find new landmarks in the new space

x*_1 \approx y

We then solve for what the *old* land marks should have been,
generating the new landmark pairs:
{L^T x*_1, y}
'''

import PyCACalebExtras.SetBackend
plt = PyCACalebExtras.SetBackend.SetBackend()
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
# import matplotlib.pyplot as plt
# cc.SelectGPU()
plt.ion()
plt.close('all')

block = 4

# copy in original value for L
if block == 3:
    L = np.array([[ -1.42920637e-03,   3.44344604e-03,   1.10900629e-03,  -5.69603398e+00],
                  [ -3.97277738e-04,   5.49000394e-04,   5.36922451e-03,  -1.50456241e+01],
                  [  3.45590047e-03,   1.16899799e-03,  -4.63752168e-04,   2.48173995e+00],
                  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
elif block == 4:                # this is the original affine
    L = np.array([[ -1.28682464e-03,   3.33338223e-03,  -1.23943699e-04,  -7.62900313e+00],
                  [ -1.56979553e-04,  -1.15843364e-04,   3.85013822e-03,   4.63150436e+01],
                  [  3.17062856e-03,   1.40563728e-03,   8.45825397e-05,   6.60237793e+00],
                  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

if block == 3:
    pass
    # lms = list([[ , , ], [ , , ]],
    #            [[ , , ], [ , , ]],
    #        )

elif block == 4:
    # [[ , , ], [ , , ]],
    # lms = [[[-47.5, 6.5, -30.2], [-49.0, 8.5, -34.7]],
    #        [[-0.5, 6.5, 56.4], [-7.2, 8.5, 54.2]],
    #        [[-76.4, 49.5, -35.5], [-71.0, 49.5, -37.5]],
    #        [[-17.4, 43.9, 7.5], [-19.0, 41.9, 6.5]],
    #        [[61.9, 6.4, 27.5], [62.5, 14.2, 31.5]],
    #        [[22.7, 29.4, -55.5], [27.5, 34.7, -46.5]],
    #        [[28.7, 57.3, 40.5], [23.1, 59.2, 35.5]],
    #        [[-30.3, 41.3, 30.5], [-31.4, 38.6, 31.5]],
    #        [[32.6, 3.5, -47.8], [30.0, 2.5, -46.9]],
    #        [[4.8, 3.5, -4.5], [5.5, 2.5, -3.1]],
    #        [[49.5, 8.5, 44.5], [47.0, 13.5, 49.5]],
    #        [[5.4, 4.5, -5.2], [5.9, 2.5, -3.6]],
    #        [[-48.2, 7.5, -28.4], [-49.8, 7.5, -31.7]],
    #        [[41.5, 67.9, 0.5], [44.3, 64.0, -6.5]],
    #        [[-14.2, 57.4, -39.5], [-15.9, 53.8, -37.5]],
    #        [[-24.1, 85.8, -34.5], [-26.4, 82.5, -30.5]],
    #        [[-34.1, 51.5, 16.6], [-32.7, 48.5, 18.8]],
    #        [[-56.2, 59.5, -24.8], [-48.7, 62.5, -21.5]],
    #        [[-16.4, 41.5, 9.1], [-17.8, 40.5, 9.4]],
    #        [[27.7, 55.5, 57.7], [18.9, 51.5, 58.5]],
    #    ]
    # These are properly reversed!
    lms = [[[1.4, 43.5, -24.3], [20.6, 44.5, 8.6]],
           [[33.4, 55.5, 43.3], [-19.2, 52.5, -50.8]],
           [[-32.3, 47.5, 16.0], [-31.0, 45.5, 17.4]],
           [[25.6, 76.5, 16.0], [1.5, 75.5, -27.6]],
           [[5.8, 76.5, -28.1], [23.9, 75.5, 5.5]],
           [[7.7, 86.5, 38.9], [-31.3, 84.5, -25.4]],
           [[24.8, 88.5, 5.5], [8.4, 85.5, -20.4]],
           [[-24.8, 49.5, 55.2], [-59.7, 49.5, -11.3]],
           [[-59.9, 52.5, -5.2], [-28.6, 48.5, 54.1]],
           [[21.8, -4.5, 59.3], [-39.3, -3.5, -53.8]],
       ]

print "new landmarks for block ", block
lmxform =[[list(np.dot(np.linalg.inv(L), [lmpair[0][0], lmpair[0][1], lmpair[0][2], 1])[0:3]),
          lmpair[1]] for lmpair in lms]
# print lmxform
for lmpair in lmxform:
    print str(lmpair) + ','
