import numpy as np
from BFI_reg_landmarks import get_new_landmarks

# parameters
block = 4                       # 1, 2, 3, 4
# isHD = False

# Find Least Squares Affine Transformation
landmarks = get_new_landmarks(block)
print 'Finding the Affine Transformation for block', block
# if isHD:
#     ds = 2
# else:
#     ds = 8

n = len(landmarks)              # number of landmarks
Msub = np.ones((n, 4))
b = np.ones(n*3)
for i in xrange(n):
    Msub[i, 0:3] = landmarks[i][1][:]
    b[i] = landmarks[i][0][0]
    b[i+n] = landmarks[i][0][1]
    b[i+2*n] = landmarks[i][0][2]
M = np.zeros((n*3, 12))
M[0:n, 0:4] = Msub
M[n:2*n, 4:8] = Msub
M[2*n:3*n, 8:12] = Msub

if n == 4:
    z = np.linalg.solve(M, b)
else:
    z = np.linalg.lstsq(M, b)
    z = z[0]

# Note A: BFI -> MRI, Ainv: MRI -> BFI
Ainv = np.zeros((4, 4))
Ainv[0][:] = z[0:4]
Ainv[1][:] = z[4:8]
Ainv[2][:] = z[8:12]
Ainv[3][3] = 1
A = np.linalg.inv(Ainv)

for lm in landmarks:
    BFItoMRI = np.dot(A, lm[0] + [1.0])[0:3]
    MRI = lm[1]
    BFI = lm[0]
    MRItoBFI = np.dot(Ainv, lm[1] + [1.0])[0:3]
    print '[BFI to MRI, MRI]:', np.around(BFItoMRI, 2), MRI, 'error = ', \
        np.around(np.linalg.norm(BFItoMRI- MRI), 2)
    print '[BFI, MRI to BFI]:', np.around(BFI, 2), np.around(MRItoBFI, 2)

np.set_printoptions(linewidth=100)
print repr(A)

# print Ainv[0, 3], Ainv[1, 3], Ainv[2, 3]
