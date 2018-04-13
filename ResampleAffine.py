import PyCA.Core as ca
import PyCACalebExtras.Common as cc
import numpy as np


def ApplyAffine(Iout, Im, A, bg=ca.BACKGROUND_STRATEGY_PARTIAL_ZERO):
    '''Applies an Affine matrix A to an image Im using the Image3D
    grid (size, spacing, origin) of the two images (Input and Output)

    '''
    # algorithm outline:  Create a temporary large grid, then perform
    # real affine transforms here, then crop to be the size of the out grid
    ca.SetMem(Iout, 0.0)

    A = np.matrix(A)

    bigsize = [max(Iout.grid().size().x, Im.grid().size().x),
               max(Iout.grid().size().y, Im.grid().size().y),
               max(Iout.grid().size().z, Im.grid().size().z)]
    idgrid = ca.GridInfo(ca.Vec3Di(bigsize[0], bigsize[1], bigsize[2]),
                         ca.Vec3Df(1, 1, 1),
                         ca.Vec3Df(0, 0, 0))
    # newgrid = Iout.grid()       # not a true copy!!!!!
    newgrid = ca.GridInfo(Iout.grid().size(),
                          Iout.grid().spacing(),
                          Iout.grid().origin())

    mType = Iout.memType()
    Imbig = cc.PadImage(Im, bigsize)
    h = ca.Field3D(idgrid, mType)
    ca.SetToIdentity(h)
    if isinstance(Im, ca.Field3D):
        Ioutbig = ca.Field3D(idgrid, mType)
    else:
        Ioutbig = ca.Image3D(idgrid, mType)

    # note:  x_real' = A*x_real; x_real' given (input grid)
    # solution: x_real = A^-1 * x_real
    # where x_real = x_index*spacing + origin
    # and x_real' = x_index'*spacing' + origin'
    # x_index' is really given, as is both spacings/origins
    # and we plug in the solution for x_index' into applyH

    if A.shape[1] == 3:          # 2D affine matrix
        x = ca.Image3D(idgrid, mType)
        y = ca.Image3D(idgrid, mType)
        xnew = ca.Image3D(idgrid, mType)
        ynew = ca.Image3D(idgrid, mType)
        ca.Copy(x, h, 0)
        ca.Copy(y, h, 1)

        # convert x,y to world coordinates
        x *= Iout.grid().spacing().x
        y *= Iout.grid().spacing().y
        x += Iout.grid().origin().x
        y += Iout.grid().origin().y

        # Matrix Multiply (All in real coords)
        Ainv = A.I
        ca.MulC_Add_MulC(xnew, x, Ainv[0, 0], y, Ainv[0, 1])
        ca.MulC_Add_MulC(ynew, x, Ainv[1, 0], y, Ainv[1, 1])
        xnew += (Ainv[0, 2])
        ynew += (Ainv[1, 2])     # xnew and ynew are now in real coords

        # convert back to index coordinates
        xnew -= Im.grid().origin().x
        ynew -= Im.grid().origin().y
        xnew /= Im.grid().spacing().x
        ynew /= Im.grid().spacing().y

        ca.SetToZero(h)
        ca.Copy(h, xnew, 0)
        ca.Copy(h, ynew, 1)

    elif A.shape[1] == 4:         # 3D affine matrix
        x = ca.Image3D(idgrid, mType)
        y = ca.Image3D(idgrid, mType)
        z = ca.Image3D(idgrid, mType)
        xnew = ca.Image3D(idgrid, mType)
        ynew = ca.Image3D(idgrid, mType)
        znew = ca.Image3D(idgrid, mType)
        ca.Copy(x, h, 0)
        ca.Copy(y, h, 1)
        ca.Copy(z, h, 2)

        x *= Iout.grid().spacing().x
        y *= Iout.grid().spacing().y
        z *= Iout.grid().spacing().z
        x += Iout.grid().origin().x
        y += Iout.grid().origin().y
        z += Iout.grid().origin().z

        # Matrix Multiply (All in real coords)
        Ainv = A.I
        ca.MulC_Add_MulC(xnew, x, Ainv[0, 0], y, Ainv[0, 1])
        ca.Add_MulC_I(xnew, z, Ainv[0, 2])
        xnew += (Ainv[0, 3])
        ca.MulC_Add_MulC(ynew, x, Ainv[1, 0], y, Ainv[1, 1])
        ca.Add_MulC_I(ynew, z, Ainv[1, 2])
        ynew += (Ainv[1, 3])
        ca.MulC_Add_MulC(znew, x, Ainv[2, 0], y, Ainv[2, 1])
        ca.Add_MulC_I(znew, z, Ainv[2, 2])
        znew += (Ainv[2, 3])

        # convert to index coordinates
        xnew -= Im.grid().origin().x
        ynew -= Im.grid().origin().y
        znew -= Im.grid().origin().z
        xnew /= Im.grid().spacing().x
        ynew /= Im.grid().spacing().y
        znew /= Im.grid().spacing().z

        ca.Copy(h, xnew, 0)
        ca.Copy(h, ynew, 1)
        ca.Copy(h, znew, 2)

    Imbig.setGrid(idgrid)

    ca.ApplyH(Ioutbig, Imbig, h, bg)
    # crop Ioutbig -> Iout
    ca.SubVol(Iout, Ioutbig, ca.Vec3Di(0, 0, 0))
    Iout.setGrid(newgrid)   # change back
