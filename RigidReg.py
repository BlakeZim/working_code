'''RigidReg.py - Caleb Rottman
uses a gradient descent registration for 2D or 3D rigid registration'''
import sys
import PyCA.Core as ca

import matplotlib.pyplot as plt
plt.ion()   # tell it to use interactive mode -- see results immediately
import numpy as np

import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd


def RigidReg(Is, It,
             theta_step=.0001,
             t_step=.01,
             a_step=0,
             maxIter=350,
             plot=True,
             origin=None,
             theta=0,           # only applies for 2D
             t=None,            # only applies for 2D
             Ain=np.matrix(np.identity(3))):

    Idef = ca.Image3D(It.grid(), It.memType())
    gradIdef = ca.Field3D(It.grid(), It.memType())
    h = ca.Field3D(It.grid(), It.memType())
    ca.SetToIdentity(h)
    x = ca.Image3D(It.grid(), It.memType())
    y = ca.Image3D(It.grid(), It.memType())
    DX = ca.Image3D(It.grid(), It.memType())
    DY = ca.Image3D(It.grid(), It.memType())
    diff = ca.Image3D(It.grid(), It.memType())
    scratchI = ca.Image3D(It.grid(), It.memType())

    ca.Copy(x, h, 0)
    ca.Copy(y, h, 1)
    if origin is None:
        origin = [(Is.grid().size().x+1)/2.0,
                  (Is.grid().size().y+1)/2.0,
                  (Is.grid().size().z+1)/2.0]
    x -= origin[0]
    y -= origin[1]

    numel = It.size().x * It.size().y * It.size().z
    immin, immax = ca.MinMax(It)
    imrng = max(immax-immin, .01)
    t_step /= numel*imrng
    theta_step /= numel*imrng
    a_step /= numel*imrng
    energy = []
    a = 1

    if cc.Is3D(Is):
        if theta:
            print "theta is not utilized in 3D registration"
        z = ca.Image3D(It.grid(), It.memType())
        DZ = ca.Image3D(It.grid(), It.memType())
        ca.Copy(z, h, 2)
        z -= origin[2]

        A = np.matrix(np.identity(4))
        cc.ApplyAffineReal(Idef, Is, A)
#        cc.ApplyAffine(Idef, Is, A, origin)

        t = [0, 0, 0]
        for i in xrange(maxIter):
            ca.Sub(diff, Idef, It)
            ca.Gradient(gradIdef, Idef)
            ca.Copy(DX, gradIdef, 0)
            ca.Copy(DY, gradIdef, 1)
            ca.Copy(DZ, gradIdef, 2)

            # take gradient step for the translation
            ca.Mul(scratchI, DX, diff)
            t[0] += t_step*ca.Sum(scratchI)
            ca.Mul(scratchI, DY, diff)
            t[1] += t_step*ca.Sum(scratchI)
            ca.Mul(scratchI, DZ, diff)
            t[2] += t_step*ca.Sum(scratchI)

            A[0,3] = t[0]
            A[1,3] = t[1]
            A[2,3] = t[2]
            if a_step > 0:
                DX *= x
                DY *= y
                DZ *= z
                DZ += DX
                DZ += DY
                DZ *= diff
                d_a = a_step*ca.Sum(DZ)
                a_prev = a
                a += d_a
                # multiplying by a/a_prev is equivalent to adding (a-aprev)
                A = A*np.matrix([[a/a_prev, 0, 0, 0],
                                 [0, a/a_prev, 0, 0],
                                 [0, 0, a/a_prev, 0],
                                 [0, 0, 0, 1]])


            # Z rotation
            ca.Copy(DX, gradIdef, 0)
            ca.Copy(DY, gradIdef, 1)
            DX *= y
            ca.Neg_I(DX)
            DY *= x
            ca.Add(scratchI, DX, DY)
            scratchI *= diff
            theta = -theta_step*ca.Sum(scratchI)
            # % Recalculate A
            A = A*np.matrix([[np.cos(theta), np.sin(theta), 0, 0],
                             [-np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

            # Y rotation
            ca.Copy(DX, gradIdef, 0)
            ca.Copy(DZ, gradIdef, 2)
            DX *= z
            ca.Neg_I(DX)
            DZ *= x
            ca.Add(scratchI, DX, DZ)
            scratchI *= diff
            theta = -theta_step*ca.Sum(scratchI)
            # % Recalculate A
            A = A*np.matrix([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0, 0],
                             [-np.sin(theta), 0,  np.cos(theta), 0],
                             [0, 0, 0, 1]])

            # X rotation
            ca.Copy(DY, gradIdef, 1)
            ca.Copy(DZ, gradIdef, 2)
            DY *= z
            ca.Neg_I(DY)
            DZ *= y
            ca.Add(scratchI, DY, DZ)
            scratchI *= diff
            theta = -theta_step*ca.Sum(scratchI)
            # Recalculate A
            A = A*np.matrix([[1, 0, 0, 0],
                             [0, np.cos(theta), np.sin(theta), 0],
                             [0, -np.sin(theta), np.cos(theta), 0],
                             [0, 0, 0, 1]])

            cc.ApplyAffineReal(Idef, Is, A)
            #        cc.ApplyAffine(Idef, Is, A, origin)

            # % display Energy (and other figures) at the end
            energy.append(ca.Sum2(diff))
            if (i == maxIter-1) or (i > 75 and abs(energy[-1]-energy[-50]) < immax):
                cd.DispImage(diff, title='Difference Image', colorbar=True)
                plt.figure()
                plt.plot(energy)
                cd.DispImage(Idef, title='Deformed Image')
                break

    elif cc.Is2D(Is):
        # theta = 0
        if t is None:
            t = [0, 0]

        # A = np.array([[a*np.cos(theta), np.sin(theta), t[0]],
        #               [-np.sin(theta), a*np.cos(theta), t[1]],
        #               [0, 0, 1]])

        A = np.copy(Ain)
        cc.ApplyAffineReal(Idef,Is,A)
        # ca.Copy(Idef, Is)
        for i in xrange(1, maxIter):
            # [FX,FY] = gradient(Idef)
            ca.Sub(diff, Idef, It)
            ca.Gradient(gradIdef, Idef)
            ca.Copy(DX, gradIdef, 0)
            ca.Copy(DY, gradIdef, 1)

            # take gradient step for the translation
            ca.Mul(scratchI, DX, diff)
            t[0] += t_step*ca.Sum(scratchI)
            ca.Mul(scratchI, DY, diff)
            t[1] += t_step*ca.Sum(scratchI)

            # take gradient step for the rotation theta
            if a_step > 0:
                # d/da
                DX *= x
                DY *= y
                DY += DX
                DY *= diff
                d_a = a_step*ca.Sum(DY)
                a += d_a
            # d/dtheta
            ca.Copy(DX, gradIdef, 0)
            ca.Copy(DY, gradIdef, 1)
            DX *= y
            ca.Neg_I(DX)
            DY *= x
            ca.Add(scratchI, DX, DY)
            scratchI *= diff
            d_theta = theta_step*ca.Sum(scratchI)
            theta -= d_theta

            # Recalculate A, Idef
            A = np.matrix([[a*np.cos(theta), np.sin(theta), t[0]],
                           [-np.sin(theta), a*np.cos(theta), t[1]],
                           [0, 0, 1]])
            A = Ain*A

            cc.ApplyAffineReal(Idef, Is, A)
            #        cc.ApplyAffine(Idef, Is, A, origin)

            # % display Energy (and other figures) at the end
            energy.append(ca.Sum2(diff))
            if (i == maxIter-1) or (i > 75 and abs(energy[-1]-energy[-50]) < immax):
                if i == maxIter-1:
                    print "not converged in ", maxIter, " Iterations"
                if plot:
                    cd.DispImage(diff, title='Difference Image', colorbar=True)
                    plt.figure()
                    plt.plot(energy)
                    cd.DispImage(Idef, title='Deformed Image')
                break
    return A
