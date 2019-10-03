import numpy as np
import scipy.misc as sp
import cv2
from matplotlib import pyplot as plt
import kernels

def gp(sten, ksid, ks, Kinv):
    weights = np.dot(ks[:, ksid], Kinv)
    iterp = np.dot(weights, sten)
    return iterp

def loadsten(dat):
    return np.array([dat[0, 0], dat[1, 0], dat[2, 0],
                     dat[0, 1], dat[1, 1], dat[2, 1],
                     dat[0, 2], dat[1, 2], dat[2, 2]])

def interp(width, height, lx, ly, ks, Kinv, dat_in, dat_out):
    #3 Channels
    for k in range(3):
        #Need to think of border cases
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                sten = loadsten(dat_in[i - 1:i + 2, j - 1:j + 2, k])
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        ksid = iref + jref * lx
                        dat_out[ii, jj, k] = gp(sten, ksid, ks, Kinv)
#Just copy border for now
        for i in range(0, height, height - 1):
            for j in range(width):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]
        for i in range(height):
            for j in range(0, width, width - 1):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]


#Main Program
img = cv2.imread('bad_receipts/2838_6893a1e4-1f38-49e2-a14c-5e0359623d4f.jpg')
hc, wc = img.shape[:2]
#GP Parameters
dx = 1. / wc
dy = 1. / hc
l = 12*min(dx,dy)
lx = 4
ly = 4

pnt =np.array([[-dx, -dy], [0, -dy], [dx, -dy], 
               [-dx,   0], [0,   0], [dx,   0], 
               [-dx,  dy], [0,  dy], [dx,  dy]])

K = np.zeros((9,9))
for i in range(9): 
	for j in range(9): 
#		K[i,j] = matern5(pnt[i], pnt[j], l)
		K[i,j] = kernels.matern3(pnt[i], pnt[j], l) 
#		K[i,j] = sqrexp(pnt[i], pnt[j], l)

Kinv = np.linalg.inv(K) 
kp = np.zeros((lx*ly, 2))
start = -0.25 
if lx == 4: 
    start = -0.375 
for j in range(ly):
    for i in range(lx):  
        idk = i + lx*j
        kp[idk, 0] = (start + 1.0/lx*i)*dx
        kp[idk, 1] = (start + 1.0/ly*j)*dy

ks = np.zeros((9,lx*ly))
for idy in range(lx*ly): 
    for idx in range(9): 
#	ks[idx, idy] = sqrexp(kp[idy, :], pnt[idx][:],l)
#	ks[idx, idy] = matern5(kp[idy, :], pnt[idx][:], l) 
        ks[idx, idy] = kernels.matern3(kp[idy, :], pnt[idx][:], l) 

#resolution for finely interpolated image
hf = hc * ly
wf = wc * lx
img2 = np.zeros((hf, wf, 3))

interp(wc, hc, lx, ly, ks, Kinv, img, img2)
cv2.imwrite('testgp9.jpg', img2)
