import numpy as np
import scipy.misc as sp
import cv2
from matplotlib import pyplot as plt 

def gp(sten, ksid, ks, Kinv):
#	one = np.ones(9)
#	mc = np.dot(one, Kinv.dot(sten))/(np.dot(one,Kinv.dot(one)))
	weight = np.dot(ks[:,ksid], Kinv)
#	res = np.zeros(9)
#	res[:] = [x-mc for x in sten] 
#	iterp = mc + np.dot(weight, res)
  iterp = np.dot(weights, sten) 
	return iterp

def loadsten(dat):
	sten = np.array([dat[0,0], dat[1,0], dat[2,0], dat[0,1], dat[1,1], dat[2,1], dat[0,2], dat[1,2], dat[2,2]])
	return sten 

def build_ks(dx, dy, lx, ly,l):
	ks = np.zeros((9,lx*ly))
	ks[:,0] = [-((0.75*dx)**2 + (0.75*dy)**2)/(2*l**2),
						-((0.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (1.25*dy)**2)/(2*l**2)]
 
	ks[:,1] = [-((1.25*dx)**2 + (0.75*dy)**2)/(2*l**2),
						-((0.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (1.25*dy)**2)/(2*l**2)] 

	ks[:,2] = [-((0.75*dx)**2 + (1.25*dy)**2)/(2*l**2),
						-((0.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.75*dy)**2)/(2*l**2)] 

	ks[:,3] = [-((1.25*dx)**2 + (1.25*dy)**2)/(2*l**2),
						-((0.25*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (1.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.25*dy)**2)/(2*l**2), 
						-((1.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((0.25*dx)**2 + (0.75*dy)**2)/(2*l**2), 
						-((0.75*dx)**2 + (0.75*dy)**2)/(2*l**2)] 
	ks = np.exp(ks)
	return ks	


def build_Kinv(dx, dy, l):
	K = np.eye(9)
#Building covariance matrix 
	a = np.exp(-dx**2/(2*l**2))
	b = np.exp(-4.*dx**2/(2*l**2))
	c = np.exp(-dy**2/(2*l**2))
	d = np.exp(-(dx**2+dy**2)/(2*l**2))
	e = np.exp(-((2*dx)**2+dy**2)/(2*l**2))
	f = np.exp(-4.*dx**2/(2*l**2))
	g = np.exp(-(dx**2 + (2*dy)**2)/(2*l**2))
	h = np.exp(-((2*dx)**2 + (2*dy)**2)/(2*l**2))
	K[0,1:] = [a, b, c, d, e, f, g, h] 
	K[1,2:] = [a, d, c, d , g, f, g]
	K[2,3:] = [e, d, c, h, g, f] 
	K[3,4:] = [a, b, c, d, e]
	K[4,5:] = [a, d, c, d]
	K[5,6:] = [e, d, c]
	K[6,7:] = [a, b]
	K[7,8]  = a
	K[1:,0] = K[0,1:]
	K[2:,1] = K[1,2:]
	K[3:,2] = K[2,3:]
	K[4:,3] = K[3,4:]
	K[5:,4] = K[4,5:]
	K[6:,5] = K[5,6:]
	K[7:,6] = K[6,7:]
	K[8,7]  = K[7,8]
#Invert Covariance Matrix 
	Kinv = np.linalg.inv(K) 
	return Kinv

def interp(width, height, lx, ly, ks, Kinv, dat_in, dat_out):
#3 Channels
	for k in range(3):
#Need to think of border cases 
		for i in range(1,height-1):
			for j in range(1,width-1):
				sten = loadsten(dat_in[i-1:i+2,j-1:j+2,k])
				for iref in range(lx):	
					ii = i*lx + iref
					for jref in range(ly): 
						jj = j*ly + jref
						ksid = iref + jref*lx 
						dat_out[ii,jj,k] = gp(sten, ksid, ks, Kinv)
#Just copy border for now
		for i in range(0, height, height-1):
			for j in range(width):
				for iref in range(lx):
					ii = i*lx + iref
					for jref in range(ly):
						jj = j*ly + jref
						dat_out[ii,jj,k] = dat_in[i,j,k]
		for i in range(height):
			for j in range(0,width,width-1):
				for iref in range(lx):
					ii = i*lx + iref
					for jref in range(ly):
						jj = j*ly + jref
						dat_out[ii,jj,k] = dat_in[i,j,k]

#Main Program
img = cv2.imread('downsampled.jpg')
hc, wc = img.shape[:2]
#GP Parameters 
dx = 1./wc 
dy = 1./hc 
l = 0.21
lx = 2
ly = 2

#resolution for finely interpolated image 
hf = hc*ly
wf = wc*lx
img2 = np.zeros((hf,wf,3)) 

ks = build_ks(dx, dy, lx, ly, l) 
Kinv = build_Kinv(dx,dy, l)
interp(wc, hc, lx, ly,ks, Kinv, img, img2)
cv2.imwrite('interpgp9.jpg',img2)
