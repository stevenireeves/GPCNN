import numpy as np 

def gp(sten, ksid):
	global Kinv, ks
	one = np.ones(5)
	mc = np.dot(one, Kinv.dot(sten))/(np.dot(one,Kinv.dot(one)))
	weight = np.dot(ks[:,ksid], Kinv)
	res = np.zeros(5)
	res[:] = [x-mc for x in sten] 
	iterp = mc + np.dot(weight, res)
	return iterp

def interp(width, height, lx, ly, dat_in, dat_out):
#3 Channels
	for k in range(3):
#Need to think of border cases 
		for i in range(1,height-1):
			for j in range(1,width-1):
				sten = np.array([dat_in[i,j-1,k], dat_in[i-1,j,k], dat_in[i,j,k], dat_in[i+1,j,k], dat_in[i,j+1,k]])
				for iref in range(lx):	
					ii = i*lx + iref
					for jref in range(ly): 
						jj = j*ly + jref
						ksid = iref + jref*lx 
						dat_out[ii,jj,k] = gp(sten, ksid)
#Just copy border for now
		for i in range(0, height, height-1):
			for j in range(0, width, width-1):
				for iref in range(lx):
					ii = i*lx + iref
					for jref in range(ly):
						jj = j*ly + jref
						dat_out[ii,jj,k] = dat_in[i,j,k]


