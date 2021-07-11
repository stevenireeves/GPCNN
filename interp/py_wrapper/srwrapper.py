import ctypes
import numpy as np 
import numpy.ctypeslib as npct
import cv2 
import time

libinterp = ctypes.cdll.LoadLibrary('../cpp_pipeline/interp.so')
filename = input('Enter a filename ')
img = cv2.imread(filename)#.astype(np.float32)
#interpolate = libinterp.interpolate
interpolate = libinterp.interpolate_color
interpolate.restype = None
'''
interpolate.argtypes = [npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int)]
'''
interpolate.argtypes = [npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_ubyte),
                        npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_float), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int)]
size = np.asarray(img.shape, dtype=np.int32)
upsamp = np.array([2, 2],dtype=np.int32)
#img_out = np.zeros((size[0]*upsamp[0],upsamp[1]*size[1],3), dtype=np.float32)
b = img[:,:,0].flatten()
g = img[:,:,1].flatten()
r = img[:,:,2].flatten()
b1 = np.zeros((size[0]*2,size[1]*2),dtype=np.float32)
g1 = np.zeros((size[0]*2,size[1]*2),dtype=np.float32)
r1 = np.zeros((size[0]*2,size[1]*2),dtype=np.float32)
start = time.time()
interpolate(b, g, r, b1, g1, r1, upsamp, np.array([size[1], size[0]], dtype=np.int32))
#interpolate(b, b1, upsamp, np.array([size[1], size[0]], dtype=np.int32))
#interpolate(g, g1, upsamp, np.array([size[1], size[0]], dtype=np.int32))
#interpolate(r, r1, upsamp, np.array([size[1], size[0]], dtype=np.int32))
stop = time.time()
img_out = np.stack([b1.reshape(size[0]*2, size[1]*2),
                    g1.reshape(size[0]*2, size[1]*2), 
                    r1.reshape(size[0]*2, size[1]*2)], axis=2)

print(stop - start)
print(img_out.shape)
#img_out = img_out.astype(np.uint8)
cv2.imwrite('img_out.jpg', img_out)
