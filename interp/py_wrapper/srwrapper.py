import ctypes
import numpy as np 
import numpy.ctypeslib as npct

libinterp = ctypes.cdll.LoadLibrary('../cpp_pipeline/interp.dylib')
array_1d_float = npct.ndpointer(dtype=np.float, ndim=1, flags='CONTIGUOUS')

