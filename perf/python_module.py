print("From python: Within python module")

import os,sys
HERE = os.getcwd()
sys.path.insert(0,HERE)

import numpy as np
import cupy
from cupy.cuda import memory

#Consider inserting the following
# -> from cupy.cuda import memory
# def my_function(iary):
#      a_cupy = cupy.ndarray(
#               iary.shape,
#               iary.dtype(iary.dtype.name),
#               cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
#               iary,
#               ,
#               a_chx,
#               0), 0),
#               strides=a_chx.strides,
#               )

def my_function1(a):
    b = cupy.asarray(a)
    print("In Python: CALC Start H->D")
    b *= 5 
    b *= b 
    b += b 
    print("In Python: CALC DONE H->D")

def my_function2(a):
    b = cupy.ndarray(
                a.__array_interface__['shape'][0],
                cupy.dtype(a.dtype.name),
                cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                           a.__array_interface__['data'][0], #<---Pointer?
                                           a.size,
                                           a,
                                           0), 0),
                strides=a.__array_interface__['strides'])
    print("In Python: CALC Start H->D")
    b *= 5 
    b *= b 
    b += b 
    print("In Python: CALC DONE H->D")
