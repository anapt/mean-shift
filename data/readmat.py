import scipy.io as sio
import numpy as np
from decimal import Decimal

mat_contents = sio.loadmat('r15.mat')

# X 600 x 2 double
oct_a = mat_contents['X']
print(type(oct_a))
print(type(oct_a[0][0]))
np.ndarray.tofile(oct_a, "X.bin")

# L 600 x 1 double (but actually int)
oct_b = mat_contents['L']
print(len(oct_b))
print(type(oct_b))
print(type(oct_b[0][0]))
# output is <class 'numpy.uint8'>
# uint8 : Unsigned integer (0 to 255)

print(oct_b[0][0])

np.ndarray.tofile(oct_b, "L.bin")


