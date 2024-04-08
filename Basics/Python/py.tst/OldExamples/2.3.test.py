'''from math import *
'''r = [ 1, 1, 2, 3, 5, 8, 13, 21 ]
x = 1.0
y = 1.5
z = -2.2
r = [ x, y, z ]
r = [ 2*x, x+y, z/sqrt(x**2+y**2) ]
print(r)
from math import sqrt
r = [ 1.0, 1.5, -2.2 ]
length = sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
print(length)
r = [ 1.0, 1.5, -2.2 ]
r[1] = 3.5
print(r)

from math import log
print("--------------------")
r = [ 1.0, 1.5, 2.2 ]
r.append(6.1)
print(r)
logr = list(map(log,r))
print(logr)
'''
'''

from numpy import *
#zeros
a = zeros(4,float)
print(a)
