from numpy import *
from pylab import imshow,show

def mandelbrot(c,n=100):
	
	z = 0
	for i in range(n):
		z = c + z**2
		#print(z)

	z_mod = sqrt(z * z.conjugate())
	return z_mod<2
	
x,y = mgrid[-2:2:1000j,-2:2:1000j]

c = double(x) + double(y)*1.0j
#print(mandelbrot(c))

imshow(mandelbrot(c))
show()