from pylab import imshow,show,gray,hot
from numpy import loadtxt
data = loadtxt("./xp/circular.txt",float)
#imshow(data,origin="lower")
gray()
#hot()
#imshow(data)
#imshow(data,origin="lower",extent=[0,10,0,5])

imshow(data,origin="lower",extent=[0,10,0,5],aspect=2.0)
show()
