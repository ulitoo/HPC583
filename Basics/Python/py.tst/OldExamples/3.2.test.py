
from numpy import *
from pylab import *
#from matplotlib import 

data = loadtxt('./xp/stm.txt')
print(data)
gray()
#spec
imshow(data)
xlim(150,550)
ylim(100,500)
show()