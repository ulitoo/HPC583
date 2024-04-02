
from pylab import plot,show,ylim,xlabel,ylabel
from numpy import linspace,sin,loadtxt,cos

#data = loadtxt("kk2.txt",float)
#print (data)

x = linspace(0,10,100)


y1 = sin(x)
y2 = cos(x)
plot(x,y1,"k-")
plot(x,y2,"k--")
ylim(-1.1,1.1)
xlabel("x axis")
ylabel("y = sin x or y = cos x")

show()

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:17:58 2013

@author: akels
"""

from numpy import *
from pylab import *

data = loadtxt('exmples/stm.txt')
gray()
imshow(data)
xlim(150,550)
ylim(100,500)