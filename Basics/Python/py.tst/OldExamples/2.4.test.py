from numpy import dot,array,loadtxt,sqrt
'''
a = zeros(4,complex)
print("--------------------")
print(a)
print("--------------------")
a = zeros([3,4],float)
print(a)
print("--------------------")
from numpy import empty
a = empty(4,float)
print(a)
print("--------------------")
a = array([[1,2,3],[4,5,6]],int)
a[1,2] = 4
print(a)
'''
print("--------------------")
#from numpy import loadtxt
a = loadtxt("kk.txt",float)
c = loadtxt("kk.txt",float)
print(a)
b = 2*a
print("--------------------")
print(b)
print("--------------------")
print (dot(a,b)+3*a)
#b = array(list(map(sqrt,a)),float)
b=sqrt(a)
print(b)
print(b.shape)
print (b.size)
print(sum(sum(a)))
print(a.size)
print(len(a))
print(sum(sum(a))/a.size)