from pylab import scatter,xlabel,ylabel,xlim,ylim,show
from numpy import loadtxt
data = loadtxt("./xp/stars.txt",float)
print (data)
x = data[:,0]
y = data[:,1]
scatter(x,y,s=50,alpha=0.09)
xlabel("Temperature")
ylabel("Magnitude")
xlim(0,13000)
ylim(-5,20)
#xlim(13000,0)
#ylim(20,-5)

show()
