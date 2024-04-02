from numpy import array,copy,loadtxt,log,exp
#from math import exp
def factorial(n):
    f = 1.0
    for k in range(1,n+1):
        f *= k
    return f
print("--------------------")
#for n in range(10):
#    print(factorial(n))
values = loadtxt("vector.txt",float)
suma = sum(values)
mean = sum(values*values)/len(values)
print("mean:",mean)
print("sum:",suma)
#logs = array(list(map(log,values)),float)
#geometric = exp(sum(logs)/len(logs))
geometric = exp(sum(log(values))/len(values))
#print("Logs:",logs)
print("Geom:",geometric)
