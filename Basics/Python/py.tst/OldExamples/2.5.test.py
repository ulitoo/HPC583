from numpy import array,copy
a = array([1,1],int)
b = copy(a)
a[0] = 2
print(a)
print(b)
print("--------------------")
r = [ 1, 3, 5 ]
for n in r:
    print(n)
    print(2*n)
print("Finished")
print("--------------------")
r = range(5)
print (r)
for n in range(10//2):
    print("Hello again :",2**n)
print("--------------------")   
R = 1.097e-2
for m in [1,2,3]:
    print("Series for m =",m)
    for k in [1,2,3,4,5]:
        n = m + k
        invlambda = R*(1/m**2-1/n**2)
        print(" ",1/invlambda,"nm")
