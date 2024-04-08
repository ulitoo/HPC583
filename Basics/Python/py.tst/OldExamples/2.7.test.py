def factors(n):
    factorlist = []
    k = 2
    while k<=n:
        while n%k==0:
            factorlist.append(k)
            n //= k
        k += 1
    return factorlist

#print('aa')

#n = float(input("Enter n: "))
#factores = factors(n)
#print("Factors of:",n," are:",factores)

for n in range(2,1000):
    if len(factors(n))==1:
        print(n,end=" ")

for n in range(1000000000):
    if n%10000000==0:
        print("Step",n)
