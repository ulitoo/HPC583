
'''
x = int(input("Enter a whole number no greater than ten: "))
if x>10:
    print("You entered a number greater than ten.")
    #print("Let me fix that for you.")
    #x = 10
if x>10 or x<1:
    print("Your number is either too big or too small.")


print("Your number is",x)
'''

'''

x = int(input("Enter a whole number no greater than ten: "))
while x>10:
    print("This is greater than ten. Please try again.")
    x = int(input("Enter a whole number no greater than ten: "))
    if x==111:
        break
print("Your number is",x)


'''


#f1 = 1
#f2 = 1
f1,f2,i = 1,1,0
while f1<=1000000:
    print(f1)
    f1,f2 = f2,f1*(4*i+2)/(i+2)

    i+=1
    #fnext = f1 + f2
    #f1 = f2
    #f2 = fnext


