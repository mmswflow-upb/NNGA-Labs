

#a) LargerYetSmaller

def LargerYetSmaller():
    N = int(input("Provide a number N for larger yet smaller: "))
    j = 1
    for i in range (1, N):
        if i*i <= N:
            j = i
        else: break
    print(j)


LargerYetSmaller()


