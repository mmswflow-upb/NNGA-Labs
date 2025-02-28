n = input("Give an input: ")

try:
    if int(n) > 0: #This way it wont change the original value, only inside the if statement
        print(n,' is greater than zero')
    elif int(n) == 0:
        print(n, ' is equal to zero')
    else:
        print(n, ' is lower than zero')

except ValueError:
    print("Please enter a valid number")
