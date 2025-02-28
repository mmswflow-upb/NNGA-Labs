import numpy as np

grade = 4.45 # out of 10
print(np.ceil(grade)) # round up
print(np.floor(grade)) # round down

b = np.array(range(1,9)) # range is exclusive of the last number
print(b)
print("Length of b: ", len(b))
print(b[4]) # Index is still the same it remains unchanged by the np library
print(b[0:4]) # exclusive of the last number

print(b[-2]) # going reverse from the end of the array (so second last element)

print(b[:3]) # from the start to the 3rd element

print(b[3:]) # from 3rd index to the 
print(b[::-1]) # reverse the array 1 by 1 because of increment -1, if -2 then 2 by 2 and so on
print(b[3::]) # from the 3rd element to the end

print(b[-3::]) # We go till the end normally we start from the third element but from the end
