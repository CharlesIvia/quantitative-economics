import numpy as np
from math import pi

a = np.arange(15).reshape(3, 5)
print(a)

print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))

#Array creation

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

c = np.zeros((3, 4))
print(c)

d = np.ones((3, 4), dtype=np.int16)
print(d)

e = np.empty((2, 3))
print(e)

#The arrange fn

f = np.arange(10, 50, 10)  # Step size of 10
print(f)

g = np.arange(0, 2, 0.3)
print(g)

print(len(g) % 2 == 0)

#linspace

h = np.linspace(0, 2, 9)  # 9 numbers from 0 to 2
print(h)

i = np.linspace(0, 2*pi, 100)
print(pi)
print(i)


#BASIC OPERATIONS

j = np.array([20, 30, 40, 50])
k = np.arange(4)
l = j - k
print(l)

m = k**2
print(m)

#multiplication

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])

print(A * B)  # Elementwise product

print(A @ B)  # matrix product

print(A.dot(B))  # another matrix product


#Universal fns

C = np.arange(3)
c_expo = np.exp(C)
print(c_expo)

c_sqrt = np.sqrt(C)
print(c_sqrt)

#Indexing, Slicing and Iterating

#One dimensional arrays

D = np.arange(10)**3
print(D)

D[:6:2] = 1000  # from start to position 6, exclusive, set every 2nd element to 1000
print(D)

D = np.arange(10)**3
for i in D:
    print((i**(1/3)))


E = np.arange(10, 60, 2).reshape(5, 5)
print(len(E))
print(E)

for row in E:
    print(row)


for element in E.flat:
    print(element)


#Columns

column = []

for i in E:
    column.append(i[1])

print(column)