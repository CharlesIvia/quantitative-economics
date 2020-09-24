import numpy as np

# CREATING A MATRIX MANUALLY

A = [[1, 4, 5], [-5, 8, 9]]

print(A)

print("A[1] = ", A[1])

column = []

for row in A:
    column.append(row[2])

print("3rd column = ", column)

# USING NP MULTIDIMENSIONAL ARRAY

a = np.array([[1, 2, 3], [3, 4, 5]])
print(a)
print(type(a))


zeros_array = np.zeros((2, 3))
print(zeros_array)

# NUMERICAL OPS ON A MATRIX

# Addition of two matrices
first = np.array([[2, 4], [5, -6]])
second = np.array([[9, -3], [3, 6]])

print(first)
print(second)
result = first + second
print(result)

# Multiplication using dot()

result = first.dot(second)
print(result)


# TRANSPOSE OF A MATRIX

fourth = np.array([[1, 1], [2, 1], [3, -3]])
print(fourth)

fourth_transpose = fourth.transpose()

print(fourth_transpose)

# ACCESSING ELEMENTS OF A MATRIX

fifth = np.array([[1, 4, 5, 12], [-5, 8, 9, 0], [-6, 7, 11, 19]])

print(fifth)

# first element of the first row

print("fifth[0][0] = ", fifth[0][0])

# third element of secon row

print("fifth[1][2] = ", fifth[1][2])

# last element of last row

print("fifth[-1][-1] =", fifth[-1][-1])

# reverse
to_rep = fifth[0][::-1]
fifth[0] = to_rep
print(fifth)


# ACCESS ROWS OF A MATRIX

sixth = np.array([[1, 4, 5, 12], [-5, 8, 9, 0], [-6, 7, 11, 19]])

print("sixth[0] = ", sixth[0])  # first row
print("sixth[2] = ", sixth[2])  # Third Row
print("sixth[-1] = ", sixth[-1])  # Last Row (3rd row in this case)

# ACCESS COLUMNS OF A MATRIX

seventh = np.array([[1, 4, 5, 12], [-5, 8, 9, 0], [-6, 7, 11, 19]])

print("seventh[:, 0] = ", seventh[:, 0])  # first column
print("seventh[:,3] = ", seventh[:, 3])  # Fourth Column
print("seventh[:,-1] = ", seventh[:, -1])
# Last Column (4th column in this case)


# SLICING OF A MATRIX

# slicing a one dimensional nump array

letters = np.array([1, 3, 5, 7, 9, 7, 5])

# third to fifth elements

print(letters[2:5])

# 1st to 2nd elements
print(letters[:-5])

# 6th to last elements
print(letters[5:])

# All elements=- first to last

print(letters[:])

# reversing a list

print(letters[::-1])

# Slicing a MATRIX


my_matrix = np.array([[1, 4, 5, 12, 14], [-5, 8, 9, 0, 17], [-6, 7, 11, 19, 21]])

print(my_matrix)

# two rows ans four columns

print(my_matrix[:2, :4])

# first row all columns

print(
    my_matrix[
        :1,
    ]
)

# all rows third column

print(my_matrix[:, 2])

# all rows third to fifth column

print(my_matrix[:, 2:5])
