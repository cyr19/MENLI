from combine_1 import *

matrices = get_all_matrices('mt')

# ref adv
matrix1 = matrices['ref']['adv']

# ref MT-seg
matrix2 = matrices['ref']['mt-seg']

# src adv
matrix3 = matrices['src']['adv']

# src MT-seg
matrix4 = matrices['src']['mt-seg']

print('ref-adv')
print_wining_matrix(matrix1)
print('ref-MT-seg')
print_wining_matrix(matrix2)
print('src-adv')
print_wining_matrix(matrix3)
print('src-MT-seg')
print_wining_matrix(matrix4)
