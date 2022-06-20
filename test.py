import numpy as np

# a = np.array([[1, 2],
#        [0, 0],
#        [1, 0],
#        [0, 2],
#        [2, 1],
#        [1, 0],
#        [1, 0],
#        [0, 0],
#        [1, 0],
#       [2, 2]])

# # a[a[:,2].argsort()]
# a[np.lexsort(np.fliplr(a).T)]
# print(a)

# # array([[0, 0, 1],
# #        [1, 2, 3],
# #        [4, 5, 6]])

# Xắp xếp dựa trên phần tử thứ 2
def take_second(elem):
    return elem[0]


# list ngẫu nhiên
random = [[2, 2], [3, 4], [4, 1], [1, 3]]

# sắp xếp list với key
sorted_list = sorted(random, key=take_second)

# hiển thị list
print('List đã được sắp xếp:', sorted_list)