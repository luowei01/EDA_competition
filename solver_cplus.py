import random
import ctypes
import numpy as np
# Load the DLL
cpp_lib = ctypes.CDLL('solver.dll', winmode=0)


def v_compute(state, action):
    my_list = np.array(sum(sum(state, []), []), dtype=np.int32)
    m, n, p = 2, len(state[0]), 6
    my_array = (ctypes.c_int * len(my_list))(*my_list)
    cpp_lib.v_compute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cpp_lib.v_compute.restype = ctypes.POINTER(ctypes.c_int)
    # cpp_lib.print_array(array_ptr, ctypes.c_int(24))
    result_ptr = cpp_lib.v_compute(my_array, ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(p), ctypes.c_int(action))
    # 使用 ctypes 的 contents 属性获取指针指向的整数数组
    result_array = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_int * len(my_list))).contents
    # 将列表转换为 NumPy 数组
    original_array = np.array(list(result_array))
    # 改变形状为二维数组
    reshaped_list = original_array.reshape(m, n, p).tolist()
    # 释放内存（重要步骤）
    # cpp_lib.destroy_vector.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cpp_lib.destroy_vector(result_ptr)
    return reshaped_list


state = [[[3, 1, 3, 2, 1, 170], [4, 1, 3, 2, 1, 170]], [[1, 0, 3, 2, 0, 140], [2, 0, 3, 2, 0, 140]]]
for i in range(10):
    state = v_compute(state, 2)
