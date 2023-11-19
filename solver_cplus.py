import ctypes
import numpy as np
# Load the DLL
cpp_lib = ctypes.CDLL('solver.dll', winmode=0)


def run_SA(init_state, pinsCode, ref_width):
    my_list = np.array(sum(sum(init_state, []), []), dtype=np.int32)
    m, n, p, pinsCodeSize = 2, len(init_state[0]), 6, len(pinsCode)
    my_array = (ctypes.c_int * len(my_list))(*my_list)
    pinsCode = (ctypes.c_int * len(pinsCode))(*pinsCode)
    cpp_lib.run_SA.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cpp_lib.run_SA.restype = ctypes.POINTER(ctypes.c_int)
    # cpp_lib.print_array(array_ptr, ctypes.c_int(24))
    result_ptr = cpp_lib.run_SA(my_array, ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(p),
                                pinsCode, ctypes.c_int(pinsCodeSize), ctypes.c_int(ref_width))
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
