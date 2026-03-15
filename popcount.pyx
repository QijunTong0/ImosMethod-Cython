# popcount.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t

# GCC/Clang 組み込み関数の宣言
cdef extern from *:
    """
    static inline int popcount64(unsigned long long x) {
        return __builtin_popcountll(x);
    }
    static inline int popcount32(unsigned int x) {
        return __builtin_popcount(x);
    }
    """
    int popcount64(uint64_t x) nogil
    int popcount32(uint32_t x) nogil



def popcount_elementwise64(np.ndarray[int64_t, ndim=1] arr):
    """
    int64 の 1D array に対して、各要素の popcount を int32 array で返す。
    """
    cdef Py_ssize_t i, n = arr.shape[0]
    cdef np.ndarray[int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int64_t* src = <int64_t*> arr.data
    cdef int32_t* dst = <int32_t*> out.data

    with nogil:
        for i in range(n):
            dst[i] = popcount64(<uint64_t> src[i])

    return out


# ------------------------------------------------------------------
# 要素ごとの popcount を返す（int32）
# ------------------------------------------------------------------
def popcount_elementwise32(np.ndarray[int32_t, ndim=1] arr):
    """
    int32 の 1D array に対して、各要素の popcount を int32 array で返す。
    """
    cdef Py_ssize_t i, n = arr.shape[0]
    cdef np.ndarray[int32_t, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef int32_t* src = <int32_t*> arr.data
    cdef int32_t* dst = <int32_t*> out.data

    with nogil:
        for i in range(n):
            dst[i] = popcount32(<uint32_t> src[i])

    return out
