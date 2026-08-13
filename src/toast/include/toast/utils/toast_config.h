#ifndef PYTOAST_CONFIG_H
#define PYTOAST_CONFIG_H


#if !defined(TOAST_WITH_CUDA) && !defined(TOAST_NO_CUDA) && defined(__CUDACC__)
#  define TOAST_WITH_CUDA 1
#endif

#endif // PYTOAST_CONFIG_H
