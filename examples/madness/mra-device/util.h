#ifndef MRA_DEVICE_UTIL_H
#define MRA_DEVICE_UTIL_H

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define VARSCOPE __device__
#else // __CUDA_ARCH__
#define SCOPE
#define VARSCOPE
#endif // __CUDA_ARCH__

#endif // MRA_DEVICE_UTIL_H