#ifndef HAVE_MRA_DEVICE_FUNCTIONS_H
#define HAVE_MRA_DEVICE_FUNCTIONS_H

#include "util.h"
#include "types.h"
#include "key.h"
#include "tensorview.h"

namespace mra {


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    SCOPE T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
    }

    /// Computes square of distance between two coordinates
    template <typename T>
    SCOPE T distancesq(const Coordinate<T,1>& p, const Coordinate<T,1>& q) {
        T x = p[0]-q[0];
        return x*x;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,2>& p, const Coordinate<T,2>& q) {
        T x = p[0]-q[0], y = p[1]-q[1];
        return x*x + y*y;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,3>& p, const Coordinate<T,3>& q) {
        T x = p[0]-q[0], y = p[1]-q[1], z=p[2]-q[2];
        return x*x + y*y + z*z;
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,1>& q, T* rsq, std::size_t N) {
        const T x = p(0);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,3>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#endif // __CUDA_ARCH__
    }

} // namespace mra


#endif // HAVE_MRA_DEVICE_FUNCTIONS_H