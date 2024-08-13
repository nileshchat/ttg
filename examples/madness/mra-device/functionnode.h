#ifndef HAVE_FUNCTIONNODE_H
#define HAVE_FUNCTIONNODE_H

#include "../../mrakey.h"
#include "tensor.h"

namespace mra {
    template <typename T, Dimension NDIM>
    class FunctionReconstructedNode {
    public: // temporarily make everything public while we figure out what we are doing
        using key_type = Key<NDIM>;
        using tensor_type = Tensor<T,NDIM>;
        static constexpr bool is_function_node = true;
        key_type key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        mutable T sum = 0.0; //< If recurring up tree (e.g., in compress) can use this to also compute a scalar reduction
        bool is_leaf = false; //< True if node is leaf on tree (i.e., no children).
        std::array<bool, Key<NDIM>::num_children> is_child_leaf = { false };
        tensor_type coeffs; //< if !is_leaf these are junk (and need not be communicated)
        FunctionReconstructedNode() = default; // Default initializer does nothing so that class is POD
        FunctionReconstructedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        {}
        //T normf() const {return (is_leaf ? coeffs.normf() : 0.0);}
        bool has_children() const {return !is_leaf;}
    };


    template <typename T, Dimension NDIM>
    class FunctionCompressedNode {
    public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;
        Key<NDIM> key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::array<bool, Key<NDIM>::num_children> is_child_leaf; //< True if that child is leaf on tree
        Tensor<T,NDIM> coeffs; //< Always significant
        FunctionCompressedNode(std::size_t K)
        : coeffs(2*K)
        { }
        FunctionCompressedNode(const Key<NDIM>& key, std::size_t K)
        : key(key)
        , coeffs(2*K)
        { }

        //T normf() const {return coeffs.normf();}
        bool has_children(size_t childindex) const {assert(childindex<Key<NDIM>::num_children); return !is_leaf[childindex];}
    };


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
    }

    /// Computes square of distance between two coordinates
    template <typename T>
    T distancesq(const Coordinate<T,1>& p, const Coordinate<T,1>& q) {
        T x = p[0]-q[0];
        return x*x;
    }

    template <typename T>
    T distancesq(const Coordinate<T,2>& p, const Coordinate<T,2>& q) {
        T x = p[0]-q[0], y = p[1]-q[1];
        return x*x + y*y;
    }

    template <typename T>
    T distancesq(const Coordinate<T,3>& p, const Coordinate<T,3>& q) {
        T x = p[0]-q[0], y = p[1]-q[1], z=p[2]-q[2];
        return x*x + y*y + z*z;
    }

    template <typename T>
    void distancesq(const Coordinate<T,3>& p, const TensorView<T,1>& q, T* rsq, std::size_t N) {
        const T x = p(0);
#ifdef __CUDA_ARCH__
        int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
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
    void distancesq(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
#ifdef __CUDA_ARCH__
        int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
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
    void distancesq(const Coordinate<T,3>& p, const TensorView<T,3>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef __CUDA_ARCH__
        int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
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

#endif // HAVE_FUNCTIONNODE_H
