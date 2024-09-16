#ifndef HAVE_KERNELS_H
#define HAVE_KERNELS_H

#include <cstddef>
#include "gaussian.h"
#include "tensorview.h"
#include "key.h"
#include "domain.h"
#include "types.h"
#include "ttg/device/device.h"

#if defined(TTG_ENABLE_CUDA)
#include <cuda.h>
typedef cudaStream_t stream_t;
#elif defined(TTG_ENABLE_HOST)
typedef decltype(ttg::device::current_stream()) stream_t;
#else
#warning Unknown device model, please add appropriate header includes!
#endif

/* Returns the total size of temporary memory needed for
 * the project() kernel. */
template<mra::Dimension NDIM>
std::size_t project_tmp_size(std::size_t K) {
  const size_t K2NDIM = std::pow(K,NDIM);
  return (NDIM*K2NDIM) // xvec in fcube
       + (NDIM*K)      // x in fcube
       + (4*K2NDIM);   // workspace in transform, values, child_values, r
}

/* Explicitly instantiated for 1, 2, 3 dimensional Gaussians */
template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const mra::Domain<NDIM>& D,
  const T* gldata,
  const Fn& fn,
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  stream_t stream);

template<mra::Dimension NDIM>
std::size_t compress_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  const size_t K2NDIM = std::pow(K,NDIM);
  return (TWOK2NDIM) // s
          + K2NDIM // workspace
          + mra::Key<NDIM>::num_children // sumsq for each child and result
          ;
}


/* Explicitly instantiated for 3D */
template<typename T, mra::Dimension NDIM>
void submit_compress_kernel(
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& p_view,
  mra::TensorView<T, NDIM>& result_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<const T*, mra::Key<NDIM>::num_children>& in_ptrs,
  stream_t stream);

template<mra::Dimension NDIM>
std::size_t reconstruct_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM); // s
  const size_t K2NDIM = std::pow(K,NDIM); // workspace
  return TWOK2NDIM + K2NDIM;
}


template<typename T, mra::Dimension NDIM>
void submit_reconstruct_kernel(
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& node,
  const mra::TensorView<T, 2>& hg,
  const mra::TensorView<T, NDIM>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children>& r_arr,
  T* tmp,
  stream_t stream);

#endif // HAVE_KERNELS_H