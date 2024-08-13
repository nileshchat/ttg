#ifndef HAVE_KERNELS_H
#define HAVE_KERNELS_H

#include <cstddef>
#include "gaussian.h"
#include "tensorview.h"
#include "../../mrakey.h"

#ifndef __CUDA_ARCH__
typedef int cudaStream;
#endif

/* Returns the total size of temporary memory needed for
 * the project() kernel. */
template<mra::Dimension NDIM>
std::size_t project_tmp_size(std::size_t K) {
  const size_t K2NDIM = std::pow(K,NDIM);
  return (NDIM*K2NDIM) // xvec in fcube
       + (5*K2NDIM);   // x in fcube, workspace in transform, values, child_values, r
}

/* Explicitly instantiated for 1, 2, 3 dimensional Gaussians */
template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const Fn* fn,
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  cudaStream stream);

template<mra::Dimension NDIM>
std::size_t compress_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  const size_t K2NDIM = std::pow(K,NDIM);
  return (TWOK2NDIM) // s
          + K2NDIM // workspace
          + mra::Key<NDIM>::num_children() // sumsq for each child and result
          ;
}


/* Explicitly instantiated for 1, 2, 3D */
template<typename T, mra::Dimsion NDIM>
void submit_compress_kernel(
  mra::Key<NDIM> key,
  mra::TensorView<T, NDIM>& p_view,
  mra::TensorView<T, NDIM>& result_view,
  const mra::TensorView<T, NDIM>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<T*, mra::Key<NDIM>::num_children()>& in_ptrs,
  std::size_t K,
  cudaStream stream);


#endif // HAVE_KERNELS_H