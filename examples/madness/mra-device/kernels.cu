#include <ttg/device/device.h>
#include "tensorview.h"
#include "functionnode.h"
#include "functiondata.h"
#include "gaussian.h"
#include "../../mrakey.h"
#include "../../mradomain.h"

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
__device__ void make_xvec(const mra::TensorView<T,2>& x, mra::TensorView<T,2>& xvec, std::integral_constant<1>) {
  /* uses threads in 3 dimensions */
  xvec = x;
  /* TensorView assignment synchronizes */
}

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
__device__ void make_xvec(const mra::TensorView<T,2>& x, mra::TensorView<T,2>& xvec, std::integral_constant<2>) {
  const std::size_t K = x.dim(1);
  if (threadId.z == 0) {
    for (size_t i=blockIdx.y; i<K; i += blockDim.y) {
      for (size_t j=blockIdx.x; j<K; j += blockDim.x) {
        size_t ij = i*K + j;
        xvec(0,ij) = x(0,i);
        xvec(1,ij) = x(1,j);
      }
    }
  }
  __syncthreads();
}

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
__device__ void make_xvec(const mra::TensorView<T,2>& x, mra::TensorView<T,2>& xvec, std::integral_constant<3>) {
  const std::size_t K = x.dim(1);
  for (size_t i=threadIdx.z; i<K; i += blockDim.z) {
    for (size_t j=blockIdx.y; j<K; j += blockDim.y) {
      for (size_t k=blockIdx.x; k<K; k += blockDim.x) {
        size_t ijk = i*K*K + j*K + k;
        xvec(0,ijk) = x(0,i);
        xvec(1,ijk) = x(1,j);
        xvec(2,ijk) = x(2,k);
      }
    }
  }
  __syncthreads();
}


template <typename functorT, typename T, mra::Dimension NDIM>
__device__
void fcube(const functorT& f,
           const mra::Key<NDIM>& key,
           const T thresh,
           // output
           mra::TensorView<T,2>& values,
           std::size_t K,
           // temporaries
           mra::TensorView<T, NDIM> x,
           mra::TensorView<T, 2> xvec) {
  if (is_negligible(f,Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
      values = 0.0;
      /* TensorView assigment synchronizes */
  }
  else {
    const size_t K = values.dim(0);
    const size_t K2NDIM = std::pow(K,NDIM);
    mra::FunctionData<T, NDIM>::make_quadrature_pts(key,x);

    constexpr bool call_coord = std::is_invocable_r<T, decltype(f), Coordinate<T,NDIM>>(); // f(coord)
    constexpr bool call_1d = (NDIM==1) && std::is_invocable_r<T, decltype(f), T>(); // f(x)
    constexpr bool call_2d = (NDIM==2) && std::is_invocable_r<T, decltype(f), T, T>(); // f(x,y)
    constexpr bool call_3d = (NDIM==3) && std::is_invocable_r<T, decltype(f), T, T, T>(); // f(x,y,z)
    constexpr bool call_vec = std::is_invocable_r<void, decltype(f), SimpleTensor<T,NDIM,K2NDIM>, std::array<T,K2NDIM>&>(); // vector API

    static_assert(call_coord || call_1d || call_2d || call_3d || call_vec, "no working call");

    if constexpr (call_1d || call_2d || call_3d || call_vec) {
      make_xvec(x, xvec, std::integral_constant<NDIM>{});
      if constexpr (call_vec) {
        f(xvec,values.data());
      }
      else if constexpr (call_1d || call_2d || call_3d) {
        eval_cube_vec(f, xvec, values);
      }
    }
    else if constexpr (call_coord) {
      eval_cube(f, x, values);
    }
    else {
      throw "how did we get here?";
    }
    __syncthreads();
  }
}

/* reference implementation, adapted from madness */
template <typename aT, typename bT, typename cT>
__device__
void mTxmq(std::size_t dimi, std::size_t dimj, std::size_t dimk,
           cT* __restrict__ c, const aT* a, const bT* b, std::ssize_t ldb=-1) {
  if (ldb == -1) ldb=dimj;

  /* trivial 2D implementation for devices */
  if (threadId.z == 0) {
    for (std::size_t i = threadId.y; i < dimi; i += blockDim.y) {
      cT* ci = c + i*dimj; // the row of C all threads in dim x work on
      const aT *aik_ptr = a;
      /* not parallelized */
      for (long k=0; k<dimk; ++k,aik_ptr+=dimi) {
        aT aki = *aik_ptr;
        for (std::size_t j = threadId.x; j < dimj; j += blockDim.x) {
          ci[j] += aki*b[k*ldb+j];
        }
      }
    }
  }
  __syncthreads();
}
template <Dimension NDIM, typename T>
__device__
void transform(const mra::TensorView<T,NDIM>& t,
               const mra::TensorView<T,2>& c,
               mra::TensorView<T,NDIM>& result,
               mra::TensorView<T, NDIM>& workspace) {
  const cT* pc = c.ptr();
  resultT *t0=workspace.ptr(), *t1=result.ptr();
  if (t.ndim() & 0x1) std::swap(t0,t1);
  const size_t dimj = c.dim(1);
  size_t dimi = 1;
  for (size_t n=1; n<t.ndim(); ++n) dimi *= dimj;
  mTxmq(dimi, dimj, dimj, t0, t.ptr(), pc);
  for (size_t n=1; n<t.ndim(); ++n) {
    mTxmq(dimi, dimj, dimj, t1, t0, pc);
    std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

template<std::size_t NDIM>
__device__
std::array<mra::Slice, NDIM> get_child_slice(mra::Key<NDIM> key, std::size_t K, int child) {
  std::array<Slice,NDIM> slices;
  for (size_t d = 0; d < NDIM) {
    int b = (child>>d) & 0x1;
    slices[d] = Slice(K*b, K*(b+1));
  }
  return slices;
}

template<typename Fn, typename T, mra::Dimension NDIM>
__global__ fcoeffs_kernel1(
  const Fn f,
  mra::Key<NDIM> key,
  T* tmp,
  const T* phibar_ptr,
  std::size_t K,
  T thresh)
{
  int tid = threadIdx.x;
  int blockid = blockIdx.x;
  const std::size_t K2NDIM = std::pow(K, NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  /* reconstruct tensor views from pointers
   * make sure we have the values at the same offset (0) as in kernel 1 */
  auto values       = mra::TensorView<T, NDIM>(&tmp[0       ], 2*K);
  auto r            = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM+1*K2NDIM], K);
  auto child_values = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM+2*K2NDIM], K);
  auto workspace    = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM+3*K2NDIM], K);
  auto x            = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM+4*K2NDIM], K);
  auto x_vec        = mra::TensorView<T, 2   >(&tmp[TWOK2NDIM+5*K2NDIM], NDIM, K2NDIM);
  auto phibar       = mra::TensorView<T, 2   >(phibar_ptr, K, K);
  /* compute one child per block */
  if (blockid < key.num_children) {
    mra::Key<NDIM> child = key.child_at(blockid);
    fcube(f, child, thresh, child_values, K, x, xvec);
    transform(child_values,phibar,r, K, workspace);
    auto child_slice = get_child_slice(key, K, blockid);
    values(child_slice) = r;
  }
}

template<typename T, mra::Dimension NDIM>
__global__ fcoeffs_kernel2(
  mra::Key<NDIM> key,
  T* coeffs_ptr,
  const T* hgT_ptr,
  T* tmp,
  bool *is_leaf,
  std::size_t K,
  T thresh)
{
  const int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
  const std::size_t K2NDIM = std::pow(K, NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  /* reconstruct tensor views from pointers
   * make sure we have the values at the same offset (0) as in kernel 1 */
  auto values = mra::TensorView<T, NDIM>(&tmp[0], 2*K);
  auto r = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
  auto workspace = mra::TensorView<T, NDIM>(&tmp[2*TWOK2NDIM], K);
  auto hgT = mra::TensorView<T, 2>(hgT_ptr, 2*K, 2*K);
  auto coeffs = mra::TensorView<T, NDIM>(coeffs_ptr, K);

  T fac = std::sqrt(Domain<NDIM>:: template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
  values *= fac;
  // Inlined: filter<T,K,NDIM>(values,r);
  transform<NDIM>(values, hgT, r, workspace);

  auto child_slice = get_child_slice(key, K, 0);
  auto r_slice = r(child_slice);
  coeffs = r_slice; // extract sum coeffs
  r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
  /* TensorView assignment synchronizes */
  if (tid == 0) {
    /* TODO: compute the norm across threads */
    *is_leaf = (r.normf() < truncate_tol(key,thresh)); // test norm of difference coeffs
  }
}

template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const Fn* fn,
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  cudaStream stream)
{
  /**
   * Launch two kernels: one with multiple blocks, one with a single block.
   * We use two kernels here because of the synchronization that is required in between
   * (i.e., we have to wait for all children to be computed on before moving to the second part)
   * TODO: We could batch these together into a graph that is launched at once.
   *       Alternatively, we could call the second kernel from the first
   */

  const std::size_t K = coeffs_view.dim(0);
  dim3 thread_dims = dim3(1);
  if constexpr (NDIM >= 3) {
    thread_dims = dim3(K, K, K);
  } else if constexpr (NDIM == 2) {
    thread_dims = dim3(K, K, 1);
  } else if constexpr (NDIM == 1) {
    thread_dims = dim3(K, 1, 1);
  }

  /* launch one block per child */
  coeffs_kernel1<<<key.num_children, thread_dims, 0, stream>>>(
    fn, key, tmp, phibar_view.data(), K, thresh);
  /* launch one block only */
  coeffs_kernel2<<<1, thread_dims, 0, stream>>>(
    key, coeffs_view.data(), hgT_view.data(),
    is_leaf_scratch, K, thresh);
}

/**
 * Instantiate for 1, 2, 3 dimensional Gaussian
 */

 template
 void submit_fcoeffs_kernel<Gaussian<double, 1>, double, 1>(
   const mra::Gaussian<double, 1>* fn,
   const mra::Key<1>& key,
   mra::TensorView<double, 1>& coeffs_view,
   const mra::TensorView<double, 2>& phibar_view,
   const mra::TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   cudaStream stream);


 template
 void submit_fcoeffs_kernel<Gaussian<double, 2>, double, 2>(
   const mra::Gaussian<double, 2>* fn,
   const mra::Key<2>& key,
   mra::TensorView<double, 2>& coeffs_view,
   const mra::TensorView<double, 2>& phibar_view,
   const mra::TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   cudaStream stream);

 template
 void submit_fcoeffs_kernel<Gaussian<double, 3>, double, 3>(
   const mra::Gaussian<double, 3>* fn,
   const mra::Key<3>& key,
   mra::TensorView<double, 3>& coeffs_view,
   const mra::TensorView<double, 2>& phibar_view,
   const mra::TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   cudaStream stream);




/**
 * Compress kernels
 */

template <typename T, Dimension NDIM, typename accumulatorT>
__device__ void sumabssq(mra::TensorView<T, NDIM>& a, accumulatorT* sum) const {
  int tid = threadIdx.x + blockIdx.y + blockIdx.z;
  accumulatorT s = 0.0;
  /* play it safe: set sum to zero before the atomic increments */
  if (tid == 0) { *sum = 0.0; }
  __syncthreads();
  /* every thread computes a partial sum (likely 1 element only) */
  mra::foreach_idx(a, [&](auto... idx) mutable {
    accumulatorT x = a(idx...);
    s += x*x;
  });
  /* accumulate thread-partial results into sum
   * if we had shared memory we could use that here but for now run with atomics */
  atomicAdd(sum, s);
  __syncthreads();
  return *sum;
}

template<typename T, mra::Dimsion NDIM>
__global__ void compress_kernel(
  mra::Key<NDIM> key,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  T* sumsqs, // sumsqs[0]: sum over sumsqs of p; sumsqs[1]: sumsq of result
  const std::array<T*, mra::Key<NDIM>::num_children> in_ptrs,
  std::size_t K)
{
  bool is_t0 = !!(threadIdx.x + threadIdx.y + threadIdx.z);
  int blockid = blockIdx.x;
  {   // Collect child coeffs and leaf info
    /* construct tensors */
    const size_t K2NDIM    = std::pow(  K,NDIM);
    const size_t TWOK2NDIM = std::pow(2*K,NDIM);
    auto s = mra::TensorView<T,NDIM>(&tmp[0], 2*K);
    auto r = mra::TensorView<T,NDIM>(result_ptr, K);
    auto p = mra::TensorView<T,NDIM>(p_ptr, K);
    auto hgT = mra::TensorView<T,NDIM>(hgT_ptr, K);
    auto workspace = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM], K);
    auto child_slice = get_child_slice(key, K, blockid);
    if (is_t0) {
      sumsqs[0] = 0.0;
    }
    __
    if (blockid < key.num_children) {
      mra::TensorView<T, NDIM> in(in_ptrs[blockid], K);
      sumabssq(in, &sums[blockid]);
      s(child_slice) = in;
      //filter<T,K,NDIM>(s,d);  // Apply twoscale transformation=
      transform<NDIM>(s, hgT, r, workspace);
    }
    if (key.level() > 0 && blockid == 0) {
      p = r(child_slice);
      r(child_slice) = 0.0;
      sumabssq(r, &sums[key.num_children]); // put result sumsq at the end
    }
  }
}


template<typename T, mra::Dimsion NDIM>
void submit_compress_kernel(
  mra::Key<NDIM> key,
  mra::TensorView<T, NDIM>& p_view,
  mra::TensorView<T, NDIM>& result_view,
  const mra::TensorView<T, NDIM>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<T*, mra::Key<NDIM>::num_children>& in_ptrs,
  std::size_t K,
  cudaStream stream)
{
  const std::size_t K = p_view.dim(0);
  dim3 thread_dims = dim3(1);
  if constexpr (NDIM >= 3) {
    thread_dims = dim3(K, K, K);
  } else if constexpr (NDIM == 2) {
    thread_dims = dim3(K, K, 1);
  } else if constexpr (NDIM == 1) {
    thread_dims = dim3(K, 1, 1);
  }

  compress_kernel<<<key.num_children, thread_dims, 0, stream>>>(
    key, p_view.data(), result_view.data(), hgT_view.data(), tmp, sumsqs, in_ptrs, K);
}


/* Instantiations for 1, 2, 3D */
template
void submit_compress_kernel<double, 1>(
  mra::Key<1> key,
  mra::TensorView<double, 1>& p_view,
  mra::TensorView<double, 1>& result_view,
  const mra::TensorView<double, 1>& hgT_view,
  double* tmp,
  const std::array<double*, mra::Key<1>::num_children>& in_ptrs,
  std::size_t K,
  cudaStream stream);

template
void submit_compress_kernel<double, 2>(
  mra::Key<2> key,
  mra::TensorView<double, 2>& p_view,
  mra::TensorView<double, 2>& result_view,
  const mra::TensorView<double, 2>& hgT_view,
  double* tmp,
  const std::array<double*, mra::Key<2>::num_children>& in_ptrs,
  std::size_t K,
  cudaStream stream);

template
void submit_compress_kernel<double, 3>(
  mra::Key<3> key,
  mra::TensorView<double, 3>& p_view,
  mra::TensorView<double, 3>& result_view,
  const mra::TensorView<double, 3>& hgT_view,
  double* tmp,
  const std::array<double*, mra::Key<3>::num_children>& in_ptrs,
  std::size_t K,
  cudaStream stream);


/**
 * kernel for reconstruct
 */

template<typename T, mra::Dimension NDIM>
__global__ void reconstruct_kernel(
  mra::Key<NDIM> key,
  T* node_ptr,
  T* r_ptr,
  T* tmp,
  T* hg_ptr,
  std::size_t K)
{
  bool is_t0 = !!(threadIdx.x + threadIdx.y + threadIdx.z);
  int blockid = blockIdx.x;

  const size_t K2NDIM    = std::pow(  K,NDIM);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  auto r = mra::TensorView<T, NDIM>(r_ptr, K);
  auto s = mra::TensorView<T, NDIM>(&tmp[0], 2*K);
  auto workspace = mra::TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
  auto hg = mra::TensorView<T, NDIM>(hg_ptr, K);

  auto child_slice = get_child_slice(key, K, blockid);
  if (key.level() != 0 && blockid == 0) node.get().coeffs(child_slice) = from_parent;

  FixedTensor<T,2*K,NDIM> s;
  //unfilter<T,K,NDIM>(node.get().coeffs, s);
  transform<NDIM>(node, hg, s, workspace);

  /* TODO: split the kernel here and launch a kernel with 1<<NDIM blocks */
  r.get().coeffs = T(0.0);
  r.get().is_leaf = false;
  //::send<1>(key, r, out); // Send empty interior node to result tree
  bcast_keys[1].push_back(key);

  KeyChildren<NDIM> children(key);
  for (auto it=children.begin(); it!=children.end(); ++it) {
      const Key<NDIM> child= *it;
      r.get().key = child;
      r.get().coeffs = s(child_slices[it.index()]);
      r.get().is_leaf = node.get().is_leaf[it.index()];
      if (r.get().is_leaf) {
          //::send<1>(child, r, out);
          bcast_keys[1].push_back(child);
      }
      else {
          //::send<0>(child, r.coeffs, out);
          bcast_keys[0].push_back(child);
      }
  }
}
