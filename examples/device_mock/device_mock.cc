// clang-format off

#include <ttg.h>
#include <ttg/view.h>
#include "../matrixtile.h"

#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>

#include "ttg/util/meta.h"

#include <ttg/util/multiindex.h>

using Key2 = ttg::MultiIndex<2>;
using Key3 = ttg::MultiIndex<3>;

/* number of tiles */
#define KT 100

template<typename T>
auto make_gemm(ttg::Edge<Key3, MatrixTile<T>>& A,
               ttg::Edge<Key3, MatrixTile<T>>& B,
               ttg::Edge<Key2, MatrixTile<T>>& output_result)
{

  ttg::Edge<Key3, MatrixTile<double>> C;
  auto f_cpu = [=](const Key3& key,
                   const MatrixTile<T>&  A,
                   const MatrixTile<T>&  B,
                   MatrixTile<T>&  C,
                   std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                              ttg::Out<Key3, MatrixTile<T>>>& out)
  {
    int m = key[0];
    int n = key[1];
    int k = key[2];

    /*
    if(k == 0) {
      dlprng(C.data(), 1789, A.mb()*B.nb());
    }
    dgemm(A.data(), A.mb(), A.nb(),
          B.data(), B.mb(), B.nb(),
          1.0,
          C.data(), C.nb(), C.nb());
    */

    if( k == KT-1 || C.data()[0] < 1e-9 ) {
      ttg::send<0>(Key2{m, n}, std::move(C));
    } else {
      ttg::send<1>(Key3{m, n, k+1}, std::move(C));
    }
  };

  auto f_gpu_host_views = [=](const Key3& key,
                              const MatrixTile<T>& A,
                              const MatrixTile<T>& B,
                              MatrixTile<T>&& C)
  {
    // The default ViewScope::SyncIn scope tells the runtime that the data should be copied
    // to the device before the kernel callable is invoked.
    ttg::View<const MatrixTile<T>, const T> dev_A = ttg::make_view( A, ttg::ViewSpan(A.data(), A.size()) );
    ttg::View<const MatrixTile<T>, const T> dev_B = ttg::make_view( B, ttg::ViewSpan(B.data(), B.size()) );
    ttg::View<MatrixTile<T>, T> dev_C;
    ttg::View<T, T> dev_tmp;
    T *host_tmp = new(T);
    // ViewScope::SyncOut tells the runtime system that the view should be synchronized back to the
    // host before invoking the output callable.
    dev_tmp = ttg::make_view( *host_tmp, ttg::ViewSpan(host_tmp, 1, ttg::ViewScope::SyncOut) );

    int k = key[2];
    if(0 == k) {
      // ViewScope::Allocate tells the runtime system that the device view needs to be allocated but doesn't need to be
      // initialized with C.data(). However, C.data() is still associated with the device memory, so if the
      // runtime system evicts that data from the device, it will be first copied back into C.data().
      dev_C = ttg::make_view( C, ttg::ViewSpan(C.data(), C.size(), ttg::ViewScope::Allocate) );
    } else {
      dev_C = ttg::make_view( C, ttg::ViewSpan(C.data(), C.size()) );
    }

    return std::make_tuple(dev_A, dev_B, dev_C, dev_tmp);
  };

  auto f_gpu_kernel = [=](const Key3& key,
                                ttg::View<const MatrixTile<T>, const T>& dev_A,
                                ttg::View<const MatrixTile<T>, const T>& dev_B,
                                ttg::View<MatrixTile<T>, T>& dev_C,
                          ttg::View<T, T>& dev_tmp)
  {
    int k = key[2];

    const MatrixTile<T>&  A = dev_A.get_host_object();
    const MatrixTile<T>&  B = dev_B.get_host_object();
    MatrixTile<T>&  C = dev_C.get_host_object();
    T& host_tmp = dev_tmp.get_host_object();
    auto beta = 1.0;
    if(k == 0) {
        //cublasDplrng(dev_C.get(0), C.mb(), C.nb());
    }

    /*
    cublasDgemm(dev_A.get(0), A.mb(), A.nb(),
                dev_B.get(0), B.mb(), B.nb(),
                beta,
                dev_C.get(0), C.mb(), C.nb());

    cudaMemcpyAsync(&dev_C.get(0)[0], host_tmp, sizeof(T), cudaDeviceToHost);
    */
  };

  auto f_gpu_output_flows = [=](const Key3& key,
                                const MatrixTile<T>&  A,
                                const MatrixTile<T>&  B,
                                MatrixTile<T>&  C,
                                T& host_tmp)
  {
    int m = key[0];
    int n = key[1];
    int k = key[2];

    if( k == KT-1 || host_tmp < 1e-9 ) {
        ttg::send<0>(Key2{m, n}, std::move(C));
    } else {
      ttg::send<1>(Key3{m, n, k+1}, std::move(C));
    }
    delete &host_tmp;
  };

  //ttg::meta::type_printer<decltype(ttg::edges(output_result, C))> x;

  /* If we only have GPU */
  auto gemm_tt = ttg::make_device_tt<Key3>(f_gpu_host_views, f_gpu_kernel, f_gpu_output_flows, ttg::ExecutionSpace::CUDA,
                                     ttg::edges(A, B, C), ttg::edges(output_result, C),
                                     "GEMM", {"A", "B", "C"}, {"output_result", "C"});

#if 0
  /* Alternative: to get both type of tasklets: */
  auto gemm_tt = ttg::make_device_tt(f_cpu, f_gpu_host_views, f_gpu_kernel, f_gpu_output_flows, ttg::ExecutionSpace::CUDA,
                                     ttg::edges(A, B), ttg::edges(output_result, C),
                                     "GEMM", {"A", "B"}, {"output_result", "C"});
#endif
  return gemm_tt;
}

int main(int argc, char **argv)
{

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int N = 1024;
  int M = N;
  int NB = 128;
  int check = 0;
  int nthreads = -1;
  const char* prof_filename = nullptr;

  if (argc > 1) {
    N = M = atoi(argv[1]);
  }

  if (argc > 2) {
    NB = atoi(argv[2]);
  }

  if (argc > 3) {
    check = atoi(argv[3]);
  }

  if (argc > 4) {
    nthreads = atoi(argv[4]);
  }

  ttg::initialize(argc, argv, nthreads);

  auto world = ttg::default_execution_context();

  ttg::Edge<Key3, MatrixTile<double>> edge_a, edge_b;
  ttg::Edge<Key2, MatrixTile<double>> edge_out;

  auto gemm_tt = make_gemm(edge_a, edge_b, edge_out);



  ttg::finalize();
  return 0;
}
