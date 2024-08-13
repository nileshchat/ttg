#include <ttg.h>
#include "tensor.h"
#include "tensorview.h"
#include "functionnode.h"
#include "functiondata.h"
#include "kernels.h"
#include "gaussian.h"
#include "functionfunctor.h"
#include "../../mrakey.h"

#if 0
/// Project the scaling coefficients using screening and test norm of difference coeffs.  Return true if difference coeffs negligible.
template <typename FnT, typename T, mra::Dimension NDIM>
static bool fcoeffs(
  const ttg::Buffer<FnT>& f,
  const mra::FunctionData<T, NDIM>& functiondata,
  const mra::Key<NDIM>& key,
  const T thresh,
  mra::Tensor<T,NDIM>& coeffs)
{
  bool status;

  if (mra::is_negligible(*f.host_ptr(),mra::Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
    coeffs = 0.0;
    status = true;
  }
  else {

    /* global function data */
    // TODO: need to make our own FunctionData with dynamic K
    const auto& phibar = functiondata.get_phibar();
    const auto& hgT = functiondata.get_hgT();

    const std::size_t K = coeffs.dim(0);

    /* temporaries */
    bool is_leaf;
    auto is_leaf_scratch = ttg::make_scratch(&is_leaf, ttg::scope::Allocate);
    const std::size_t tmp_size = project_tmp_size<NDIM>(K);
    T* tmp = new T[tmp_size]; // TODO: move this into make_scratch()
    auto tmp_scratch = ttg::make_scratch(tmp, ttg::scope::Allocate, tmp_size);

    /* TODO: cannot do this from a function, need to move it into the main task */
    co_await ttg::device::select(f, coeffs.buffer(), phibar.buffer(), hgT.buffer(), tmp, is_leaf_scratch);
    auto coeffs_view = coeffs.current_view();
    auto phibar_view = phibar.current_view();
    auto hgT_view    = hgT.current_view();
    T* tmp_device = tmp_scratch.device_ptr();
    bool *is_leaf_device = is_leaf_scratch.device_ptr();
    FnT* f_ptr = f.current_device_ptr();

    /* submit the kernel */
    submit_fcoeffs_kernel(f_ptr, key, coeffs_view, phibar_view, hgT_view, tmp_device,
                          is_leaf_device, ttg::device::current_stream());

    /* wait and get is_leaf back */
    co_await ttg::device::wait(is_leaf_scratch);
    status = is_leaf;
    /* todo: is this safe? */
    delete[] tmp;
  }
  co_return status;
}
#endif // 0

template<typename FnT, typename T, mra::Dimension NDIM>
auto make_project(
  ttg::Buffer<FnT>& f,
  const mra::FunctionData<T, NDIM>& functiondata,
  const T thresh, /// should be scalar value not complex
  ttg::Edge<mra::Key<NDIM>, void> control,
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, NDIM>> result)
{

  auto fn = [&](const mra::Key<NDIM>& key) -> ttg::device::Task {
    using tensor_type = typename mra::Tensor<T, NDIM>;
    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionReconstructedNode<T, NDIM>;
    node_type result;
    tensor_type& coeffs = result.coeffs;

    if (key.level() < initial_level(f)) {
      std::vector<mra::Key<NDIM>> bcast_keys;
      /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                We need to fix broadcast to support any ranges */
      for (auto child : children(key)) bcast_keys.push_back(child);
      ttg::broadcastk<0>(bcast_keys);
      coeffs.current_view() = T(1e7); // set to obviously bad value to detect incorrect use
      result.is_leaf = false;
    }
    else if (mra::is_negligible<FnT,T,NDIM>(*f.host_ptr(), mra::Domain<NDIM>:: template bounding_box<T>(key), mra::truncate_tol(key,thresh))) {
      /* zero coeffs */
      coeffs.current_view() = T(0.0);
      result.is_leaf = true;
    }
    else {
      /* here we actually compute: first select a device */
      //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
      /**
       * BEGIN FCOEFFS HERE
       * TODO: figure out a way to outline this into a function or coroutine
       */

      /* global function data */
      // TODO: need to make our own FunctionData with dynamic K
      const auto& phibar = functiondata.get_phibar();
      const auto& hgT = functiondata.get_hgT();

      const std::size_t K = coeffs.dim(0);

      /* temporaries */
      bool is_leaf;
      auto is_leaf_scratch = ttg::make_scratch(&is_leaf, ttg::scope::Allocate);
      const std::size_t tmp_size = project_tmp_size<NDIM>(K);
      T* tmp = new T[tmp_size]; // TODO: move this into make_scratch()
      auto tmp_scratch = ttg::make_scratch(tmp, ttg::scope::Allocate, tmp_size);

      /* TODO: cannot do this from a function, need to move it into the main task */
      co_await ttg::device::select(f, coeffs.buffer(), phibar.buffer(),
                                   hgT.buffer(), tmp_scratch, is_leaf_scratch);
      auto coeffs_view = coeffs.current_view();
      auto phibar_view = phibar.current_view();
      auto hgT_view    = hgT.current_view();
      T* tmp_device = tmp_scratch.device_ptr();
      bool *is_leaf_device = is_leaf_scratch.device_ptr();
      FnT* f_ptr = f.current_device_ptr();

      /* submit the kernel */
      submit_fcoeffs_kernel(f_ptr, key, coeffs_view, phibar_view, hgT_view, tmp_device,
                            is_leaf_device, ttg::device::current_stream());

      /* wait and get is_leaf back */
      co_await ttg::device::wait(is_leaf_scratch);
      result.is_leaf = is_leaf;
      /* todo: is this safe? */
      delete[] tmp;
      /**
       * END FCOEFFS HERE
       */

      if (!result.is_leaf) {
        std::vector<mra::Key<NDIM>> bcast_keys;
        for (auto child : children(key)) bcast_keys.push_back(child);
        ttg::broadcastk<0>(bcast_keys);
      }
    }
    ttg::send<1>(key, std::move(result)); // always produce a result
  };

  return ttg::make_tt(std::move(fn), ttg::edges(control), ttg::edges(result));
}

template<std::size_t NDIM, typename Value, std::size_t I, std::size_t Is...>
static void select_compress_send(const mra::Key<NDIM>& parent, Value&& value,
                                 std::size_t child_idx,
                                 std::index_sequence<I, Is...>) {
  if (child_idx == I) {
    return ttg::device::send<I>(parent, std::forward<Value>(p));
  } else {
    return select_compress_send(parent, std::forward<Value>(p), child_idx, std::index_sequence<Is...>);
  }
}

// With data streaming up the tree run compression
template <typename T, Dimension NDIM>
static void do_compress(const Key<NDIM>& key,
                        const auto&... input_frns
                 /*const FunctionReconstructedNode<T,K,NDIM> &in0,
                   const FunctionReconstructedNode<T,K,NDIM> &in1,
                   const FunctionReconstructedNode<T,K,NDIM> &in2,
                   const FunctionReconstructedNode<T,K,NDIM> &in3,
                   const FunctionReconstructedNode<T,K,NDIM> &in4,
                   const FunctionReconstructedNode<T,K,NDIM> &in5,
                   const FunctionReconstructedNode<T,K,NDIM> &in6,
                   const FunctionReconstructedNode<T,K,NDIM> &in7*/) {
  //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
  //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
    auto& child_slices = FunctionData<T,NDIM>::get_child_slices();
    constexpr const auto num_children = Key<NDIM>::num_children;
    auto K = in0.coeffs.dim(0);
    mra::FunctionCompressedNode<T,NDIM> result(key, K); // The eventual result
    auto& d = result.coeffs;
    // allocate even though we might not need it
    mra::FunctionReconstructedNode<T, NDIM> p(key, K);

    /* stores sumsq for each child and for result at the end of the kernel */
    const std::size_t tmp_size = project_tmp_size<NDIM>(K);
    T* tmp = new T[tmp_size];
    const auto& hgT = functiondata.get_hgT();
    auto tmp_scratch = ttg::device::make_scratch(tmp, ttg::scope::Allocate, tmp_size);
    T sumsqs[num_children+1];
    auto sumsqs_scratch = ttg::device::make_scratch(sumsqs, ttg::scope::Allocate, num_children+1);

    /* wait for transfers to complete */
    ttg::device::select(p.coeffs.buffer(), d.buffer(), hgT.buffer(), tmp_scratch,
                        sumsqs_scratch, (input_frns.coeffs.buffer())...);

    /* assemble input array and submit kernel */
    std::array<T*, num_children> ins = {(input_frns.coeffs.buffer().current_device_ptr())...};
    submit_compress_kernel(key, p.coeffs.current_view(), d.current_view(), hgT.current_view(),
                           tmp_scratch.device_ptr(), sumsqs_scratch.device_ptr(), ins,
                           K, ttg::device::current_stream());

    /* wait for kernel and transfer sums back */
    co_await ttg::device::wait(sumsqs_scratch);
    T sumsq = 0.0;
    T d_sumsq = sumsqs[num_children];
    {  // Collect child leaf info
      result.is_child_leaf = { input_frns.is_leaf... };
      for (size_t i : range(num_children)) {
        sumsq += sumsqs[i]; // Accumulate sumsq from child difference coeffs
      }
    }

    // Recur up
    if (key.level() > 0) {
      p.sum = tmp[num_children] + sumsq; // result sumsq is last element in sumsqs

      // will not return
      co_await ttg::device::forward(
        // select to which child of our parent we send
        ttg::device::send<0>(key, std::move(p)),
        // Send result to output tree
        ttg::device::send<1>(key, std::move(result)));
    } else {
      std::cout << "At root of compressed tree: total normsq is " << sumsq + d_sumsq << std::endl;
      co_await ttg::device::forward(
        // Send result to output tree
        ttg::device::send<1>(key, std::move(result)));
    }
}

/* forward a reconstructed function node to the right input of do_compress
 * this is a device task to prevent data from being pulled back to the host
 * even though it will not actually perform any computation */
template<typename T, mra::Dimension NDIM>
static ttg::device::Task do_send_leafs_up(const Key<NDIM>& key, const mra::FunctionReconstructedNode<T, NDIM>& node) {
  co_await select_compress_send(key.parent, p, key.childindex(), std::index_sequence<Key<NDIM>::num_children>{});
}


/// Make a composite operator that implements compression for a single function
template <typename T, Dimension NDIM>
static auto make_compress(
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>> in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionCompressedNode<T, NDIM>> out) {
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, NDIM>> recur("recur");

  constexpr const std::size_t num_children = mra::Key<NDIM>::num_children;
  // creates the right number of edges for nodes to flow from send_leafs_up to compress
  // send_leafs_up will select the right input for compress
  auto create_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return ttg::edges((Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>>{})...);
  };
  auto send_to_compress_edges = create_edges(std::index_sequence<num_children>{});
  return std::make_tuple(ttg::make_tt(&do_send_leafs_up<T,NDIM>, edges(ttg::fuse(recur, in)), send_to_compress_edges, "send_leaves_up"),
                         ttg::make_tt(&do_compress<T,NDIM>, send_to_compress_edges, edges(recur,out), "do_compress"));
}



template <typename T, Dimension NDIM>
void do_reconstruct(const mra::Key<NDIM>& key,
                    mra::FunctionCompressedNode<T, NDIM>& node,
                    const mra::Tensor<T, NDIM> from_parent) {
  const auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
  if (key.level() != 0) node.get().coeffs(child_slices[0]) = from_parent;

  auto s = mra::Tensor<T,NDIM>(2*K);
  auto r = FunctionReconstructedNode<T,NDIM>(key, K);
  r.get().coeffs = T(0.0);
  r.get().is_leaf = false;
  //::send<1>(key, r, out); // Send empty interior node to result tree
  bcast_keys[1].push_back(key);

  KeyChildren<NDIM> children(key);
  for (auto it=children.begin(); it!=children.end(); ++it) {
      const mra::Key<NDIM> child= *it;
      r.get().key = child;
      r.get().coeffs = s(child_slices[it.index()]);
      r.get().is_leaf = node.get().is_leaf[it.index()];
      if (r.get().is_leaf) {
          ::send<1>(child, r, out);
          //bcast_keys[1].push_back(child);
      }
      else {
          ::send<0>(child, r.coeffs, out);
          //bcast_keys[0].push_back(child);
      }
  }
  //ttg::broadcast<0>(bcast_keys[0], r.get().coeffs, out);
  //ttg::broadcast<1>(bcast_keys[1], std::move(r), out);
}


template <typename T, size_t K, Dimension NDIM>
auto make_reconstruct(const cnodeEdge<T,K,NDIM>& in, rnodeEdge<T,K,NDIM>& out, const std::string& name = "reconstruct") {
  ttg::Edge<Key<NDIM>,FixedTensor<T,K,NDIM>> S("S");  // passes scaling functions down

  auto s = ttg::make_tt_tpl(&do_reconstruct<T,K,NDIM>, ttg::edges(in, S), ttg::edges(S, out), name, {"input", "s"}, {"s", "output"});

  if (ttg::default_execution_context().rank() == 0) {
    s->template in<1>()->send(Key<NDIM>{0,{0}}, FixedTensor<T,K,NDIM>()); // Prime the flow of scaling functions
  }

  return s;
}



template<typename T, mra::Dimension NDIM>
void test(std::size_t K) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  mra::Domain<NDIM>::set_cube(-6.0,6.0);

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T, NDIM>> project_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionCompressedNode<T, NDIM>> compress_result;

  // define a Gaussian
  auto gaussian = mra::Gaussian<T, NDIM>(T(3.0), {T(0.0),T(0.0),T(0.0)});
  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(&gaussian);
  auto project = make_project(gauss_buffer, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(project_result, compress_result);
  auto reconstruct = make_reconstruct(compress_result);
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);

  test<double, 3>(10);

  ttg::finalize();
}