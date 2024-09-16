#ifndef TTG_MRA_TENSOR_H
#define TTG_MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>
#include <ttg/serialization.h>
#include <ttg/serialization/std/array.h>
#include <madness/world/world.h>

#include "tensorview.h"

namespace mra {

  template<typename T, Dimension NDIM, class Allocator = std::allocator<T>>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Allocator>> {
  public:
    using value_type = std::decay_t<T>;
    using size_type = std::size_t;
    using allocator_type = Allocator;
    using view_type = TensorView<value_type, NDIM>;
    using const_view_type = std::add_const_t<TensorView<value_type, NDIM>>;
    using buffer_type = ttg::Buffer<value_type, allocator_type>;

    static constexpr Dimension ndim() { return NDIM; }

    using dims_array_t = std::array<size_type, ndim()>;

    //template<typename Archive>
    //friend madness::archive::ArchiveSerializeImpl<Archive, Tensor>;

  private:
    using ttvalue_type = ttg::TTValue<Tensor<T, NDIM, Allocator>>;
    dims_array_t m_dims;
    buffer_type m_buffer;

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(size());
    }

    template<std::size_t... Is>
    static auto create_dims_array(std::size_t dim, std::index_sequence<Is...>) {
      return std::array{(Is, dim)...};
    }

  public:
    Tensor() = default;

    /* generic */
    explicit Tensor(std::size_t dim)
    : ttvalue_type()
    , m_dims(create_dims_array(dim, std::make_index_sequence<NDIM>{}))
    , m_buffer(size())
    { }

    template<typename... Dims, typename = std::enable_if_t<(sizeof...(Dims) > 1)>>
    Tensor(Dims... dims)
    : ttvalue_type()
    , m_dims({static_cast<size_type>(dims)...})
    , m_buffer(size())
    {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }


    Tensor(Tensor<T, NDIM, Allocator>&& other) = default;

    Tensor& operator=(Tensor<T, NDIM, Allocator>&& other) = default;

    /* Deep copy ctor und op are not needed for PO since tiles will never be read
    * and written concurrently. Hence shallow copies are enough, will all
    * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
    * can handle data sharing without excessive copying */
    Tensor(const Tensor<T, NDIM, Allocator>& other)
    : ttvalue_type()
    , m_dims(other.m_dims)
    , m_buffer(other.size())
    {
      std::copy_n(other.data(), size(), this->data());
    }

    Tensor& operator=(const Tensor<T, NDIM, Allocator>& other) {
      m_dims = other.m_dims;
      this->realloc();
      std::copy_n(other.data(), size(), this->data());
      return *this;
    }

    size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    size_type dim(Dimension dim) const {
      return m_dims[dim];
    }

    auto& buffer() {
      return m_buffer;
    }

    const auto& buffer() const {
      return m_buffer;
    }

    value_type* data() {
      return m_buffer.host_ptr();
    }

    const value_type* data() const {
      return m_buffer.host_ptr();
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    view_type current_view() {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    const view_type current_view() const {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar &m_dims &m_buffer;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }
  };

  template <typename tensorT>
  requires(tensorT::is_tensor)
  std::ostream&
  operator<<(std::ostream& s, const tensorT& t) {
    if (t.size() == 0) {
      s << "[empty tensor]\n";
      return s;
    }

    const Dimension ndim = t.ndim();

    auto dims = t.dims();
    size_t maxdim = std::max_element(dims.begin(), dims.end());
    size_t index_width = std::max(std::log10(maxdim), 6.0);
    std::ios::fmtflags oldflags = s.setf(std::ios::scientific);
    long oldprec = s.precision();
    long oldwidth = s.width();

    const Dimension lastdim = ndim-1;
    const size_t lastdimsize = t.dim(lastdim);

    for (auto it=t.begin(); it!=t.end(); ) {
      const auto& index = it.index();
      s.unsetf(std::ios::scientific);
      s << '[';
      for (Dimension d=0; d<(ndim-1); d++) {
        s.width(index_width);
        s << index[d];
        s << ",";
      }
      s << " *]";
      // s.setf(std::ios::scientific);
      s.setf(std::ios::fixed);
      for (size_t i=0; i<lastdimsize; ++i,++it) { //<<< it incremented here!
        // s.precision(4);
        s << " ";
        //s.precision(8);
        //s.width(12);
        s.precision(6);
        s.width(10);
        s << *it;
      }
      s.unsetf(std::ios::scientific);
      if (it != t.end()) s << std::endl;
    }

    s.setf(oldflags,std::ios::floatfield);
    s.precision(oldprec);
    s.width(oldwidth);

    return s;
  }
} // namespace mra

#endif // TTG_MRA_TENSOR_H