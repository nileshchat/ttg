#ifndef TTG_PARSEC_FWD_H
#define TTG_PARSEC_FWD_H

#include "ttg/fwd.h"
#include "ttg/util/typelist.h"

#include <future>

extern "C" struct parsec_context_s;

namespace ttg_parsec {

  template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs = ttg::typelist<>>
  class TT;

  /// \internal the OG name
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using Op [[deprecated("use TT instead")]] = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;
  /// \internal the name in the ESPM2 paper
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using TemplateTask = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;

  class WorldImpl;

  inline void make_executable_hook(ttg::World&);

  inline void ttg_initialize(int argc, char **argv, int num_threads = -1, parsec_context_s * = nullptr);

  inline void ttg_finalize();

  [[noreturn]]
  static inline void ttg_abort();

  inline ttg::World ttg_default_execution_context();

  inline void ttg_execute(ttg::World world);

  inline void ttg_fence(ttg::World world);

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr);

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr);

  template <typename Callback>
  inline void ttg_register_callback(ttg::World world, Callback &&callback);

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world);

  inline void ttg_sum(ttg::World world, double &value);

  /// broadcast
  /// @tparam T a serializable type
  template <typename T>
  static void ttg_broadcast(ttg::World world, T &data, int source_rank);

  /* device definitions */
  template<typename T, typename Allocator = std::allocator<std::decay_t<T>>>
  struct Buffer;

  template<typename T>
  struct Ptr;

  template<typename T>
  struct devicescratch;

  template<typename T>
  struct TTValue;

  template<typename T, typename... Args>
  inline Ptr<T> make_ptr(Args&&... args);

  template<typename T>
  inline Ptr<std::decay_t<T>> get_ptr(T&& obj);

  template<typename... Views>
  inline bool register_device_memory(std::tuple<Views&...> &views);

  template<typename... Buffer>
  inline void post_device_out(std::tuple<Buffer&...> &b);

  template<typename... Buffer>
  inline void mark_device_out(std::tuple<Buffer&...> &b);

  inline int num_devices();

#if 0
  template<typename... Args>
  inline std::pair<bool, std::tuple<ptr<std::decay_t<Args>>...>> get_ptr(Args&&... args);
#endif

}  // namespace ttg_parsec

#endif  // TTG_PARSEC_FWD_H
