#ifndef TTG_PARSEC_DEVICEFUNC_H
#define TTG_PARSEC_DEVICEFUNC_H

#if defined(TTG_HAVE_CUDART)
#include <cuda.h>
#endif

#include "ttg/parsec/task.h"
#include <parsec.h>
#include <parsec/mca/device/device_gpu.h>

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <parsec/mca/device/cuda/device_cuda.h>
#elif defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
#include <parsec/mca/device/hip/device_hip.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT

namespace ttg_parsec {
  namespace detail {
    template<typename... Views, std::size_t I, std::size_t... Is>
    inline bool register_device_memory(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {
      static_assert(I < MAX_PARAM_COUNT,
                    "PaRSEC only supports MAX_PARAM_COUNT device input/outputs. "
                    "Increase MAX_PARAM_COUNT and recompile PaRSEC/TTG.");
      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;
      parsec_ttg_task_base_t *caller = detail::parsec_ttg_caller;
      assert(nullptr != caller->dev_ptr);
      parsec_gpu_task_t *gpu_task = caller->dev_ptr->gpu_task;
      parsec_flow_t *flows = caller->dev_ptr->flows;

      auto& view = std::get<I>(views);
      bool is_current = false;
      static_assert(ttg::meta::is_buffer_v<view_type> || ttg::meta::is_devicescratch_v<view_type>);
      /* get_parsec_data is overloaded for buffer and devicescratch */
      parsec_data_t* data = detail::get_parsec_data(view);
      /* TODO: check whether the device is current */

      if (nullptr != data) {
        auto access = PARSEC_FLOW_ACCESS_RW;
        if constexpr (std::is_const_v<view_type>) {
          // keep the flow at RW if it was RW to make sure we pull the data back out eventually
          //if (flows[I].flow_flags != PARSEC_FLOW_ACCESS_RW) {
            access = PARSEC_FLOW_ACCESS_READ;
          //}
        } else if constexpr (ttg::meta::is_devicescratch_v<view_type>) {
          if (view.scope() == ttg::scope::Allocate) {
            access = PARSEC_FLOW_ACCESS_WRITE;
          }
        }

        //std::cout << "register_device_memory task " << detail::parsec_ttg_caller << " data " << I << " "
        //          << data << " size " << data->nb_elts << std::endl;

        /* build the flow */
        /* TODO: reuse the flows of the task class? How can we control the sync direction then? */
        flows[I] = parsec_flow_t{.name = nullptr,
                                .sym_type = PARSEC_SYM_INOUT,
                                .flow_flags = static_cast<uint8_t>(access),
                                .flow_index = I,
                                .flow_datatype_mask = ~0 };

        gpu_task->flow_nb_elts[I] = data->nb_elts; // size in bytes
        gpu_task->flow[I] = &flows[I];

        /* set the input data copy, parsec will take care of the transfer
        * and the buffer will look at the parsec_data_t for the current pointer */
        //detail::parsec_ttg_caller->parsec_task.data[I].data_in = data->device_copies[data->owner_device];
        assert(nullptr != data->device_copies[0]->original);
        caller->parsec_task.data[I].data_in = data->device_copies[0];
        caller->parsec_task.data[I].source_repo_entry = NULL;
      } else {
        /* ignore the flow */
        flows[I] = parsec_flow_t{.name = nullptr,
                                 .sym_type = PARSEC_FLOW_ACCESS_NONE,
                                 .flow_flags = 0,
                                 .flow_index = I,
                                 .flow_datatype_mask = ~0 };
        gpu_task->flow[I] = &flows[I];
        gpu_task->flow_nb_elts[I] = 0; // size in bytes
        caller->parsec_task.data[I].data_in = nullptr;

      }

      if constexpr (sizeof...(Is) > 0) {
        is_current |= register_device_memory(views, std::index_sequence<Is...>{});
      }
      return is_current;
    }
  } // namespace detail

  /* Takes a tuple of ttg::Views or ttg::buffers and register them
   * with the currently executing task. Returns true if all memory
   * is current on the target device, false if transfers are required. */
  template<typename... Views>
  inline bool register_device_memory(std::tuple<Views&...> &views) {
    bool is_current = true;
    if (nullptr == detail::parsec_ttg_caller) {
      throw std::runtime_error("register_device_memory may only be invoked from inside a task!");
    }

    if (nullptr == detail::parsec_ttg_caller->dev_ptr) {
      throw std::runtime_error("register_device_memory called inside a non-gpu task!");
    }

    if constexpr (sizeof...(Views) > 0) {
      is_current = detail::register_device_memory(views, std::index_sequence_for<Views...>{});
    }

    /* reset all entries in the current task */
    for (int i = sizeof...(Views); i < MAX_PARAM_COUNT; ++i) {
      detail::parsec_ttg_caller->parsec_task.data[i].data_in = nullptr;
      detail::parsec_ttg_caller->dev_ptr->flows[i].flow_flags = PARSEC_FLOW_ACCESS_NONE;
      detail::parsec_ttg_caller->dev_ptr->flows[i].flow_index = i;
      detail::parsec_ttg_caller->dev_ptr->gpu_task->flow[i] = &detail::parsec_ttg_caller->dev_ptr->flows[i];
      detail::parsec_ttg_caller->dev_ptr->gpu_task->flow_nb_elts[i] = 0;
    }

    return is_current;
  }

  namespace detail {
    template<typename... Views, std::size_t I, std::size_t... Is, bool DeviceAvail = false>
    inline void mark_device_out(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {

      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;
      auto& view = std::get<I>(views);

      /* get_parsec_data is overloaded for buffer and devicescratch */
      parsec_data_t* data = detail::get_parsec_data(view);
      parsec_gpu_task_t *gpu_task = detail::parsec_ttg_caller->dev_ptr->gpu_task;
      parsec_gpu_exec_stream_t *stream = detail::parsec_ttg_caller->dev_ptr->stream;

      if (data->owner_device != 0) {
        /* enqueue the transfer into the compute stream to come back once the compute and transfer are complete */
        parsec_device_gpu_module_t *device_module = detail::parsec_ttg_caller->dev_ptr->device;
        device_module->memcpy_async(device_module, stream,
                                    data->device_copies[0]->device_private,
                                    data->device_copies[data->owner_device]->device_private,
                                    data->nb_elts, parsec_device_gpu_transfer_direction_d2h);
      }
      if constexpr (sizeof...(Is) > 0) {
        // recursion
        mark_device_out(views, std::index_sequence<Is...>{});
      }
    }
  } // namespace detail

  template<typename... Buffer>
  inline void mark_device_out(std::tuple<Buffer&...> &b) {

    if (nullptr == detail::parsec_ttg_caller) {
      throw std::runtime_error("mark_device_out may only be invoked from inside a task!");
    }

    if (nullptr == detail::parsec_ttg_caller->dev_ptr) {
      throw std::runtime_error("mark_device_out called inside a non-gpu task!");
    }

    detail::mark_device_out(b, std::index_sequence_for<Buffer...>{});
  }

  namespace detail {

    template<typename... Views, std::size_t I, std::size_t... Is>
    inline void post_device_out(std::tuple<Views&...> &views, std::index_sequence<I, Is...>) {

      using view_type = std::remove_reference_t<std::tuple_element_t<I, std::tuple<Views&...>>>;

      if constexpr (!std::is_const_v<view_type>) {
        auto& view = std::get<I>(views);

        /* get_parsec_data is overloaded for buffer and devicescratch */
        parsec_data_t* data = detail::get_parsec_data(view);
        data->device_copies[0]->version = data->device_copies[data->owner_device]->version;
        parsec_data_transfer_ownership_to_copy(data, 0, PARSEC_FLOW_ACCESS_READ);
      }

      if constexpr (sizeof...(Is) > 0) {
        // recursion
        post_device_out(views, std::index_sequence<Is...>{});
      }
    }
  } // namespace detail
  template<typename... Buffer>
  inline void post_device_out(std::tuple<Buffer&...> &b) {
    detail::post_device_out(b, std::index_sequence_for<Buffer...>{});
  }

} // namespace ttg_parsec

#endif // TTG_PARSEC_DEVICEFUNC_H
