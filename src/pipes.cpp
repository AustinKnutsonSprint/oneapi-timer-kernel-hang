//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <thread>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
#include <CL/sycl/intel/fpga_extensions.hpp>
namespace INTEL = sycl::intel; // Namespace alias for backward compatibility
#else
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

using EventMessagePipe = INTEL::pipe<
    class EventMessagePipeClass,
    int,
    4>;

using TimerPipe = INTEL::pipe<
   class TimerPipeClass,
   unsigned long long,
   4>;

using HostTimerPipe = INTEL::pipe<
   class HostTimerPipeClass,
   int,
   4>;

// Forward declare the kernel names
// (This will become unnecessary in a future compiler version.)
class PersistentKernel;
class EventKernel;
class HostKernel;
class TimerKernel;

int main(int argc, char *argv[])
{
   double fmax = 100;

   if (argc > 1)
   {
      std::string option(argv[1]);
      if (option == "-h" || option == "--help")
      {
         std::cout << "Usage: \n<executable> <data size>\n\nFAILED\n";
         return 1;
      }
      else
      {
         fmax = std::stod(option);
      }
   }

   std::cout << "fmax:  " << fmax << "\n";

   try
   {
      #if defined(FPGA_EMULATOR)
      INTEL::fpga_emulator_selector device_selector;
      #else
      INTEL::fpga_selector device_selector;
      #endif
      queue q(device_selector, dpc_common::exception_handler);

      size_t num_items = 8;
      size_t num_bytes = num_items * sizeof(unsigned long long);
      unsigned long long *device_data = malloc_device<unsigned long long>(num_items, q);
      unsigned long long *host_data = (unsigned long long *)malloc(num_bytes);
      unsigned long long volatile *timer_data = malloc_device<unsigned long long>(1, q);

      q.memset(device_data, 0, num_bytes).wait();
      memset(host_data, 0, num_bytes);
      q.memset((void *)timer_data, 0, sizeof(unsigned long long));

      auto persistent_e = q.submit([&](handler &h) {
         h.single_task<PersistentKernel>([=]() {
            [[intelfpga::disable_loop_pipelining]]
            while (true)
            {
               bool has_event_msg = false;
               int message = EventMessagePipe::read(has_event_msg);
               if (has_event_msg)
               {
                  ONEAPI::atomic_fence(ONEAPI::memory_order::acquire, ONEAPI::memory_scope::device);
                  device_data[message] = *timer_data;
                  ONEAPI::atomic_fence(ONEAPI::memory_order::release, ONEAPI::memory_scope::device);
                  if (message == 0)
                     break;
               }

               bool has_timer = false;
               TimerPipe::read(has_timer);
               if (has_timer)
               {
                  HostTimerPipe::write(1);
               }
            }
         });
      });

      unsigned long long fmax_sec(fmax * 1000000);
      std::cout << "fmax_sec: " << fmax_sec << std::endl;

      q.submit([&](handler &h) {
         h.single_task<TimerKernel>([=]() {
            *timer_data = 0;
            [[intelfpga::disable_loop_pipelining]]
            for (int ticks = 0; ticks < 10; ++ticks)
            {
               unsigned long long i = 0;
               [[intelfpga::disable_loop_pipelining]]
               while (true)
               {
                  if (++i > fmax_sec)
                     break;
               }
               ONEAPI::atomic_fence(ONEAPI::memory_order::acquire, ONEAPI::memory_scope::device);
               *timer_data = *timer_data + 1;
               ONEAPI::atomic_fence(ONEAPI::memory_order::release, ONEAPI::memory_scope::device);
               TimerPipe::write(1);
            }
         });
      });

      auto start = std::chrono::system_clock::now();
      for (int ticks = 0; ticks < 10; ++ticks)
      {
         auto host_e = q.submit([&](handler &h) {
            h.single_task<HostKernel>([=]() {
               HostTimerPipe::read();
            });
         });
         host_e.wait();
         auto end = std::chrono::system_clock::now();
         std::cout << ticks << ": " << std::chrono::duration<double>(end - start).count() << std::endl;
         start = end;
      }

      std::cout << "Sending shutdown message to persistent kernel" << std::endl;

      q.submit([&](handler &h) {
         h.single_task<EventKernel>([=]() {
            EventMessagePipe::write(0);
         });
      });

      std::cout << "Waiting for persistent kernel shutdown" << std::endl;
      persistent_e.wait();
      std::cout << "Persistent kernel shutdown" << std::endl;

      q.memcpy(host_data, device_data, num_bytes).wait();

      for (size_t i = 0; i < num_items; ++i)
         std::cout << host_data[i] << std::endl;

      std::cout << "Freeing memory" << std::endl;
      sycl::free(device_data, q);
      std::free(host_data);
      sycl::free((void *) timer_data, q);

      std::cout << "Success" << std::endl;
      return 0;
   }
   catch (sycl::exception const &e)
   {
      // Catches exceptions in the host code
      std::cout << "Caught a SYCL host exception:\n"
                << e.what() << "\n";

      // Most likely the runtime couldn't find FPGA hardware!
      if (e.get_cl_code() == CL_DEVICE_NOT_FOUND)
      {
         std::cout << "If you are targeting an FPGA, please ensure that your "
                      "system has a correctly configured FPGA board.\n";
         std::cout << "If you are targeting the FPGA emulator, compile with "
                      "-DFPGA_EMULATOR.\n";
      }
      std::terminate();
   }

   return 0;
}
