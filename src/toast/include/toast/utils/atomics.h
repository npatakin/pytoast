#ifndef PYTOAST_ATOMICS_H
#define PYTOAST_ATOMICS_H

#include <atomic>
#include <cstdint>
#include <cstring>

#include "vector_math.h"   // for HOSTDEVICE


HOSTDEVICE void toast_atomic_add(float* dst, float value) {
#if defined(__CUDA_ARCH__)
    atomicAdd(dst, value);
#else
    auto* bits = reinterpret_cast<std::atomic<uint32_t>*>(dst);
    uint32_t old_bits = bits->load(std::memory_order_relaxed);
    uint32_t new_bits;
    do {
        float old_value;
        std::memcpy(&old_value, &old_bits, sizeof(float));
        const float new_value = old_value + value;
        std::memcpy(&new_bits, &new_value, sizeof(float));
        // compare_exchange_weak refreshes old_bits on failure
    } while (!bits->compare_exchange_weak(old_bits, new_bits,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed));
#endif
}


HOSTDEVICE void toast_atomic_add(double* dst, double value) {
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 600
#error "pytoast needs compute capability 6.0 or newer: double-precision atomicAdd \
is not available before sm_60. Set TORCH_CUDA_ARCH_LIST to 6.0+ (the double \
kernels are compiled unconditionally by AT_DISPATCH_FLOATING_TYPES)."
#endif
    atomicAdd(dst, value);
#else
    auto* bits = reinterpret_cast<std::atomic<uint64_t>*>(dst);
    uint64_t old_bits = bits->load(std::memory_order_relaxed);
    uint64_t new_bits;
    do {
        double old_value;
        std::memcpy(&old_value, &old_bits, sizeof(double));
        const double new_value = old_value + value;
        std::memcpy(&new_bits, &new_value, sizeof(double));
    } while (!bits->compare_exchange_weak(old_bits, new_bits,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed));
#endif
}


static_assert(sizeof(std::atomic<uint32_t>) == sizeof(float),
              "atomic<uint32_t> must be layout-compatible with float");
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(double),
              "atomic<uint64_t> must be layout-compatible with double");

#endif // PYTOAST_ATOMICS_H
