#ifndef PYTOAST_LAUNCH_HELPERS_V2_H
#define PYTOAST_LAUNCH_HELPERS_V2_H

#include <tuple>
#include <utility>
#include <type_traits>

#include <ATen/Parallel.h>

#include <toast/utils/vector_math.h>

#ifdef TOAST_WITH_CUDA
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>
#define CULAMBDA __host__ __device__
#else
#define CULAMBDA
#endif

#define CUDA_DEFAULT_NUM_THREADS 1024


template <typename T>
T* data_ptr(const torch::Tensor& tensor) {
    return (T*)tensor.data_ptr();
}

#ifdef TOAST_WITH_CUDA

template <typename F>
__global__ void linear_kernel(int num_elements, F f) {
    uint element_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_idx >= num_elements) {return;}
    f(element_idx);
}

template <int threads = CUDA_DEFAULT_NUM_THREADS, typename F, typename ...Args>
void launch_kernel_cuda(
    uint num_elements,
    const torch::Device& device,
    const F& f,
    const Args&... args
) {
    // A zero-element launch computes a grid of 0 blocks, which CUDA rejects with
    // "invalid configuration argument".  Empty batches are legitimate (they fall
    // out of filtering), so there is simply nothing to do.
    if (num_elements == 0) { return; }
    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint num_blocks = (num_elements + threads - 1) / threads;
    f<<<num_blocks, threads, 0, stream>>>(args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int threads = CUDA_DEFAULT_NUM_THREADS, typename F>
void launch_linear_kernel_cuda(uint num_points, const torch::Device& device, F f) {
    // see launch_kernel_cuda: a 0-block grid is an invalid launch configuration
    if (num_points == 0) { return; }
    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks((num_points + threads - 1) / threads, 1, 1);
    linear_kernel<<<blocks, threads, 0, stream>>>(num_points, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif // TOAST_WITH_CUDA


#define TOAST_NO_CUDA_SUPPORT_MSG \
    "pytoast was built without CUDA support (CPU-only build); this operation " \
    "does not accept CUDA tensors. Rebuild with the CUDA toolkit available, " \
    "or move your tensors to the CPU."


template <int cuda_threads = CUDA_DEFAULT_NUM_THREADS, typename F>
void launch_linear_kernel(uint num_points, const torch::Device& device, F f) {
    if (device.is_cpu()) {
        at::parallel_for(0, num_points, 32768, [&](int64_t begin, int64_t end) {
            // printf("parallel_for: [%ld; %ld) size=%ld\n", begin, end, end - begin); fflush(stdout);
            for (uint i = begin; i < end; ++i) {
                f(i);
            }
        });
    } else {
#ifdef TOAST_WITH_CUDA
        launch_linear_kernel_cuda<cuda_threads>(num_points, device, f);
#else
        TORCH_CHECK(false, TOAST_NO_CUDA_SUPPORT_MSG);
#endif
    }
}

template <typename Ptr>
auto byte_add(Ptr p, int64_t stride) {
    using T = std::remove_pointer_t<Ptr>;

    if constexpr (std::is_const_v<T>) {
        auto cp = reinterpret_cast<const char*>(p);
        cp += stride;
        return reinterpret_cast<Ptr>(cp);
    } else {
        auto cp = reinterpret_cast<char*>(p);
        cp += stride;
        return reinterpret_cast<Ptr>(cp);
    }
}

template <typename Tuple, std::size_t... Is>
void add_strides_bytes_impl(Tuple& t, const int64_t* strides, std::index_sequence<Is...>) {
    ((std::get<Is>(t) = byte_add(std::get<Is>(t), strides[Is])), ...);
}

template <typename... Ptrs>
void add_strides_bytes(std::tuple<Ptrs...>& t, const int64_t* strides) {
    static_assert((std::is_pointer_v<Ptrs> && ...),
                  "All tuple elements must be pointers");

    add_strides_bytes_impl(t, strides, std::index_sequence_for<Ptrs...>{});
}


// Implementation helper
template <typename... Ts, std::size_t... Is>
auto map_pointers_cast_impl(char** data, std::index_sequence<Is...>) {
    return std::make_tuple(
        reinterpret_cast<Ts>(data[Is])...
    );
}

// Main function
template <typename... Ts>
auto map_pointers_cast(char** data) {
    return map_pointers_cast_impl<Ts...>(
        data,
        std::index_sequence_for<Ts...>{}
    );
}


#ifdef TOAST_WITH_CUDA

template <uint num_args>
auto make_offset_calc(const at::TensorIterator& iter){
    auto offset_calc =
        make_element_offset_calculator<num_args>(iter);
    for (uint op_idx = 0; op_idx < num_args; ++op_idx) {
        auto last_dim = iter.tensor(op_idx).size(-1);
        for (uint i = 0; i != offset_calc.dims; ++i) {
            offset_calc.strides_[i][op_idx] /= last_dim;
            // printf("op=%d dim=%d stride=%d\n", op_idx, i, offset_calc.strides_[i][op_idx]);
        }
    }
    return offset_calc;
}

#endif // TOAST_WITH_CUDA


template <typename Tuple>
constexpr std::size_t tuple_length_v = std::tuple_size_v<std::remove_reference_t<Tuple>>;


template <std::size_t N>
struct zero_offsets {
    static constexpr std::size_t size = N;

    CULAMBDA constexpr std::size_t operator[](std::size_t) const noexcept {
        return 0;
    }
};

struct same_offsets {
    uint offset;

    CULAMBDA uint operator[](const uint&) const noexcept {
        return offset;
    }
};


inline at::TensorIterator make_tensor_iterator(
    const std::vector<at::Tensor>& outputs,
    const std::vector<at::Tensor>& inputs,
    std::vector<int64_t> squash_dims = {-1},
    bool check_same_dtype = true,
    bool check_mem_overlap = true
) {
    const int64_t ndim = outputs[0].dim();

    auto check_tensor = [ndim](const at::Tensor& t, const char* role, int idx) {
        TORCH_CHECK(t.dim() == ndim,
            role, "[", idx, "] has dim=", t.dim(), " but outputs[0] has dim=", ndim);
        TORCH_CHECK(t.stride(-1) == 1,
            role, "[", idx, "] must have stride 1 in last dimension, got ", t.stride(-1));
    };

    for (int i = 0; i < outputs.size(); ++i) check_tensor(outputs[i], "output", i);
    for (int i = 0; i < inputs.size();  ++i) check_tensor(inputs[i],  "input",  i);

    for (auto& d : squash_dims) {
        if (d < 0) d += ndim;
    }

    at::TensorIteratorConfig config;
    config.check_all_same_device(true)
        .check_all_same_dtype(check_same_dtype)
        .resize_outputs(false)
        .set_check_mem_overlap(check_mem_overlap)
        .declare_static_shape(outputs[0].sizes(), squash_dims);

    for (const auto& output : outputs) {
        config.add_output(output);
    }
    for (const auto& input: inputs) {
        config.add_input(input);
    }
    return config.build();
}


template <typename Iter, typename... Ts, std::size_t... Is>
auto tensor_ptrs_impl(Iter& iter, std::index_sequence<Is...>) {
    return std::make_tuple(
        static_cast<Ts>(iter.tensor(Is).data_ptr())...
    );
}

// Main function
template <typename... Ts, typename Iter>
auto tensor_ptrs(Iter& iter) {
    static_assert((std::is_pointer_v<Ts> && ...),
                  "All Ts must be pointer types");

    return tensor_ptrs_impl<Iter, Ts...>(
        iter,
        std::index_sequence_for<Ts...>{}
    );
}

inline bool all_contiguous(const at::TensorIterator& iter) {
    bool result = true;
    for (uint idx = 0; idx < iter.ntensors(); ++idx) {
        result &= iter.tensor(idx).is_contiguous();
    }
    return result;
}


#define TOAST_DEFAULT_GRAIN at::internal::GRAIN_SIZE

#define TOAST_KERNEL_CPU_BODY_GRAIN(CAPTURE_LIST, GRAIN, ...) \
        iter.for_each(CAPTURE_LIST(char** data, const int64_t* strides, int64_t n) {\
            auto ptrs = map_pointers_cast<PtrsType...>(data);\
            for (int64_t idx = 0; idx < n; ++idx) {\
                constexpr zero_offsets<tuple_length_v<decltype(ptrs)>> offsets{};\
                __VA_ARGS__\
                add_strides_bytes(ptrs, strides);\
            }\
        }, GRAIN);

#define TOAST_KERNEL_CPU_BODY(CAPTURE_LIST, ...) \
        iter.for_each(CAPTURE_LIST(char** data, const int64_t* strides, int64_t n) {\
            auto ptrs = map_pointers_cast<PtrsType...>(data);\
            for (int64_t idx = 0; idx < n; ++idx) {\
                constexpr zero_offsets<tuple_length_v<decltype(ptrs)>> offsets{};\
                __VA_ARGS__\
                add_strides_bytes(ptrs, strides);\
            }\
        });


#define TOAST_KERNEL_NO_CUDA_BODY \
        (void)num_elements; (void)ctg;\
        TORCH_CHECK(false, TOAST_NO_CUDA_SUPPORT_MSG);


#ifdef TOAST_WITH_CUDA

#define DEFINE_KERNEL(kernel_name, ...)  \
template <int Threads, typename ...PtrsType> \
void kernel_name(at::TensorIterator& iter) { \
    const auto num_elements = iter.num_output_elements(); \
    const bool ctg = all_contiguous(iter); \
\
    if (iter.device().is_cpu()) {\
        TOAST_KERNEL_CPU_BODY([], __VA_ARGS__)\
    } else {\
        const auto ptrs = tensor_ptrs<PtrsType...>(iter);\
        if (!ctg) {\
            auto offset_calc = make_offset_calc<tuple_length_v<decltype(ptrs)>>(iter);\
            launch_linear_kernel_cuda<Threads>(num_elements, iter.device(), [ptrs, offset_calc] CULAMBDA (uint idx) {\
                const auto offsets = offset_calc.get(idx);\
                __VA_ARGS__\
            });\
        } else {\
            launch_linear_kernel_cuda<Threads>(num_elements, iter.device(), [ptrs] CULAMBDA (uint idx) {\
                const same_offsets offsets{idx};\
                __VA_ARGS__\
            });\
        }\
    }\
}

#else

#define DEFINE_KERNEL(kernel_name, ...)  \
template <int Threads, typename ...PtrsType> \
void kernel_name(at::TensorIterator& iter) { \
    const auto num_elements = iter.num_output_elements(); \
    const bool ctg = all_contiguous(iter); \
\
    if (iter.device().is_cpu()) {\
        TOAST_KERNEL_CPU_BODY([], __VA_ARGS__)\
    } else {\
        TOAST_KERNEL_NO_CUDA_BODY\
    }\
}

#endif // TOAST_WITH_CUDA


#define DISPATCH_KERNEL_THREADS(name, float_threads, double_threads, ...)\
    AT_DISPATCH_FLOATING_TYPES(iter.ninputs() ? iter.input_dtype() : iter.dtype(), #name, [&]() {\
        using dtype = dtype_t<scalar_t>; \
        using vec2 = dtype::vec2; \
        using vec3 = dtype::vec3; \
        using vec4 = dtype::vec4; \
        using quat_t = quat<dtype>; \
        using se3_t = se3<dtype>; \
        constexpr int threads = std::is_same_v<scalar_t, float> ? float_threads : double_threads; \
        name<threads, __VA_ARGS__>(iter); \
    });

#define DISPATCH_KERNEL(name, ...)\
    DISPATCH_KERNEL_THREADS(name, CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS, __VA_ARGS__)



#ifdef TOAST_WITH_CUDA

#define DEFINE_KERNEL_CAPTURE(kernel_name, ...)  \
template <int Threads, int Grain, typename CapturesTuple, typename ...PtrsType> \
void kernel_name(at::TensorIterator& iter, CapturesTuple captures) { \
    const auto num_elements = iter.num_output_elements(); \
    const bool ctg = all_contiguous(iter); \
\
    if (iter.device().is_cpu()) {\
        TOAST_KERNEL_CPU_BODY_GRAIN([captures], Grain, __VA_ARGS__)\
    } else {\
        const auto ptrs = tensor_ptrs<PtrsType...>(iter);\
        if (!ctg) {\
            auto offset_calc = make_offset_calc<tuple_length_v<decltype(ptrs)>>(iter);\
            launch_linear_kernel_cuda<Threads>(num_elements, iter.device(), [ptrs, offset_calc, captures] CULAMBDA (uint idx) {\
                const auto offsets = offset_calc.get(idx);\
                __VA_ARGS__\
            });\
        } else {\
            launch_linear_kernel_cuda<Threads>(num_elements, iter.device(), [ptrs, captures] CULAMBDA (uint idx) {\
                const same_offsets offsets{idx};\
                __VA_ARGS__\
            });\
        }\
    }\
}

#else

#define DEFINE_KERNEL_CAPTURE(kernel_name, ...)  \
template <int Threads, int Grain, typename CapturesTuple, typename ...PtrsType> \
void kernel_name(at::TensorIterator& iter, CapturesTuple captures) { \
    const auto num_elements = iter.num_output_elements(); \
    const bool ctg = all_contiguous(iter); \
\
    if (iter.device().is_cpu()) {\
        TOAST_KERNEL_CPU_BODY_GRAIN([captures], Grain, __VA_ARGS__)\
    } else {\
        TOAST_KERNEL_NO_CUDA_BODY\
    }\
}

#endif // TOAST_WITH_CUDA

#define DISPATCH_KERNEL_THREADS_CAPTURE(name, float_threads, double_threads, captures_expr, ...)\
    AT_DISPATCH_FLOATING_TYPES(iter.ninputs() ? iter.input_dtype() : iter.dtype(), #name, [&]() {\
        using dtype = dtype_t<scalar_t>; \
        using vec2 = typename dtype::vec2; \
        using vec3 = typename dtype::vec3; \
        using vec4 = typename dtype::vec4; \
        using quat_t = quat<dtype>; \
        using se3_t = se3<dtype>; \
        constexpr int threads = std::is_same_v<scalar_t, float> ? float_threads : double_threads; \
        auto _captures = (captures_expr); \
        name<threads, TOAST_DEFAULT_GRAIN, decltype(_captures), __VA_ARGS__>(iter, _captures); \
    });


#define DISPATCH_KERNEL_CAPTURE(name, captures_expr, ...)\
    DISPATCH_KERNEL_THREADS_CAPTURE(name, CUDA_DEFAULT_NUM_THREADS, CUDA_DEFAULT_NUM_THREADS, captures_expr, __VA_ARGS__)


#define CLS(ptr) std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>


#define INLINE_KERNEL(num_elements, device, ...)\
    if (device.is_cpu()) {\
        at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {\
        for (int64_t idx = begin; idx < end; ++idx) {\
            __VA_ARGS__\
        }\
        });\
    } else {\
        launch_linear_kernel(num_elements, device, [data] CULAMBDA (uint idx) {\
            __VA_ARGS__\
        });\
    }



#endif //PYTOAST_LAUNCH_HELPERS_V2_H
